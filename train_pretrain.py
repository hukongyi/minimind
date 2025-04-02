# 导入标准库
import os  # 用于与操作系统交互，例如读取环境变量、创建目录
import platform  # 用于获取操作系统平台信息 (虽然在此脚本中未直接使用其返回值)
import argparse  # 用于解析命令行参数
import time  # 用于计时，例如计算训练时长
import math  # 用于数学运算，例如计算余弦学习率
import warnings  # 用于控制警告信息的显示

# 导入第三方库
import pandas as pd  # 数据处理库 (虽然在此脚本中未直接使用，可能在导入的模块中使用或为未来扩展保留)
import torch  # PyTorch核心库，用于张量计算和神经网络
import torch.distributed as dist  # PyTorch分布式训练库
from torch import optim, nn  # PyTorch优化器 (optim) 和神经网络模块 (nn)
from torch.nn.parallel import DistributedDataParallel  # 用于分布式数据并行训练的包装器
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
)  # 余弦退火学习率调度器 (虽然未使用，但导入了)
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器和分布式采样器
from contextlib import nullcontext  # 用于创建空上下文管理器，方便在不同设备类型间切换

# 导入Hugging Face Transformers库
from transformers import AutoTokenizer  # 用于自动加载预训练模型的分词器

# 导入自定义模块
from model.model import MiniMindLM  # 导入自定义的模型类 MiniMindLM
from model.LMConfig import LMConfig  # 导入自定义的模型配置类 LMConfig
from model.dataset import PretrainDataset  # 导入自定义的数据集类 PretrainDataset

# 忽略所有警告信息，使输出更简洁
warnings.filterwarnings("ignore")


# 定义一个日志记录函数
def Logger(content):
    """
    打印日志信息。在分布式训练 (DDP) 模式下，只在主进程 (rank 0) 上打印，
    以避免多进程重复输出。
    Args:
        content: 要打印的内容。
    """
    # 检查是否处于DDP模式，或者当前进程是否是主进程 (rank 0)
    if not ddp or dist.get_rank() == 0:
        # 如果不是DDP模式，或者当前是主进程，则打印内容
        print(content)


# 定义一个计算学习率的函数
def get_lr(current_step, total_steps, lr):
    """
    计算当前步骤的学习率，使用带有预热偏移的余弦退火策略。
    公式: lr / 10 + 0.5 * lr * (1 + cos(pi * current_step / total_steps))
    Args:
        current_step: 当前训练的总步数 (跨所有epoch)。
        total_steps: 训练过程中的总步数。
        lr: 基础学习率 (命令行参数指定)。
    Returns:
        float: 当前步骤计算得到的学习率。
    """
    # 应用余弦退火公式，加上一个小的初始值 (lr/10)
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


# 定义单个epoch的训练函数
def train_epoch(epoch, wandb):
    """
    执行一个完整的训练周期 (epoch)。
    Args:
        epoch (int): 当前的epoch编号 (从0开始)。
        wandb: Weights & Biases 的 run 对象，用于日志记录，如果未使用则为 None。
    """
    # 定义损失函数：交叉熵损失。reduction='none' 表示不立即对损失进行求和或平均，
    # 以便后续可以应用 loss_mask。
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    # 记录epoch开始时间
    start_time = time.time()
    # 遍历训练数据加载器中的每个批次
    # train_loader 返回 (输入X, 目标Y, 损失掩码loss_mask)
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将输入数据、目标数据和损失掩码移动到指定的计算设备 (GPU或CPU)
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 计算当前步骤的学习率
        # 总步数 = 当前epoch * 每epoch的迭代次数 + 当前epoch内的步数
        # 总训练步数 = 总epoch数 * 每epoch的迭代次数
        lr = get_lr(
            epoch * iter_per_epoch + step,
            args.epochs * iter_per_epoch,
            args.learning_rate,
        )
        # 更新优化器中每个参数组的学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 使用自动混合精度 (AMP) 上下文管理器。
        # 如果设备是CPU，ctx 是 nullcontext，不起作用。
        # 如果设备是CUDA，ctx 是 torch.cuda.amp.autocast()，启用混合精度计算。
        with ctx:
            # 模型前向传播，获取输出结果 (包含logits和可能的辅助损失)
            res = model(X)
            # 计算原始的交叉熵损失
            # res.logits 的形状通常是 (batch_size, seq_len, vocab_size)
            # Y 的形状通常是 (batch_size, seq_len)
            # view(-1, res.logits.size(-1)) 将 logits 展平成 (batch_size * seq_len, vocab_size)
            # Y.view(-1) 将目标展平成 (batch_size * seq_len)
            # loss_fct 计算每个token的损失，形状恢复为 (batch_size, seq_len)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(
                Y.size()
            )
            # 应用损失掩码：只计算未被掩码的token的损失
            # (loss * loss_mask).sum() 计算所有有效token的总损失
            # loss_mask.sum() 计算有效token的数量
            # 两者相除得到有效token的平均损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            # 加上模型返回的辅助损失 (例如 MoE 的负载均衡损失)
            loss += res.aux_loss
            # 应用梯度累积：将当前批次的损失除以累积步数
            # 这是因为我们希望反向传播的是累积了多个小批次后的平均梯度
            loss = loss / args.accumulation_steps

        # 使用 GradScaler 进行损失缩放和反向传播
        # scaler.scale(loss) 缩放损失值，防止混合精度训练中的梯度下溢
        # .backward() 计算缩放后的梯度
        scaler.scale(loss).backward()

        # 检查是否达到了梯度累积的步数
        if (step + 1) % args.accumulation_steps == 0:
            # 在进行梯度裁剪和优化器步骤之前，需要先将梯度取消缩放
            scaler.unscale_(optimizer)
            # 对模型的参数进行梯度裁剪，防止梯度爆炸
            # args.grad_clip 是梯度的最大范数
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器步骤 (更新模型参数)
            # scaler.step 会自动检查梯度是否包含 inf 或 NaN，并相应地更新参数
            scaler.step(optimizer)
            # 更新 GradScaler 的缩放因子，为下一次迭代做准备
            scaler.update()

            # 清空优化器的梯度。set_to_none=True 可以稍微提高内存效率
            optimizer.zero_grad(set_to_none=True)

        # 检查是否达到了日志记录的间隔
        if step % args.log_interval == 0:
            # 计算从epoch开始到当前步所花费的时间
            spend_time = time.time() - start_time
            # 使用 Logger 函数打印训练信息 (只在主进程打印)
            Logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:".format(
                    epoch + 1,  # 当前epoch (从1开始计数)
                    args.epochs,  # 总epoch数
                    step,  # 当前epoch内的步数
                    iter_per_epoch,  # 每epoch的总步数
                    loss.item()
                    * args.accumulation_steps,  # 当前步的损失值 (乘以累积步数以反映累积前的损失大小)
                    optimizer.param_groups[-1][
                        "lr"
                    ],  # 当前学习率 (取最后一个参数组的LR)
                    # 估算整个epoch的剩余时间 (分钟)
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )

            # 如果启用了 wandb 并且当前是主进程
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                # 使用 wandb 记录损失、学习率和预计的epoch时间
                wandb.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60
                        - spend_time // 60,
                    }
                )

        # 检查是否达到了模型保存的间隔，并且当前是主进程
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # 将模型设置为评估模式 (例如关闭 dropout)
            model.eval()
            # 根据配置确定模型保存路径中是否包含 '_moe' 后缀
            moe_path = "_moe" if lm_config.use_moe else ""
            # 构建检查点文件的完整路径
            ckp = f"{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth"

            # 获取模型的状态字典 (state_dict)
            # 如果模型是 DDP 包装过的，需要访问 .module 属性获取原始模型
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                # 否则直接获取状态字典
                state_dict = model.state_dict()

            # 保存模型状态字典到文件
            torch.save(state_dict, ckp)
            # 将模型设置回训练模式
            model.train()


# 定义初始化模型和分词器的函数
def init_model(lm_config):
    """
    初始化模型和分词器。
    Args:
        lm_config (LMConfig): 模型的配置对象。
    Returns:
        tuple: 包含初始化后的模型 (model) 和分词器 (tokenizer) 的元组。
    """
    # 从指定路径加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained("./model/minimind_tokenizer")
    # 使用配置实例化 MiniMindLM 模型，并将其移动到指定的计算设备
    model = MiniMindLM(lm_config).to(args.device)
    # 使用 Logger 打印模型总参数量 (只计算需要梯度的参数)
    Logger(
        f"LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    # 返回模型和分词器
    return model, tokenizer


# 定义初始化分布式训练环境的函数
def init_distributed_mode():
    """
    初始化 PyTorch 分布式训练环境 (DDP)。
    """
    # 如果全局变量 ddp 为 False，则不执行任何操作并返回
    if not ddp:
        return
    # 声明将要修改的全局变量
    global ddp_local_rank, DEVICE

    # 初始化进程组，指定后端为 "nccl" (通常用于 NVIDIA GPU)
    dist.init_process_group(backend="nccl")
    # 从环境变量 "RANK" 获取当前进程的全局排名
    ddp_rank = int(os.environ["RANK"])
    # 从环境变量 "LOCAL_RANK" 获取当前进程在节点内的本地排名
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # 从环境变量 "WORLD_SIZE" 获取总的进程数
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    # 根据本地排名设置当前进程使用的设备 (例如 "cuda:0", "cuda:1")
    DEVICE = f"cuda:{ddp_local_rank}"
    # 设置当前进程默认使用的 CUDA 设备
    torch.cuda.set_device(DEVICE)


# 主程序入口
# 使用 torchrun 启动脚本时 (例如: torchrun --nproc_per_node 2 1-pretrain.py)，
# 会自动设置必要的环境变量 (RANK, LOCAL_RANK, WORLD_SIZE)
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # 添加各种命令行参数
    parser.add_argument(
        "--out_dir", type=str, default="out", help="输出目录，用于保存模型等"
    )
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。 (来自原始注释)
    parser.add_argument("--epochs", type=int, default=1, help="训练的总轮数 (epochs)")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="每个设备上的批处理大小"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="基础学习率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="计算设备 (例如 'cuda:0' 或 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="混合精度计算的数据类型 ('float16', 'bfloat16', 或 'float32')",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="是否使用 Weights & Biases 进行日志记录",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MiniMind-Pretrain",
        help="Weights & Biases 项目名称",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="数据加载器使用的工作进程数"
    )
    parser.add_argument(
        "--ddp", action="store_true", help="是否启用分布式数据并行 (DDP) 训练"
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="梯度累积的步数"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="梯度裁剪的最大范数"
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=0,
        help="学习率预热的迭代次数 (当前未使用 get_lr 函数实现)",
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="打印日志的间隔步数"
    )
    parser.add_argument(
        "--save_interval", type=int, default=100, help="保存模型检查点的间隔步数"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="本地排名 (由 torchrun 自动设置)"
    )  # DDP 需要
    parser.add_argument("--dim", default=768, type=int, help="模型维度")
    parser.add_argument("--n_layers", default=16, type=int, help="模型层数")
    parser.add_argument("--max_seq_len", default=512, type=int, help="最大序列长度")
    parser.add_argument(
        "--use_moe", default=False, type=bool, help="是否使用混合专家 (MoE)"
    )  # 注意：bool 类型参数通常用 action='store_true' 或 'store_false'
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/pretrain_hq.jsonl",
        help="预训练数据文件路径",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 根据命令行参数创建模型配置对象
    lm_config = LMConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe,
    )
    # 设置模型保存目录
    args.save_dir = os.path.join(args.out_dir)
    # 创建保存目录 (如果不存在)
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建输出目录 (如果不存在)
    os.makedirs(args.out_dir, exist_ok=True)
    # 计算每次迭代处理的总 token 数 (用于参考，未在后续代码中使用)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    # 设置 PyTorch 的随机种子以保证可复现性
    torch.manual_seed(1337)
    # 判断设备类型是 "cuda" 还是 "cpu"
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 构建 Weights & Biases 的运行名称
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置自动混合精度 (AMP) 的上下文管理器
    # 如果是 CPU，使用 nullcontext (无操作)
    # 如果是 CUDA，使用 torch.cuda.amp.autocast() 启用 AMP
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 检测是否为 DDP 运行模式
    # 如果环境变量 "RANK" 被设置 (通常由 torchrun 设置)，则 ddp 为 True
    ddp = int(os.environ.get("RANK", -1)) != -1
    # 初始化 DDP 相关变量的默认值 (用于非 DDP 或主进程)
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 如果是 DDP 模式
    if ddp:
        # 初始化分布式环境
        init_distributed_mode()
        # 更新 args.device 为当前 DDP 进程分配的设备
        args.device = torch.device(DEVICE)

    # 如果启用了 wandb 并且当前是主进程 (或者非 DDP 模式)
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        # 导入 wandb 库
        import wandb

        # 初始化 wandb run
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        # 如果不使用 wandb 或不是主进程，则将 wandb 设置为 None
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)
    # 创建预训练数据集实例
    train_ds = PretrainDataset(
        args.data_path, tokenizer, max_length=lm_config.max_seq_len
    )
    # 创建分布式采样器 (如果 DDP 启用) 或设为 None
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,  # 数据集
        batch_size=args.batch_size,  # 批处理大小
        pin_memory=True,  # 锁页内存，可能加速 CPU 到 GPU 的数据传输
        drop_last=False,  # 不丢弃最后一个不完整的批次
        shuffle=False,  # 不打乱数据顺序 (因为使用了 DistributedSampler，它会处理 shuffle)
        num_workers=args.num_workers,  # 加载数据的子进程数量
        sampler=train_sampler,  # 数据采样器 (DDP 时使用 DistributedSampler)
    )

    # 初始化 GradScaler，用于混合精度训练
    # enabled 参数根据命令行指定的 dtype 是否为 'float16' 或 'bfloat16' 来决定是否启用
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    # 初始化 AdamW 优化器，传入模型参数和学习率
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 如果是 DDP 模式
    if ddp:
        # 指定在 DDP 同步梯度时要忽略的参数或缓冲区 (这里是 "pos_cis"，可能是位置编码相关的)
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # 使用 DistributedDataParallel 包装模型，指定当前进程的设备 ID
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个 epoch 的迭代次数 (批次数)
    iter_per_epoch = len(train_loader)
    # 开始训练循环，遍历指定的 epoch 数
    for epoch in range(args.epochs):
        # 调用 train_epoch 函数执行当前 epoch 的训练
        train_epoch(epoch, wandb)
