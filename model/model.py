import math  # 导入数学库，用于数学运算，例如在注意力机制中计算平方根
import struct  # 通常用于处理二进制数据，在此代码中可能未使用，但被导入
import inspect  # 用于获取对象信息（例如函数签名），在此代码中可能未使用，但被导入
import time  # 导入时间库，可能用于性能分析或计时，在此代码中可能未使用，但被导入

from .LMConfig import LMConfig  # 从同一目录下的 LMConfig 文件导入 LMConfig 类，用于模型配置
from typing import Any, Optional, Tuple, List  # 导入类型提示，增强代码可读性和健壮性
import numpy as np  # 导入 NumPy 库，用于数值计算，虽然这里主要使用 PyTorch，但可能在某些辅助环节使用
import torch  # 导入 PyTorch 核心库
import torch.nn.functional as F  # 导入 PyTorch 神经网络函数库，例如 softmax, silu 等
from torch import nn  # 导入 PyTorch 神经网络模块基类
from transformers import PreTrainedModel  # 从 Hugging Face Transformers 库导入预训练模型基类，方便集成和使用 Transformers 生态
from transformers.modeling_outputs import CausalLMOutputWithPast  # 导入用于存储因果语言模型输出的标准格式，包含 logits 和 past_key_values


# 定义 RMSNorm (Root Mean Square Layer Normalization) 类
class RMSNorm(torch.nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization)。
    一种比 LayerNorm 更简单的归一化技术，常用于 Llama 等模型。
    它通过输入的均方根来缩放输入，并乘以一个可学习的权重。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化 RMSNorm 层。
        Args:
            dim (int): 输入特征的维度。
            eps (float): 一个小的常数，加到分母中以防止除以零，提高数值稳定性。默认为 1e-6。
        """
        super().__init__()
        self.eps = eps  # 存储 epsilon 值
        # 定义一个可学习的缩放参数（权重），初始化为全1张量
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        执行 RMS 归一化计算的核心逻辑。
        Args:
            x (torch.Tensor): 输入张量。
        Returns:
            torch.Tensor: 归一化后的张量（未乘以权重）。
        """
        # 计算输入张量 x 在最后一个维度上的均方值 (x^2 的均值)
        # keepdim=True 保持维度以便广播
        # torch.rsqrt 计算平方根的倒数 (1 / sqrt(value))
        # 乘以 x 实现归一化：x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        RMSNorm 层的前向传播。
        Args:
            x (torch.Tensor): 输入张量。
        Returns:
            torch.Tensor: 经过 RMSNorm 处理后的输出张量。
        """
        # 1. 将输入 x 转换为 float 类型进行归一化计算，以获得更高精度
        # 2. 调用 _norm 方法进行核心归一化
        # 3. 将归一化结果乘以可学习的权重 self.weight
        # 4. 使用 .type_as(x) 将结果转换回输入 x 的原始数据类型（例如 float16）
        return self.weight * self._norm(x.float()).type_as(x)


# 预计算旋转位置编码 (Rotary Positional Embeddings, RoPE) 的复数表示
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预先计算旋转位置编码所需的复数 (cisoid) 值。
    RoPE 通过将位置信息编码为旋转矩阵来注入序列中的位置信息。
    Args:
        dim (int): 每个注意力头的维度的一半 (dim // 2)。RoPE 应用于成对的维度。
        end (int): 预计算的最大序列长度。默认为 32k。
        theta (float): RoPE 中的基准频率参数。默认为 1e6 (1,000,000)，Llama 3 使用。
    Returns:
        torch.Tensor: 预计算的位置编码复数张量，形状为 (end, dim // 2)，数据类型为 complex64。
    """
    # 计算频率：freqs = 1.0 / (theta^(2k / dim)) for k in [0, 1, ..., dim/2 - 1]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 创建序列位置索引张量 t: [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)  # type: ignore # 确保 t 和 freqs 在同一设备上
    # 计算每个位置和每个频率的外积，得到相位角矩阵 freqs_out = t @ freqs.T
    # 形状为 (end, dim // 2)
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 使用 torch.polar 将相位角转换为复数形式 (cos(theta) + i*sin(theta))
    # 幅度为 1 (torch.ones_like(freqs))，相位为 freqs
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


# 应用旋转位置编码
def apply_rotary_emb(xq, xk, pos_cis):
    """
    将预计算的旋转位置编码 (pos_cis) 应用到查询 (xq) 和键 (xk) 张量上。
    Args:
        xq (torch.Tensor): 查询张量，形状通常为 (bs, seq_len, n_heads, head_dim)。
        xk (torch.Tensor): 键张量，形状通常为 (bs, seq_len, n_kv_heads, head_dim)。
        pos_cis (torch.Tensor): 预计算的 RoPE 复数张量，形状为 (seq_len, head_dim // 2)。
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 经过 RoPE 旋转后的查询和键张量。
    """
    # 定义一个内部辅助函数，用于调整 pos_cis 的形状以匹配 xq/xk 进行广播
    def unite_shape(pos_cis, x):
        """调整 pos_cis 形状以匹配 x 进行广播。"""
        ndim = x.ndim  # 获取输入张量 x 的维度数
        assert 0 <= 1 < ndim  # 确保 x 至少有 2 个维度
        # 检查 pos_cis 的形状是否符合预期 (seq_len, head_dim // 2)
        assert pos_cis.shape == (x.shape[1], x.shape[-1]) # x.shape[-1] 应该是 head_dim // 2 (因为 x 被 view_as_complex)
        # 创建一个新形状列表，只保留 x 的第 1 维 (seq_len) 和最后 1 维 (head_dim // 2)
        # 其他维度设为 1，以便广播
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        # 返回调整形状后的 pos_cis
        return pos_cis.view(*shape)

    # 将 xq 和 xk 的最后一个维度 (head_dim) 视为成对的实数，并转换为复数视图
    # 形状变为 (bs, seq_len, n_heads, head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 调整 pos_cis 的形状以匹配 xq_ (和 xk_)
    pos_cis = unite_shape(pos_cis, xq_)
    # 执行复数乘法，实现旋转
    # (a + ib) * (cos + isin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    # 将结果转回实数视图，并将最后两个维度展平回 head_dim
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    # 将输出转换回输入 xq 和 xk 的原始数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 重复 Key 和 Value 张量以支持 Grouped Query Attention (GQA) 或 Multi-Query Attention (MQA)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 Key 或 Value 张量的头重复 n_rep 次。
    用于 GQA/MQA，其中查询头的数量是键/值头数量的 n_rep 倍。
    等效于 torch.repeat_interleave(x, dim=2, repeats=n_rep)。
    Args:
        x (torch.Tensor): 输入的 Key 或 Value 张量，形状为 (bs, slen, n_kv_heads, head_dim)。
        n_rep (int): 重复次数 (n_heads // n_kv_heads)。
    Returns:
        torch.Tensor: 重复后的张量，形状为 (bs, slen, n_heads, head_dim)。
    """
    bs, slen, n_kv_heads, head_dim = x.shape  # 获取输入张量的形状
    if n_rep == 1:  # 如果重复次数为 1 (即 n_heads == n_kv_heads)，无需操作
        return x
    # 1. 在 n_kv_heads 维度后插入一个新的维度 (None)
    #    形状变为 (bs, slen, n_kv_heads, 1, head_dim)
    # 2. 使用 expand 将新维度扩展 n_rep 次，实现零内存拷贝的重复
    #    形状变为 (bs, slen, n_kv_heads, n_rep, head_dim)
    # 3. 使用 reshape 将 n_kv_heads 和 n_rep 维度合并，得到总的头数 (n_heads)
    #    最终形状为 (bs, slen, n_kv_heads * n_rep, head_dim)
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# 定义注意力机制模块
class Attention(nn.Module):
    """
    多头注意力机制模块，支持 Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
    以及可选的 Flash Attention 优化。
    """
    def __init__(self, args: LMConfig):
        """
        初始化 Attention 模块。
        Args:
            args (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        # 如果未指定 n_kv_heads，则默认为 n_heads (标准多头注意力)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数是 KV 头数的整数倍
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads  # 查询头的数量
        self.n_local_kv_heads = self.n_kv_heads  # 键/值头的数量
        # 计算重复因子，用于 GQA/MQA
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads  # 每个头的维度
        # 线性层：将输入映射到 Q, K, V 空间
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 线性层：将多头注意力输出映射回原始维度
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # Dropout 层，用于注意力权重和残差连接输出
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout  # 存储 dropout 率
        # 检查是否可以使用 Flash Attention (需要 PyTorch >= 2.0 且在配置中启用)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # 如果未使用 Flash Attention，打印警告 (代码中注释掉了)
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

        # 预计算因果注意力掩码 (causal mask)
        # 创建一个上三角矩阵，对角线以上填充 -inf，对角线及以下为 0 (通过 triu 实现)
        # 形状为 (1, 1, max_seq_len, max_seq_len) 以便广播
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        # 将 mask 注册为 buffer，它会随模型移动 (e.g., to device)，但不是可训练参数
        self.register_buffer("mask", mask, persistent=False) # persistent=False 表示不保存到 state_dict

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        """
        Attention 模块的前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状 (bsz, seq_len, dim)。
            pos_cis (torch.Tensor): 当前序列长度的 RoPE 复数张量，形状 (seq_len, head_dim // 2)。
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]):
                可选的 KV 缓存，包含上一时间步的 K 和 V 张量。
                形状为 ((bsz, past_seq_len, n_kv_heads, head_dim), (bsz, past_seq_len, n_kv_heads, head_dim))。
            use_cache (bool): 是否使用并返回 KV 缓存，用于推理加速。
        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                注意力输出张量 (bsz, seq_len, dim) 和更新后的 KV 缓存 (如果 use_cache=True)。
        """
        bsz, seq_len, _ = x.shape  # 获取输入形状：批大小、序列长度、维度

        # 1. 计算 Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. 重塑 Q, K, V 以分离头
        # xq: (bsz, seq_len, n_local_heads, head_dim)
        # xk: (bsz, seq_len, n_local_kv_heads, head_dim)
        # xv: (bsz, seq_len, n_local_kv_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 3. 应用旋转位置编码 (RoPE)
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 4. KV 缓存机制 (用于推理)
        if past_key_value is not None:
            # 如果存在过去的 K, V，将当前的 K, V 拼接到过去的 K, V 后面 (沿序列长度维度)
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 如果需要使用缓存，则存储更新后的 K, V
        past_kv = (xk, xv) if use_cache else None

        # 5. 准备用于注意力计算的 Q, K, V
        # a. 转置 Q, K, V，将头维度提前：(bsz, n_heads, seq_len, head_dim)
        # b. 对 K 和 V 应用 repeat_kv 以匹配 Q 的头数 (如果是 GQA/MQA)
        xq = xq.transpose(1, 2)
        key = repeat_kv(xk, self.n_rep).transpose(1, 2)
        value = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 6. 计算注意力输出
        # 检查是否使用 Flash Attention 优化 (且当前序列长度不为 1，因为 Flash Attention 对单 token 生成优化不大)
        if self.flash and seq_len != 1:
            # 设置 dropout 概率 (训练时使用配置值，推理时为 0)
            dropout_p = self.dropout if self.training else 0.0
            # 使用 PyTorch 内置的 scaled_dot_product_attention
            # is_causal=True 会自动应用因果掩码
            output = F.scaled_dot_product_attention(
                xq, key, value,
                attn_mask=None,  # 因果掩码由 is_causal 处理
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            # 标准的缩放点积注意力计算
            # a. 计算 Q 和 K 的点积相似度分数: (bsz, n_heads, seq_len, head_dim) @ (bsz, n_heads, head_dim, kv_seq_len) -> (bsz, n_heads, seq_len, kv_seq_len)
            #    kv_seq_len 是 K/V 的序列长度 (可能包含缓存)
            scores = (xq @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # b. 添加因果掩码 (只取当前序列长度对应的部分)
            #    self.mask 的形状是 (1, 1, max_len, max_len)，通过广播应用
            #    [:seq_len, :seq_len] 确保只对当前输入的有效部分应用掩码
            #    注意：这里的 seq_len 是当前输入的长度，而 key 的长度可能是 past_seq_len + seq_len
            #    掩码确保 query 的每个位置只能关注到 key/value 的对应位置及之前的位置
            current_kv_seq_len = key.shape[-2] # 获取 K 的实际序列长度 (可能包含 past_kv)
            scores += self.mask[:, :, :seq_len, :current_kv_seq_len]
            # c. 对分数应用 Softmax，得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq) # 转 float 计算 softmax，再转回原类型
            # d. 应用注意力 dropout
            scores = self.attn_dropout(scores)
            # e. 用注意力权重加权 V，得到输出: (bsz, n_heads, seq_len, kv_seq_len) @ (bsz, n_heads, kv_seq_len, head_dim) -> (bsz, n_heads, seq_len, head_dim)
            output = scores @ value

        # 7. 后处理
        # a. 转置输出，将头维度和序列长度维度换回: (bsz, seq_len, n_heads, head_dim)
        # b. Reshape 合并头维度: (bsz, seq_len, n_heads * head_dim) = (bsz, seq_len, dim)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # c. 应用输出线性层和残差 dropout
        output = self.resid_dropout(self.wo(output))

        # 返回最终的注意力输出和 KV 缓存
        return output, past_kv


# 定义前馈网络 (Feed Forward Network, FFN) 模块
class FeedForward(nn.Module):
    """
    标准的前馈网络模块，通常在 Transformer 块中使用。
    这里实现的是 SwiGLU (Swish Gated Linear Unit) 变体。
    FFN(x) = Dropout(W2 * (Swish(W1 * x) * (W3 * x)))
    """
    def __init__(self, config: LMConfig):
        """
        初始化 FeedForward 模块。
        Args:
            config (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        # 如果配置中未指定隐藏层维度 (hidden_dim)，则按 Llama 的方式计算
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim  # 通常 FFN 隐藏层是输入维度的 4 倍
            hidden_dim = int(2 * hidden_dim / 3)  # Llama/Mistral 使用 2/3 * (4 * dim)
            # 将 hidden_dim 调整为 config.multiple_of 的最接近倍数，以提高硬件效率
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        # 定义 SwiGLU 所需的三个线性层
        # w1 和 w3 用于计算门控值和被门控的值
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        # w2 用于将门控后的结果映射回原始维度
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        # Dropout 层
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        FeedForward 模块的前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状 (bsz, seq_len, dim)。
        Returns:
            torch.Tensor: 输出张量，形状 (bsz, seq_len, dim)。
        """
        # 计算 SwiGLU:
        # 1. F.silu(self.w1(x)) 计算 Swish(W1 * x)
        # 2. self.w3(x) 计算 W3 * x
        # 3. 两者逐元素相乘
        # 4. 通过 self.w2 映射回原始维度
        # 5. 应用 dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# 定义混合专家 (Mixture of Experts, MoE) 的门控网络模块
class MoEGate(nn.Module):
    """
    MoE 门控网络。负责为每个输入 token 计算路由权重，
    决定将 token 发送给哪些专家进行处理。
    """
    def __init__(self, config: LMConfig):
        """
        初始化 MoEGate 模块。
        Args:
            config (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        self.config = config  # 存储配置
        self.top_k = config.num_experts_per_tok  # 每个 token 选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 总的可路由专家数量

        self.scoring_func = config.scoring_func  # 计算路由分数的函数 ('softmax')
        self.alpha = config.aux_loss_alpha  # 辅助损失 (auxiliary loss) 的系数，用于负载均衡
        self.seq_aux = config.seq_aux  # 是否使用序列级别的辅助损失计算方式

        self.norm_topk_prob = config.norm_topk_prob  # 是否对 top-k 专家的权重进行归一化
        self.gating_dim = config.dim  # 门控网络的输入维度 (等于模型维度)
        # 定义门控权重参数，形状为 (总专家数, 模型维度)
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()  # 初始化权重

    def reset_parameters(self) -> None:
        """使用 Kaiming 均匀分布初始化门控权重。"""
        import torch.nn.init as init
        # Kaiming 初始化常用于 ReLU 及其变种激活函数
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        MoEGate 模块的前向传播。
        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态，形状 (bsz, seq_len, dim)。
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - topk_idx: 每个 token 选择的 top-k 专家的索引，形状 (bsz * seq_len, top_k)。
                - topk_weight: 对应 top-k 专家的权重，形状 (bsz * seq_len, top_k)。
                - aux_loss: 辅助损失值 (标量)。
        """
        bsz, seq_len, h = hidden_states.shape  # 获取输入形状
        # 将输入 reshape 为 (bsz * seq_len, dim) 以便进行线性变换
        hidden_states = hidden_states.view(-1, h)
        # 计算路由 logits：输入乘以门控权重 (转置)
        # F.linear(input, weight.T) 等效于 input @ weight.T
        logits = F.linear(hidden_states, self.weight, None) # (bsz * seq_len, n_routed_experts)

        # 根据配置的评分函数计算路由分数
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1) # 对每个 token 的专家分数进行 softmax
        else:
            # 如果使用了不支持的评分函数，则报错
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 从分数中选出 top-k 的专家及其权重
        # topk_weight: top-k 的分数
        # topk_idx: top-k 分数对应的专家索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果 top_k > 1 且配置要求归一化 top-k 权重
        if self.top_k > 1 and self.norm_topk_prob:
            # 计算 top-k 权重的和
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20 # 加 epsilon 防除零
            # 归一化权重，使其和为 1
            topk_weight = topk_weight / denominator

        # 计算辅助损失 (仅在训练时且 alpha > 0 时)
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # 使用原始的 softmax 分数计算辅助损失
            aux_topk = self.top_k
            # 将 top-k 索引 reshape 回 (bsz, seq_len * top_k)
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux: # 序列级辅助损失 (类似 Mixtral)
                # 将分数 reshape 回 (bsz, seq_len, n_routed_experts)
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 计算每个专家被选中的次数 (在整个 batch * seq_len 中)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # 使用 scatter_add_ 统计每个 batch 内，每个专家被选中的总次数
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts) # 除以期望的选中次数进行归一化
                # 辅助损失 = (每个专家被选中的频率 * 该专家在序列上的平均得分) 的和，再跨 batch 求平均
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else: # token 级辅助损失 (类似 GShard)
                # 创建一个 one-hot 掩码，标记每个 token 选择了哪些专家
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 计算每个专家被选中的频率 (跨整个 batch * seq_len)
                ce = mask_ce.float().mean(0) # (n_routed_experts,)
                # 计算每个专家获得的平均路由分数 (跨整个 batch * seq_len)
                Pi = scores_for_aux.mean(0) # (n_routed_experts,)
                # 计算每个专家处理的 token 比例 (fraction of tokens dispatched to expert i)
                fi = ce * self.n_routed_experts # 乘以专家数，得到负载因子
                # 辅助损失 = sum(Pi * fi) * alpha
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 如果不计算辅助损失，则设为 0
            aux_loss = 0

        # 返回 top-k 索引、top-k 权重和辅助损失
        return topk_idx, topk_weight, aux_loss


# 定义使用混合专家 (MoE) 的前馈网络模块
class MOEFeedForward(nn.Module):
    """
    混合专家 (MoE) 前馈网络层。
    它包含多个并行的标准 FeedForward 网络（专家），
    并通过一个门控网络 (MoEGate) 动态地为每个 token 选择一部分专家进行计算。
    """
    def __init__(self, config: LMConfig):
        """
        初始化 MOEFeedForward 模块。
        Args:
            config (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        self.config = config  # 存储配置
        # 创建一个包含 n_routed_experts 个 FeedForward 专家的列表
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 实例化门控网络
        self.gate = MoEGate(config)
        # 如果配置了共享专家 (shared experts)，则创建一个共享的 FeedForward 网络
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)
        # 初始化辅助损失属性
        self.aux_loss = 0.0

    def forward(self, x):
        """
        MOEFeedForward 模块的前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状 (bsz, seq_len, dim)。
        Returns:
            torch.Tensor: 输出张量，形状 (bsz, seq_len, dim)。
        """
        identity = x  # 保存原始输入，用于可能的共享专家计算或残差连接
        orig_shape = x.shape  # 保存原始输入形状
        bsz, seq_len, _ = x.shape # 获取批大小和序列长度

        # 1. 通过门控网络获取专家选择结果和辅助损失
        # topk_idx: (bsz * seq_len, top_k)
        # topk_weight: (bsz * seq_len, top_k)
        # aux_loss: scalar
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 将输入 reshape 为 token 列表: (bsz * seq_len, dim)
        x = x.view(-1, x.shape[-1])
        # 将 top-k 索引展平: (bsz * seq_len * top_k,)
        flat_topk_idx = topk_idx.view(-1)

        # 2. 根据训练/推理模式处理专家计算
        if self.training:
            # 训练模式：将每个 token 复制 top_k 次，分别送入选中的专家
            # a. 重复输入 x，使其长度变为 bsz * seq_len * top_k
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # b. 初始化一个空的输出张量 y
            y = torch.empty_like(x, dtype=torch.float16) # 使用 float16 减少内存占用，需要确保专家输出也是 float16
            # c. 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有应该由当前专家 i 处理的 token 的索引 (在重复后的 x 中)
                expert_mask = (flat_topk_idx == i)
                if expert_mask.any(): # 如果有 token 分配给这个专家
                    # 将这些 token 输入专家网络
                    expert_output = expert(x[expert_mask])
                    # 将专家输出写回 y 的对应位置，确保数据类型一致
                    y[expert_mask] = expert_output.to(y.dtype)
            # d. 将 y reshape 回 (bsz * seq_len, top_k, dim)
            #    乘以对应的 top-k 权重 (topk_weight.unsqueeze(-1)) 进行加权
            #    在 top_k 维度上求和，得到每个 token 的最终输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # e. 将 y reshape 回原始输入形状 (bsz, seq_len, dim)
            y = y.view(*orig_shape)
        else:
            # 推理模式：使用优化的 moe_infer 方法
            # 将展平的 top-k 权重 reshape 为 (bsz * seq_len * top_k, 1)
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 3. 如果有共享专家，将其输出加到 MoE 输出上
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity) # 使用原始输入 identity

        # 4. 将计算得到的辅助损失存储在模块属性中，以便外部访问
        self.aux_loss = aux_loss
        # 返回最终输出
        return y

    @torch.no_grad() # 推理时不需要计算梯度
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        优化的 MoE 推理方法。
        它将 token 按分配的专家进行排序和分组，然后批量处理每个专家的 token，
        最后使用 scatter_add_ 将结果高效地组合起来。
        Args:
            x (torch.Tensor): 输入 token 列表，形状 (bsz * seq_len, dim)。
            flat_expert_indices (torch.Tensor): 展平的 top-k 专家索引，形状 (bsz * seq_len * top_k,)。
            flat_expert_weights (torch.Tensor): 展平的 top-k 专家权重，形状 (bsz * seq_len * top_k, 1)。
        Returns:
            torch.Tensor: MoE 层的输出，形状 (bsz * seq_len, dim)。
        """
        # 初始化输出缓存，形状与输入 x 相同
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，得到排序后的索引 (idxs)
        idxs = flat_expert_indices.argsort()
        # 计算每个专家处理的 token 数量的累积和
        # flat_expert_indices.bincount() 统计每个专家索引出现的次数
        # .cpu().numpy().cumsum(0) 计算累积和，得到每个专家处理的 token 在排序后列表中的结束位置
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 获取排序前的原始 token 索引 (在 x 中的索引)
        # idxs // self.config.num_experts_per_tok 因为每个原始 token 对应 top_k 个条目在 flat_expert_indices 中
        token_idxs = idxs // self.config.num_experts_per_tok

        # 遍历每个专家
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前专家处理的 token 在排序后列表 (idxs) 中的起始索引
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # 如果起始和结束索引相同，说明没有 token 分配给这个专家，跳过
            if start_idx == end_idx:
                continue
            # 获取当前专家模块
            expert = self.experts[i]
            # 获取分配给当前专家的原始 token 在 x 中的索引 (exp_token_idx)
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 从输入 x 中提取这些 token
            expert_tokens = x[exp_token_idx]
            # 将 token 输入专家网络，得到输出，并确保数据类型与缓存一致
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 将专家输出乘以对应的权重 (从 flat_expert_weights 中按排序后的索引 idxs 取出)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 将加权后的专家输出累加到 expert_cache 的对应原始 token 位置
            # dim=0 表示在第 0 维进行操作
            # index=exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]) 指定了输出要加到哪些行
            # src=expert_out 是要加的值
            # 这样可以正确处理 top_k > 1 的情况，将同一个 token 来自不同专家的输出加起来
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        # 返回包含所有专家加权输出总和的缓存
        return expert_cache


# 定义 Transformer 模型的基础构建块 (Block)
class MiniMindBlock(nn.Module):
    """
    一个 Transformer 层/块，包含一个注意力子层和一个前馈（或 MoE）子层。
    每个子层之前都有 RMSNorm，并应用了残差连接。
    """
    def __init__(self, layer_id: int, config: LMConfig):
        """
        初始化 MiniMindBlock。
        Args:
            layer_id (int): 当前层的 ID (从 0 开始)。
            config (LMConfig): 包含模型配置参数的对象。
        """
        super().__init__()
        self.n_heads = config.n_heads  # 总注意力头数
        self.dim = config.dim  # 模型维度
        self.head_dim = config.dim // config.n_heads  # 每个头的维度
        # 实例化注意力模块
        self.attention = Attention(config)

        self.layer_id = layer_id  # 存储层 ID
        # 实例化注意力子层前的 RMSNorm
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # 实例化前馈子层前的 RMSNorm
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # 根据配置决定使用标准 FeedForward 还是 MOEFeedForward
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        """
        MiniMindBlock 的前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状 (bsz, seq_len, dim)。
            pos_cis (torch.Tensor): 当前序列长度的 RoPE 复数张量。
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): 可选的 KV 缓存。
            use_cache (bool): 是否使用并返回 KV 缓存。
        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                块的输出张量 (bsz, seq_len, dim) 和更新后的 KV 缓存。
        """
        # 1. 注意力子层
        # a. 应用 RMSNorm (Pre-Normalization)
        normed_x = self.attention_norm(x)
        # b. 通过注意力模块
        h_attn, past_kv = self.attention(
            normed_x,
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        # c. 应用第一个残差连接
        h = x + h_attn

        # 2. 前馈子层
        # a. 应用 RMSNorm (Pre-Normalization)
        normed_h = self.ffn_norm(h)
        # b. 通过前馈（或 MoE）模块
        ff_out = self.feed_forward(normed_h)
        # c. 应用第二个残差连接
        out = h + ff_out

        # 返回块的输出和 KV 缓存
        return out, past_kv


# 定义完整的 MiniMind 语言模型
class MiniMindLM(PreTrainedModel):
    """
    MiniMind 语言模型。继承自 Hugging Face 的 PreTrainedModel，
    整合了嵌入层、多个 MiniMindBlock 层、最终归一化层和输出层。
    支持 KV 缓存和 MoE 辅助损失。
    """
    config_class = LMConfig  # 指定配置类，用于 Hugging Face 集成

    def __init__(self, params: LMConfig = None):
        """
        初始化 MiniMindLM 模型。
        Args:
            params (LMConfig, optional): 模型配置对象。如果为 None，则使用默认 LMConfig。
        """
        # 如果未提供 params，则使用默认配置
        self.params = params or LMConfig()
        # 调用父类 PreTrainedModel 的构造函数，传递配置
        super().__init__(self.params)
        # 获取词汇表大小和层数
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        # 定义词嵌入层
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        # 定义输入 dropout 层
        self.dropout = nn.Dropout(params.dropout)
        # 创建包含 n_layers 个 MiniMindBlock 的 ModuleList
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])
        # 定义最终的 RMSNorm 层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # 定义输出线性层，将模型维度映射到词汇表大小 (logits)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        # 权重绑定：让词嵌入层和输出层的权重共享，可以减少参数量并可能提高性能
        self.tok_embeddings.weight = self.output.weight
        # 预计算 RoPE 的复数表示，并注册为 buffer
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta, end=params.max_seq_len * 2), # 预计算更长以支持可能的扩展
                             persistent=False)
        # 初始化用于存储模型输出的容器对象
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        """
        MiniMindLM 模型的前向传播。
        Args:
            input_ids (Optional[torch.Tensor]): 输入 token ID，形状 (bsz, seq_len)。
            past_key_values (Optional[List[Tuple[torch.Tensor, torch.Tensor]]]):
                可选的 KV 缓存列表，每个元素对应一层的 KV 缓存。
            use_cache (bool): 是否使用并返回 KV 缓存。
            **args: 其他参数，例如 'start_pos' 用于 KV 缓存时的 RoPE 位置索引。
        Returns:
            CausalLMOutputWithPast: 包含 logits, past_key_values, 和 aux_loss 的输出对象。
        """
        # 如果未提供 past_key_values，则初始化为 None 列表
        past_key_values = past_key_values or [None] * len(self.layers)
        # 获取 RoPE 的起始位置索引，默认为 0 (用于 KV 缓存场景)
        start_pos = args.get('start_pos', 0)
        # 1. 获取词嵌入并应用 dropout
        h = self.dropout(self.tok_embeddings(input_ids))
        # 2. 获取当前输入序列对应的 RoPE 编码
        # 从预计算的 pos_cis 中切片出需要的长度
        current_seq_len = input_ids.size(1)
        pos_cis = self.pos_cis[start_pos : start_pos + current_seq_len]
        # 3. 逐层通过 Transformer Block
        past_kvs = [] # 用于存储每一层更新后的 KV 缓存
        for l, layer in enumerate(self.layers):
            # 将输入 h、RoPE 编码、该层的 KV 缓存传入 Transformer Block
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            # 存储更新后的 KV 缓存
            past_kvs.append(past_kv)
        # 4. 应用最终的归一化层
        h = self.norm(h)
        # 5. 计算输出 logits
        logits = self.output(h)
        # 6. 计算 MoE 辅助损失 (如果使用了 MoE 层)
        # 遍历所有层，如果该层的前馈网络是 MOEFeedForward，则累加其 aux_loss
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))

        # 7. 将结果存储到输出对象中
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss) # 添加辅助损失项
        self.OUT.__setitem__('past_key_values', past_kvs if use_cache else None) # 只有 use_cache=True 时才返回缓存

        # 返回输出对象
        return self.OUT

    @torch.inference_mode() # 表明此方法用于推理，禁用梯度计算以节省内存和加速
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        """
        生成文本序列。
        Args:
            input_ids (torch.Tensor): 输入的 prompt token ID，形状 (bsz, prompt_len)。
            eos_token_id (int): 结束符 token ID。
            max_new_tokens (int): 要生成的最大新 token 数量。
            temperature (float): 温度系数，用于控制生成文本的随机性。较低的值使生成更确定。
            top_p (float): Top-p (nucleus) 采样阈值。只考虑累积概率达到 p 的最小词汇集。
            stream (bool): 是否以流式方式生成 (yield 每个新 token)。
            rp (float): 重复惩罚 (Repetition Penalty) 因子。大于 1 时惩罚已出现的 token。
            use_cache (bool): 是否使用 KV 缓存加速生成。
            pad_token_id (int): 填充 token ID，用于处理 batch 中不同长度的 prompt。
            **args: 其他传递给 forward 方法的参数。
        Returns:
            torch.Tensor or Generator:
                如果 stream=False，返回包含完整生成序列的张量 (bsz, prompt_len + generated_len)。
                如果 stream=True，返回一个生成器，每次 yield 新生成的 token 部分 (bsz, new_tokens_this_step)。
        """
        # 如果是流式生成，直接调用 _stream 方法并返回生成器
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 非流式生成 (一次性生成整个序列)
        generated = [] # 存储每个 batch item 的生成结果
        # 遍历 batch 中的每个输入序列
        for i in range(input_ids.size(0)):
            # 移除输入序列中的 padding token
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0) # 保持 batch 维度
            # 调用 _stream 生成器获取所有生成的 token
            out_stream = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            # 从生成器中提取每个时间步生成的最后一个 token
            tokens_list = [tokens[:, -1:] for tokens in out_stream]
            # 将所有生成的 token 拼接起来
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else torch.empty((1, 0), dtype=non_pad.dtype, device=non_pad.device) # 处理空生成的情况
            # 将原始 prompt 和生成的 token 拼接成完整序列
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)

        # 对 batch 中的所有序列进行填充，使它们长度一致
        max_length = max(seq.size(1) for seq in generated) # 找到最长序列的长度
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1) # 使用 pad_token_id 填充到 max_length
            for seq in generated
        ]
        # 将填充后的序列列表拼接成一个 batch 张量
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        """
        内部流式生成函数 (生成器)。
        Args: (同 generate)
        Yields:
            torch.Tensor: 每个时间步生成的 token 部分 (bsz, new_tokens_this_step)。
        """
        # 初始化状态
        start_len = input_ids.shape[1] # 记录 prompt 的长度
        first_seq = True # 标记是否是第一个 token 的生成 (需要处理整个 prompt)
        past_kvs = None # 初始化 KV 缓存

        # 循环生成新 token，直到达到最大长度或生成 EOS token
        # 注意：这里的循环条件是 input_ids.shape[1] < start_len + max_new_tokens - 1 似乎有点问题
        # 应该是 input_ids.shape[1] < start_len + max_new_tokens
        current_len = input_ids.shape[1]
        target_len = start_len + max_new_tokens
        while current_len < target_len:
            # 调用模型 forward 方法获取 logits 和 KV 缓存
            if first_seq or not use_cache:
                # 第一次或不使用缓存时，处理整个 input_ids
                out = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args)
                first_seq = False # 不再是第一次
            else:
                # 使用缓存时，只处理最后一个 token，并传入过去的 KV 缓存
                # 需要提供 start_pos 以便正确应用 RoPE
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=current_len - 1, **args) # start_pos 是上一个 token 的位置

            # 获取最后一个时间步的 logits 和更新后的 KV 缓存
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values

            # 应用重复惩罚：降低已生成 token 的 logits
            # 获取当前已生成的 token 列表 (不包括 batch 维度)
            generated_tokens = set(input_ids.tolist()[0]) # 只考虑 batch 中的第一个样本？这可能不适用于 batch > 1
            logits[:, list(generated_tokens)] /= rp # 对已生成的 token 的 logits 除以 rp (rp > 1 时降低概率)

            # 应用温度缩放：logits / temperature
            logits /= (temperature + 1e-9) # 加 epsilon 防止除以零

            # 应用 Top-p (Nucleus) 采样
            if top_p is not None and top_p < 1.0:
                # 1. 对 logits 降序排序
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                # 2. 计算排序后 logits 的 softmax 概率
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                # 3. 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # 4. 找到累积概率首次超过 top_p 的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                # 5. 将阈值位置之后的所有 token 标记为移除 (确保至少保留一个 token)
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                # 6. 将移除标记映射回原始 token 索引
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                # 7. 将要移除的 token 的 logits 设置为负无穷大
                logits[indices_to_remove] = -float('Inf')

            # 从修改后的 logits 分布中采样下一个 token
            # F.softmax 将 logits 转换为概率分布
            # torch.multinomial 根据概率采样 1 个 token
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            # 将新生成的 token 拼接到 input_ids 后面
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            current_len += 1 # 更新当前长度

            # yield 当前生成的所有新 token (从 prompt 之后开始)
            yield input_ids[:, start_len:]

            # 如果生成的 token 是 EOS token，则停止生成
            if input_ids_next.item() == eos_token_id:
                break