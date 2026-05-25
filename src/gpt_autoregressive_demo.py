"""
GPT 自回归文本生成演示（基于真实 GPT-2 权重，手动拆解每一步）

本脚本加载 Hugging Face 的 gpt2 预训练权重，
但不用 model.generate() 黑盒，而是手动实现每一层的 forward，
逐行展示输入文本如何一步步计算出下一个词的概率。
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# =============================================================================
# 步骤 0: 加载真实 GPT-2 模型与分词器
# =============================================================================

print("=" * 60)
print("步骤 0: 加载 GPT-2 预训练模型与分词器")
print("=" * 60)

# 加载 BPE 分词器（词汇表大小 50257）
# 分词器负责：文本 -> token ID 列表，以及反向解码
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 加载预训练模型（包含全部权重）
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()  # 设为评估模式（关闭 dropout 等训练特性）

# 提取 transformer 主体（包含所有层）
transformer = model.transformer

# ---- 从模型配置中读取超参数（而非硬编码） ----
VOCAB_SIZE = transformer.wte.num_embeddings      # 50257
N_EMBD = transformer.wte.embedding_dim           # 768 (每个 token 的向量维度)
N_LAYER = transformer.config.n_layer             # 12 (Transformer 层数)
N_HEAD = transformer.config.n_head               # 12 (Attention 头数)
N_CTX = transformer.config.n_positions           # 1024 (最大序列长度)
HEAD_DIM = N_EMBD // N_HEAD                      # 64 (每个头的维度)
FF_DIM = N_EMBD * 4                              # 3072 (FFN 中间层维度，GPT-2 用 4*n_embd)

# ---- 提取顶层权重（供后续手动 forward 使用） ----
wte = transformer.wte.weight               # word token embedding, shape: (50257, 768)
wpe = transformer.wpe.weight               # positional embedding, shape: (1024, 768)
ln_f_g = transformer.ln_f.weight           # final layer norm gamma
ln_f_b = transformer.ln_f.bias             # final layer norm beta

# ---- 提取 12 层 Transformer Block 的权重 ----
# 每一层包含：LayerNorm1, Attention(QKV+c_proj), LayerNorm2, FFN(c_fc+c_proj)
# 我们一次性全部提取，存成列表，方便后续按层索引
blocks = []
for i in range(N_LAYER):
    block = transformer.h[i]  # 第 i 个 transformer block
    blocks.append({
        # Layer Norm 1 (Attention 之前)
        "ln_1_g": block.ln_1.weight,
        "ln_1_b": block.ln_1.bias,
        # Attention: QKV 合并投影 + 输出投影
        "attn_c_attn_w": block.attn.c_attn.weight,   # shape: (768, 2304)
        "attn_c_attn_b": block.attn.c_attn.bias,     # shape: (2304,)
        "attn_c_proj_w": block.attn.c_proj.weight,   # shape: (768, 768)
        "attn_c_proj_b": block.attn.c_proj.bias,     # shape: (768,)
        # Layer Norm 2 (FFN 之前)
        "ln_2_g": block.ln_2.weight,
        "ln_2_b": block.ln_2.bias,
        # Feed-Forward: c_fc (768->3072), c_proj (3072->768)
        "mlp_c_fc_w": block.mlp.c_fc.weight,         # shape: (768, 3072)
        "mlp_c_fc_b": block.mlp.c_fc.bias,           # shape: (3072,)
        "mlp_c_proj_w": block.mlp.c_proj.weight,     # shape: (3072, 768)
        "mlp_c_proj_b": block.mlp.c_proj.bias,       # shape: (768,)
    })

# GPT-2 使用权重共享：输出投影矩阵 = wte 的转置
# 这样输入嵌入和输出预测共享同一组语义空间
lm_head_weight = wte  # shape: (50257, 768), 输出时用 wte.T @ hidden

print(f"✓ 分词器加载完成，词汇表大小: {VOCAB_SIZE}")
print(f"✓ 模型加载完成: GPT-2 ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M 参数)")
print(f"✓ 模型维度 n_embd: {N_EMBD}")
print(f"✓ 模型 wpe: {transformer.wpe.weight}")
print(f"✓ Transformer 层数 n_layer: {N_LAYER}")
print(f"✓ Attention 头数 n_head: {N_HEAD} (每头 {HEAD_DIM} 维)")
print(f"✓ 最大序列长度 n_positions: {N_CTX}")
print(f"✓ FFN 中间维度: {FF_DIM}")
print(f"✓ 已提取 {len(blocks)} 层 Block 权重，供手动 forward 使用")
print()

# 验证分词器：展示几个常见词的 token 化结果
print("分词器示例:")
for text in ["hello", " world", "ing", " transformer"]:
    tokens = tokenizer.encode(text)
    tokens_decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"  '{text}' -> token IDs: {tokens} -> 拆分: {tokens_decoded}")
print()


# =============================================================================
# 步骤 1: Token Embedding —— 把分词后的整数序列映射成向量
# =============================================================================
# TODO: 用 tokenizer 把输入文本转成 token ID 列表
# TODO: 从模型权重中提取 word embedding 矩阵 (wte)
# TODO: 通过查表得到每个 token 的嵌入向量，shape: (seq_len, n_embd)


# =============================================================================
# 步骤 2: Positional Encoding —— 给每个位置加上位置信息
# =============================================================================
# TODO: 从模型权重中提取 positional embedding 矩阵 (wpe)
# TODO: 为每个位置 (0, 1, 2, ..., seq_len-1) 查找位置向量
# TODO: 与 Token Embedding 相加，得到模型的初始输入 hidden_states


# =============================================================================
# 步骤 3: Transformer Block —— 堆叠多层，每层包含 Attention + FFN
# =============================================================================
# GPT-2 有 12 层 Transformer Block，每层结构相同：
#
# 3a. Layer Norm: 对输入做归一化（ln_1）
# 3b. Q/K/V 投影 + Attention 计算: 从 c_attn 权重中拆分出 W_q, W_k, W_v
#       - GPT-2 把 Q/K/V 的投影合并成一个大的 c_attn 权重矩阵
#       - 需要将其拆分为三部分，每部分 shape: (n_embd, n_embd)
# 3c. Causal Mask: 构造上三角为 -inf 的掩码，防止看到未来 token
# 3d. Softmax + 加权求和: 计算注意力输出
# 3e. 输出投影 (c_proj): 把 attention 结果映射回 n_embd 维度
# 3f. 残差连接: hidden_states = hidden_states + attn_output
#
# 3g. Layer Norm: 第二次归一化（ln_2）
# 3h. Feed-Forward: 从 c_fc 和 c_proj 权重实现 FFN
#       - 第一层: hidden_states @ W_fc + b_fc  (n_embd -> 4*n_embd)
#       - GELU 激活（GPT-2 用 GELU 而非 ReLU）
#       - 第二层: @ W_proj + b_proj  (4*n_embd -> n_embd)
# 3i. 残差连接: hidden_states = hidden_states + ffn_output
#
# TODO: 实现单层 Transformer Block 的前向传播
# TODO: 用循环堆叠 12 层


# =============================================================================
# 步骤 4: 最终 Layer Norm + Output Projection
# =============================================================================
# TODO: 最后一层之后做 final Layer Norm（ln_f）
# TODO: 用 word embedding 的转置作为输出投影矩阵（权重共享）
# TODO: 计算 logits: hidden_states @ wte.T，shape: (seq_len, vocab_size)
# TODO: logits[i] 表示预测第 i+1 个 token 的分数


# =============================================================================
# 步骤 5: 自回归生成循环 —— 逐步采样生成文本
# =============================================================================
# TODO: 实现生成函数 generate(prompt, max_new_tokens, temperature)
#       - 每次把已生成的所有 token 输入模型
#       - 取最后一个位置的 logits
#       - 应用 temperature 后 softmax 采样
#       - 把新 token 拼回输入，循环往复
# TODO: 用 tokenizer.decode() 把 token ID 序列转回可读文本


# =============================================================================
# 步骤 6: 运行演示
# =============================================================================
# TODO: 演示 1: 输入一段文本，手动 forward 并打印每层的中间结果
# TODO: 演示 2: 调用 generate 生成有意义的续写文本
# TODO: 与 transformers 的 model.generate() 结果做对比验证
