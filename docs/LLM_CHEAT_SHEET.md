# LLM Technical Cheat Sheet

A fast reference for core LLM concepts with direct links to primary papers and official docs.

## 0) Fast Navigation

- Foundations: transformer basics, objective, scaling
- Tokenization: BPE/SentencePiece and tooling
- Architecture: RoPE/ALiBi, FlashAttention, MQA/GQA, MoE
- Pretraining: data, compute, optimization
- Post-training: instruction tuning, RLHF/DPO, PEFT
- Inference: KV cache, serving stacks, systems optimization
- Evaluation: perplexity, benchmark harnesses
- RAG/agents: retrieval and tool-using patterns

## 1) Foundations

### What an LLM is optimizing

- Autoregressive LM objective and perplexity (practical explanation):
  - https://huggingface.co/docs/transformers/v4.17.0/perplexity

### Transformer architecture

- Original Transformer paper:
  - https://arxiv.org/abs/1706.03762

### Scaling behavior

- Neural scaling laws:
  - https://arxiv.org/abs/2001.08361
- Compute-optimal training (Chinchilla):
  - https://arxiv.org/abs/2203.15556

## 2) Tokenization

### Core methods

- BPE for subword units:
  - https://arxiv.org/abs/1508.07909
- SentencePiece:
  - https://arxiv.org/abs/1808.06226

### Tooling

- Hugging Face Tokenizers docs:
  - https://huggingface.co/docs/tokenizers/en/index
- OpenAI `tiktoken`:
  - https://github.com/openai/tiktoken

## 3) Architecture Internals

### Positional encoding and long-context behavior

- RoPE:
  - https://arxiv.org/abs/2104.09864
- ALiBi:
  - https://arxiv.org/abs/2108.12409

### Attention efficiency

- FlashAttention:
  - https://arxiv.org/abs/2205.14135
- Multi-Query Attention (MQA):
  - https://arxiv.org/abs/1911.02150
- Grouped-Query Attention (GQA):
  - https://arxiv.org/abs/2305.13245

### Sparse experts

- Switch Transformers (MoE at scale):
  - https://arxiv.org/abs/2101.03961

## 4) Pretraining Data and Optimization

### Dataset engineering at scale

- The Pile:
  - https://arxiv.org/abs/2101.00027
- RefinedWeb (web-only high-quality filtering):
  - https://arxiv.org/abs/2306.01116

### Optimization

- AdamW (decoupled weight decay):
  - https://arxiv.org/abs/1711.05101

## 5) Post-Training and Alignment

### Instruction tuning and preference alignment

- InstructGPT (SFT + RLHF pipeline):
  - https://arxiv.org/abs/2203.02155
- DPO (RLHF alternative without explicit RL loop):
  - https://arxiv.org/abs/2305.18290

### Parameter-efficient fine-tuning

- LoRA:
  - https://arxiv.org/abs/2106.09685
- QLoRA:
  - https://arxiv.org/abs/2305.14314
- PEFT docs:
  - https://huggingface.co/docs/peft/main/en/index
- TRL docs (SFT/DPO/RLHF trainers):
  - https://huggingface.co/docs/trl/main/index

## 6) Inference and Systems

### KV cache and decoding

- Hugging Face KV cache guide:
  - https://huggingface.co/docs/transformers/main/en/kv_cache

### Serving engines

- vLLM docs:
  - https://docs.vllm.ai/
- PagedAttention paper (vLLM core idea):
  - https://arxiv.org/abs/2309.06180
- llama.cpp (local C/C++ inference):
  - https://github.com/ggml-org/llama.cpp

### Training/inference systems optimization

- `torch.compile`:
  - https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch AMP (mixed precision):
  - https://docs.pytorch.org/docs/stable/amp.html
- PyTorch distributed training:
  - https://docs.pytorch.org/docs/stable/distributed.html
- ZeRO paper:
  - https://arxiv.org/abs/1910.02054
- ZeRO-Offload paper:
  - https://arxiv.org/abs/2101.06840
- DeepSpeed ZeRO tutorial:
  - https://www.deepspeed.ai/tutorials/zero/
- Megatron-LM model parallelism:
  - https://arxiv.org/abs/1909.08053

## 7) Evaluation

### Basic and broad evaluation

- Perplexity primer and caveats:
  - https://huggingface.co/docs/transformers/v4.17.0/perplexity
- EleutherAI LM Evaluation Harness:
  - https://github.com/EleutherAI/lm-evaluation-harness

## 8) RAG and Agentic Patterns

- Retrieval-Augmented Generation (RAG):
  - https://arxiv.org/abs/2005.11401
- ReAct (reason + act loop):
  - https://arxiv.org/abs/2210.03629

## 9) Suggested Learning Order

If you are building models yourself, use this order:

1. Transformer + objective + perplexity (`1706.03762`, HF perplexity docs)
2. Tokenization (`1508.07909`, `1808.06226`, HF Tokenizers)
3. Scaling (`2001.08361`, `2203.15556`)
4. Optimization + systems (`1711.05101`, ZeRO, AMP, distributed)
5. Post-training (`2203.02155`, `2305.18290`, LoRA/QLoRA)
6. Inference (`kv_cache`, vLLM/PagedAttention, llama.cpp)
7. Evaluation (LM Eval Harness + domain-specific tests)

## 10) Quick Glossary

- Tokens: discrete units produced by a tokenizer.
- Context window: max tokens model attends to.
- Perplexity: exponentiated token-level negative log-likelihood.
- KV cache: saved attention keys/values for faster autoregressive decoding.
- SFT: supervised fine-tuning on instruction-response pairs.
- RLHF: preference alignment using reward modeling + RL or alternatives like DPO.
- PEFT: methods that update a small subset of parameters (e.g., LoRA).
