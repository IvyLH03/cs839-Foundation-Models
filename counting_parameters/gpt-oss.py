from transformers import AutoModelForCausalLM
import torch

def count_params_by_model():

  model_name = "openai/gpt-oss-20b"
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      trust_remote_code=True,
      device_map="meta"  # or other mapping so you don’t OOM
  )

  def count_params(module: torch.nn.Module, only_trainable: bool = True):
      """ Count total parameters in a module (optionally only those with requires_grad). """
      params = 0
      for p in module.parameters():
          if only_trainable and not p.requires_grad:
              continue
          params += p.numel()
      return params

  # Top-level breakdown
  print("Overall total:", count_params(model))
  print("Overall trainable:", count_params(model, only_trainable=True))

  # Breakdown per “child” module
  for name, child in model.named_children():
      c = count_params(child)
      ct = count_params(child, only_trainable=True)
      print(f"{name:20s} — total: {c:12,} | trainable: {ct:12,}")

  # If you want deeper breakdown (e.g. per transformer block h.<i>)
  for prefix, module in model.named_modules():
      # skip the root module (prefix == "")
      if prefix:
          c = count_params(module)
          if c > 0:
              print(f"{prefix:50s} : {c:,}")

def calculate_from_hyperparameters():
    """
    L (number of transformer layers): 24
• E (embedding dimension) = 2,880
• V (vocabulary size) = 201,088
• P (maximum positional indexes) = 131k? How does RoPE work?
• head dim (Key/Query/Value head dimension) = 64
• num q heads (number of query heads) = 64
• num kv groups (number of key-value groups) = 8
• num M oE blocks (number of MoE blocks) = 32
    """

    L = 24
    E = 2880
    V = 201088
    P = 131072  # assuming max position embeddings
    head_dim = 64
    num_q_heads = 64
    num_kv_groups = 8
    num_MoE_blocks = 32
    expert_per_token = 4

    RMSNorm = E

    # Embeddings
    embed = V * E
    print (f"Token Embeddings: {embed:,}")
    
    # Attention
    Q_proj = E * head_dim * num_q_heads + num_q_heads * head_dim  # weights + biases
    K_proj = E * head_dim * num_kv_groups + num_kv_groups * head_dim
    V_proj = E * head_dim * num_kv_groups + num_kv_groups * head_dim
    Out_proj = head_dim * num_q_heads * E + E  
    bias_per_head = num_q_heads
    Attention = Q_proj + K_proj + V_proj + Out_proj + bias_per_head

    print(f"Q_proj: {Q_proj:,}")
    print(f"K_proj: {K_proj:,}")
    print(f"V_proj: {V_proj:,}")
    print(f"Out_proj: {Out_proj:,}")
    print(f"Attention: {Attention:,}")

    # MLP with MoE
    router = E * num_MoE_blocks + num_MoE_blocks  
    each_expert = (E * 3 * E + 3 * E)  # weights + biases
    MLP = num_MoE_blocks * each_expert + router

    print(f"Router: {router:,}")
    print(f"Each expert: {each_expert:,}")
    print(f"MLP: {MLP:,}")

    per_layer_total = RMSNorm + Attention + RMSNorm + MLP
    print (f"Per layer total: {per_layer_total:,}")

    total = embed + L * per_layer_total + RMSNorm
    print (f"Total: {total:,}")

    total_with_unembed = embed + total
    print (f"Total with unembed: {total_with_unembed:,}")

    active_MLP = router + expert_per_token * each_expert
    active_per_layer = RMSNorm + Attention + RMSNorm + active_MLP
    active_total = embed + L * active_per_layer + RMSNorm
    print (f"Active MLP: {active_MLP:,}")
    print (f"Active per layer: {active_per_layer:,}")
    print (f"Active total: {active_total:,}")




calculate_from_hyperparameters()