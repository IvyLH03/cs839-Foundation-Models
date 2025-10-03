from transformers import GPT2Model

def calculating_by_hyperparameters():
  E = 768
  V = 50257
  H = 12
  L = 12
  P = 1024

  Embeddings = V * E + P * E
  LayerNorms = 4 * E
  Attention = 4 * E * E + 4 * E
  MLP = 8 * E * E + 5 * E
  print (f"Embeddings: {Embeddings:,}")
  print (f"LayerNorms: {LayerNorms:,}")
  print (f"Attention: {Attention:,}")
  print (f"MLP: {MLP:,}")
  Total_per_layer = LayerNorms + Attention + MLP
  print (f"Total per layer: {Total_per_layer:,}")
  Total = Embeddings + L * Total_per_layer + 2 * E
  print (f"Total: {Total:,}")


  GPT2_medium_L = 24
  GPT2_medium_E = 1024
  GPT2_medium_per_layer = 4 * GPT2_medium_E + 4 * GPT2_medium_E * GPT2_medium_E + 4 * GPT2_medium_E + 8 * GPT2_medium_E * GPT2_medium_E + 5 * GPT2_medium_E
  Total_medium = V * GPT2_medium_E + P * GPT2_medium_E + GPT2_medium_L * GPT2_medium_per_layer + 2 * GPT2_medium_E


  GPT2_large_L = 36
  GPT2_large_E = 1280
  GPT2_large_per_layer = 4 * GPT2_large_E + 4 * GPT2_large_E * GPT2_large_E + 4 * GPT2_large_E + 8 * GPT2_large_E * GPT2_large_E + 5 * GPT2_large_E
  Total_large = V * GPT2_large_E + P * GPT2_large_E + GPT2_large_L * GPT2_large_per_layer + 2 * GPT2_large_E

  GPT2_xl_L = 48
  GPT2_xl_E = 1600
  GPT2_xl_per_layer = 4 * GPT2_xl_E + 4 * GPT2_xl_E * GPT2_xl_E + 4 * GPT2_xl_E + 8 * GPT2_xl_E * GPT2_xl_E + 5 * GPT2_xl_E
  Total_xl = V * GPT2_xl_E + P * GPT2_xl_E + GPT2_xl_L * GPT2_xl_per_layer + 2 * GPT2_xl_E

  print (f"Total medium: {Total_medium:,}")
  print (f"Total large: {Total_large:,}")
  print (f"Total xl: {Total_xl:,}")


def count_params(model, is_human: bool = False):
    params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{params / 1e6:.2f}M" if is_human else params

model = GPT2Model.from_pretrained("gpt2")
print(model)
print("Total # of params:", count_params(model, is_human=False))
calculating_by_hyperparameters()