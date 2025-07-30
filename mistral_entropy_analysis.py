# Cell 1: Install dependencies
!pip install -q transformers accelerate

# Cell 2: Authenticate with Hugging Face
from huggingface_hub import login
login("Enter your huggingface key")

# Cell 3: Import required libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cell 4: Load model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id, token="Enter your huggingface key")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    token="Enter your huggingface key"
)
model.eval()

# Cell 5: Define prompt and encode it
prompt = "Explain why language models struggle with multi-step reasoning."
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with open("diagnostics.txt", "w") as f:
    f.write("\nInputs:\n" + str(inputs) + "\n")
    input_ids = inputs["input_ids"]
    f.write("\nInput IDs:\n" + str(input_ids) + "\n")

# Cell 6: Manual token-by-token generation with entropy tracking and diagnostics
max_new_tokens = 30
entropies = []
activation_norms = []
layerwise_drifts = []
embedding_spreads = []
layerwise_logits = []
all_ids = input_ids.clone()

prev_hidden = None

with open("diagnostics.txt", "a") as f:
    f.write("\nDetailed token diagnostics:\n")

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(all_ids, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            entropies.append(entropy.item())

            topk_probs, topk_ids = torch.topk(probs, k=3, dim=-1)
            top_tokens = tokenizer.convert_ids_to_tokens(topk_ids[0].tolist())
            token_probs = topk_probs[0].tolist()

            top_k_str = ", ".join([
                f"{tok} ({p:.2f})" for tok, p in zip(top_tokens, token_probs)
            ])
            f.write(f"Step {step+1} | Entropy: {entropy.item():.4f} | Top-3: {top_k_str}\n")
            print(f"Step {step+1} | Entropy: {entropy.item():.4f} | Top-3: {top_k_str}")

            last_hidden = outputs.hidden_states[-1][0, -1, :]
            activation_norms.append(last_hidden.norm().item())

            if prev_hidden is not None:
                drift = F.cosine_similarity(prev_hidden, last_hidden, dim=0).item()
                layerwise_drifts.append(drift)
            prev_hidden = last_hidden.detach()

            if step == 0:
                token_embeddings = outputs.hidden_states[-1][0, :, :].detach()
            else:
                token_embeddings = torch.cat([token_embeddings, last_hidden.unsqueeze(0)], dim=0)

            layer_logits = [layer[0, -1, :].float().cpu() for layer in outputs.hidden_states]
            layerwise_logits.append(layer_logits)
            for i, layer_logit in enumerate(layer_logits):
                top_vals, top_idxs = torch.topk(layer_logit, k=3)
                top_tokens_layer = tokenizer.convert_ids_to_tokens(top_idxs.tolist())
                top_str = ", ".join(f"{t} ({v:.2f})" for t, v in zip(top_tokens_layer, top_vals.tolist()))
                print(f"  Layer {i}: {top_str}")
                f.write(f"  Layer {i}: {top_str}\n")

            next_token = torch.argmax(probs, dim=-1)
            all_ids = torch.cat([all_ids, next_token.unsqueeze(0)], dim=-1)

# Cell 7: Decode final output and log summary
decoded = tokenizer.decode(all_ids[0], skip_special_tokens=True)
with open("diagnostics.txt", "a") as f:
    f.write("\nGenerated Text:\n" + decoded + "\n")

    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    f.write(f"\nAverage Entropy: {avg_entropy:.4f} nats\n")
    print(f"Average Entropy: {avg_entropy:.4f} nats")

    avg_norm = sum(activation_norms) / len(activation_norms) if activation_norms else 0.0
    f.write(f"Average Activation Norm: {avg_norm:.4f}\n")
    print(f"Average Activation Norm: {avg_norm:.4f}")

    avg_drift = sum(layerwise_drifts) / len(layerwise_drifts) if layerwise_drifts else 0.0
    f.write(f"Average Layerwise Cosine Drift: {avg_drift:.4f}\n")
    print(f"Average Layerwise Cosine Drift: {avg_drift:.4f}")

    token_embeddings_fp32 = token_embeddings.float()
    spread = torch.pdist(token_embeddings_fp32).mean().item() if token_embeddings_fp32.shape[0] > 1 else 0.0
    f.write(f"Token Embedding Spread: {spread:.4f}\n")
    print(f"Token Embedding Spread: {spread:.4f}")

# Cell 8: Plot entropy over generation steps
plt.figure(figsize=(10, 5))
sns.lineplot(x=list(range(1, len(entropies)+1)), y=entropies)
plt.title("Per-token Entropy during Generation")
plt.xlabel("Token Generation Step")
plt.ylabel("Entropy (nats)")
plt.grid(True)
plt.show()
