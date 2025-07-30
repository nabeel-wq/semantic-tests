Semantic Tests on Mistral-7B 

This repository provides a lightweight analysis framework for tracing the internal behavior of the mistralai/Mistral-7B-Instruct-v0.3 language model during autoregressive generation. It offers token-level insights into entropy, activation norms, embedding space dynamics, and layer-wise output distributions.

Overview

The code captures diagnostic information during text generation, focusing on:

Entropy per token — as a proxy for model uncertainty.

Top-k token predictions per layer — to trace semantic refinement across depth.

Activation norm and cosine drift — for monitoring hidden state evolution.

Embedding spread — estimating the diversity of internal representations.

These diagnostics support the analysis of multi-step reasoning performance and token-level semantic stability in transformer models.

Installation

Clone the repository and install required dependencies:

git clone https://github.com/yourusername/mistral-diagnostics.git
cd mistral-diagnostics
pip install -r requirements.txt

Dependencies include:

transformers

accelerate

torch

matplotlib

seaborn

Ensure you have access to the mistralai/Mistral-7B-Instruct-v0.3 model and a valid Hugging Face token.

Usage

Edit and run the main Python script or Jupyter notebook. Replace Enter your huggingface key with your personal token.

Example prompt (hardcoded in the script):

prompt = "Explain why language models struggle with multi-step reasoning."

The script will:

Generate tokens step-by-step (default: 30 tokens)

Log entropy, drift, and layerwise predictions to diagnostics.txt

Plot token-level entropy evolution using matplotlib

Output

Text logs (``) include:

Entropy and top-3 predictions per generation step

Per-layer top-3 logits

Norms and cosine drift between token activations

Summary statistics and generated text

Visual output:

Line plot of entropy over generation steps

File Structure

├── mistral_entropy_analysis.py     # Main diagnostic script
├── diagnostics.txt                 # Output log (generated per run)
├── README.md
└── requirements.txt

Applications

This toolkit is intended for:

Research on reasoning failures or instability in LLM outputs

Visualization of semantic convergence across transformer depth

Comparative studies between models or prompts

Supporting interpretability or optimization of LLM-based systems

My work with this model has found little difference between the above parameters (except entropy) for perceived low semantic density tasks (eg:linkedin posts), perceived high semantic density tasks (eg:an MF doom rap about quantum mechanics) and complete nonsense (asking the model to continue from  'unga bunga').

License

MIT License. You are free to use, adapt, and distribute this code with appropriate credit.

Contact

Maintained by Nabeel Ansari.For questions or collaborations, feel free to open an issue or reach out directly.

