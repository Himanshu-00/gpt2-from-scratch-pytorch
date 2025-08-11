import torch
from safetensors.torch import load_file
import tiktoken
from gpt import GPTModel
import os
from pipeline import generate, text_to_token_ids, token_ids_to_text
from model_loader import detect_model_from_state_dict, BASE_CONFIG, load_weights_directly, load_weights

def main():
    # Path to your safetensors checkpoint
    checkpoint_path = ""

    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext == ".safetensors":
        state_dict = load_file(checkpoint_path)
    elif ext == ".pth":
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported checkpoint format: {ext}. Supported: .pth, .safetensors")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    # state_dict = load_file(checkpoint_path)

    # Auto-detect config
    detected_cfg = detect_model_from_state_dict(state_dict)
    base_cfg = BASE_CONFIG.copy()
    base_cfg.update(detected_cfg)

    # Initialize model
    gpt = GPTModel(base_cfg)

    # Map device: try CUDA, then MPS, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # # Load weights with transpose fix
    # load_weights_directly(gpt, state_dict)
    # Load weights with transpose fix
    load_weights(gpt, state_dict, transpose=False)

    # Move model to device
    gpt.to(device)

    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Seed for reproducibility
    torch.manual_seed(123)

    # Your prompt
    prompt_text = "Who fine tuned you?"

    # Convert text to tokens and move to device
    idx = text_to_token_ids(prompt_text, tokenizer).to(device)

    # Generate tokens
    token_ids = generate(
        model=gpt,
        idx=idx,
        max_new_tokens=40,
        context_size=base_cfg["context_length"],
        top_k=40,
        temperature=1.0,
    )

    # Decode generated tokens
    output_text = token_ids_to_text(token_ids, tokenizer)
    print("Output text:\n", output_text)


if __name__ == "__main__":
    main()
