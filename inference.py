import torch
from safetensors.torch import load_file
import tiktoken
from gpt import GPTModel
from pipeline import generate, text_to_token_ids, token_ids_to_text
from model_loader import detect_model_from_state_dict, smart_load_state_dict, BASE_CONFIG

def main():
    # Path to your safetensors checkpoint
    checkpoint_path = "/Users/himanshuvinchurkar/Documents/Project/LLMs-from-scratch/ch05/02_alternative_weight_loading/model-gpt2-xl.safetensors"
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_file(checkpoint_path)

    # Auto-detect config
    detected_cfg = detect_model_from_state_dict(state_dict)
    base_cfg = BASE_CONFIG.copy()
    base_cfg.update(detected_cfg)

    # Initialize model
    gpt = GPTModel(base_cfg)

    # Map device: try CUDA, then MPS, fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load weights with transpose fix
    smart_load_state_dict(gpt, state_dict)

    # Move model to device
    gpt.to(device)

    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Seed for reproducibility
    torch.manual_seed(123)

    # Your prompt
    prompt_text = "Hello"

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
