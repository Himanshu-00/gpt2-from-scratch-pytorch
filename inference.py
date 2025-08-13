import torch
from safetensors.torch import load_file
import tiktoken
from gpt import GPTModel
import os
from pipeline import generate, text_to_token_ids, token_ids_to_text, generate_stream
from model_loader import detect_model_from_state_dict, BASE_CONFIG, load_weights_directly, load_weights


def detect_model_type(checkpoint_path):
    """Detect if using official weights or fine-tuned weights"""
    if "sft" in checkpoint_path.lower() or "alpaca" in checkpoint_path.lower() or "instruct" in checkpoint_path.lower():
        return "fine-tuned"
    else:
        return "official"

def format_prompt(user_prompt, model_type):
    """Format prompt based on model type"""
    if model_type == "fine-tuned":
        # Use Alpaca/Custom format for fine-tuned models
        formatted_entry = {
            "instruction": user_prompt,
            "input": ""
        }
        return format_input(formatted_entry)
    else:
        # Use direct prompt for official weights
        return user_prompt

def format_input(entry):
    """Format input using Alpaca dataset structure"""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    
    return instruction_text + input_text + "\n\n### Response:"

def chat_with_model(user_prompt, model, tokenizer, device, base_cfg, model_type, temperature=0.7, max_tokens=150, debug=False):
    """Convert prompt based on model type and generate response"""
    
    # Format based on model type
    input_text = format_prompt(user_prompt, model_type)
    
    if model_type == "fine-tuned":
        # For fine-tuned models, use your existing logic
        accumulated_text = input_text
        response_started = False
        clean_response = ""

        for new_token_text in generate_stream(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=max_tokens,
            context_size=base_cfg["context_length"],
            top_k=40,
            temperature=temperature,
            tokenizer=tokenizer,
            eos_id=50256,
            debug=debug
        ):
            accumulated_text += new_token_text
            
            if not response_started and "### Response:" in accumulated_text:
                response_started = True
                response_start = accumulated_text.find("### Response:") + len("### Response:")
                clean_response = accumulated_text[response_start:].strip()
            elif response_started:
                clean_response += new_token_text
                print(new_token_text, end="", flush=True)
                    
        return clean_response.strip()
    
    else:
        # For official weights, direct streaming
        for new_token_text in generate_stream(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=max_tokens,
            context_size=base_cfg["context_length"],
            top_k=40,
            temperature=temperature,
            tokenizer=tokenizer,
            eos_id=50256,
            debug=debug
        ):
            print(new_token_text, end="", flush=True)
        
        return "Response completed"

def interactive_chat(model, tokenizer, device, base_cfg, model_type, debug=False):
    """Interactive chat interface with model type awareness"""
    model_desc = "Fine-tuned" if model_type == "fine-tuned" else "Official (Direct prompts)"
    print(f"🤖 Chat with GPT-2 Model ({model_desc}) (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        user_input = input("\n💬 You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        print("🤖 AI: ", end="", flush=True)
        chat_with_model(user_input, model, tokenizer, device, base_cfg, model_type, debug=debug)
        print()

# def chat_with_model(user_prompt, model, tokenizer, device, base_cfg, temperature=0.7, max_tokens=150):
#     """Convert casual prompt to Alpaca format and generate response"""
#     # Convert casual prompt to Alpaca format internally
#     formatted_entry = {
#         "instruction": user_prompt,
#         "input": ""
#     }
    
#     # Use your working format_input function
#     input_text = format_input(formatted_entry)
#     # Track full accumulated text
#     accumulated_text = input_text
#     response_started = False
#     clean_response = ""

#     # Stream tokens one by one
#     for new_token_text in generate_stream(
#         model=model,
#         idx=text_to_token_ids(input_text, tokenizer).to(device),
#         max_new_tokens=40,
#         context_size=base_cfg["context_length"],
#         top_k=40,
#         temperature=1.0,
#         tokenizer=tokenizer,  # Make sure to pass tokenizer
#         eos_id=50256,
#         debug=False,
#     ):
#            # Add new token to accumulated text
#             accumulated_text += new_token_text
            
#             # Check if we've reached the response section
#             if not response_started and "### Response:" in accumulated_text:
#                 response_started = True
#                 # Extract everything after "### Response:"
#                 response_start = accumulated_text.find("### Response:") + len("### Response:")
#                 clean_response = accumulated_text[response_start:].strip()
#             elif response_started:
#                 # We're in response section, just add the new token
#                 clean_response += new_token_text
#                 print(new_token_text, end="", flush=True) # Stream only response tokens
                
    
#     return clean_response.strip()

# def interactive_chat(model, tokenizer, device, base_cfg):
#     """Interactive chat interface"""
#     print("🤖 Chat with your GPT-2 Model (type 'quit' to exit)")
#     print("=" * 50)
    
#     while True:
#         user_input = input("\n💬 You: ")
        
#         if user_input.lower() in ['quit', 'exit', 'bye']:
#             print("👋 Goodbye!")
#             break
            
#         if not user_input.strip():
#             continue
            
#         print("🤖 AI: ", end="", flush=True)
#         chat_with_model(user_input, model, tokenizer, device, base_cfg)
#         print()

def test_model_capabilities(model, tokenizer, device, base_cfg):
    """Test the model with various prompts"""
    # Detect model type
    model_type = 'fine-tuned'
    test_prompts = [
        "Rewrite 'He installed the software' in passive voice",
        "Convert 'The book was written by her' to active voice", 
        "What is 5 + 3?",
        "Convert 'hello world' to uppercase",
        "Tell me a short joke",
        "Explain photosynthesis in simple terms",
        "Write a haiku about technology"
    ]
    
    print("🧪 Testing Model Capabilities")
    print("=" * 50)
    
    for prompt in test_prompts:
        response = chat_with_model(prompt, model, tokenizer, device, base_cfg, model_type)
        print(f"\n💬 Prompt: {prompt}")
        print(f"🤖 Response: {response}")
        print("-" * 30)


def main():
    # Path to your safetensors checkpoint
    checkpoint_path = "gpt2-medium355M-sft.safetensors"

    example =  ["Give an example of a metaphor that uses the following object: Stars",
              "Rewrite the following sentence in the third person: I am anxious"]

    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext == ".safetensors":
        state_dict = load_file(checkpoint_path)
    elif ext == ".pth":
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported checkpoint format: {ext}. Supported: .pth, .safetensors")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    # state_dict = load_file(checkpoint_path)
    # Detect model type
    model_type = detect_model_type(checkpoint_path)

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

    # Load weights with transpose fix**(USE transpose=True if using official weights)
    load_weights(gpt, state_dict, transpose=False)

    # Move model to device
    gpt.to(device)

    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Seed for reproducibility
    torch.manual_seed(123)

    print("✅ Model loaded successfully!")
    
    # Choose what to do
    print("\nChoose an option:")
    print("1. Interactive chat")
    print("2. Test model capabilities")
    print("3. Single prompt")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        interactive_chat(gpt, tokenizer, device, base_cfg, model_type)
    elif choice == "2":
        test_model_capabilities(gpt, tokenizer, device, base_cfg)
    elif choice == "3":
        user_prompt = input("\nEnter your prompt: ")
        response = chat_with_model(user_prompt, gpt, tokenizer, device, base_cfg, model_type)
        print(f"\n🤖 Response: {response}")
    else:
        print("Invalid choice, starting interactive chat...")
        interactive_chat(gpt, tokenizer, device, base_cfg, model_type)

    # # Your prompt
    # prompt_text = "Hello!!1"

    # # Convert text to tokens and move to device
    # idx = text_to_token_ids(prompt_text, tokenizer).to(device)

    # # DIRECT OUTPUT AT END
    # # Generate tokens
    # token_ids = generate(
    #     model=gpt,
    #     idx=idx,
    #     max_new_tokens=40,
    #     context_size=base_cfg["context_length"],
    #     top_k=40,
    #     temperature=1.0,
    # )


    # # Stream tokens one by one
    # for new_token_text in generate_stream(
    #     model=gpt,
    #     idx=idx,
    #     max_new_tokens=40,
    #     context_size=base_cfg["context_length"],
    #     top_k=40,
    #     temperature=1.0,
    #     tokenizer=tokenizer  # Make sure to pass tokenizer
    # ):
    #     print(new_token_text, end="", flush=True)  # Stream each token


    # def format_input(entry):
    #     instruction_text = (
    #         f"Below is an instruction that describes a task. "
    #         f"Write a response that appropriately completes the request."
    #         f"\n\n### Instruction:\n{entry['instruction']}"
    #     )

    #     input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    #     return instruction_text + input_text


    # # Instead of manually formatting, use your training format:
    # test_entry = {
    #     "instruction": "Create a restaurant review",
    #     "input": "The Yellow Hen"
    # }

    # # Use the SAME format_input function from training
    # input_text = format_input(test_entry)

    # # Then generate
    # token_ids = generate(
    #     model=gpt,
    #     idx=text_to_token_ids(input_text, tokenizer).to(device),
    #     max_new_tokens=50,
    #     context_size=BASE_CONFIG["context_length"],
    #     temperature=0.7,
    #     eos_id=50256
    # )


    #     # Optional: add delay for typewriter effect
    #     # time.sleep(0.05)
    

    # # Decode generated tokens
    # output_text = token_ids_to_text(token_ids, tokenizer)
    # print("Output text:\n", output_text)


if __name__ == "__main__":
    main()
