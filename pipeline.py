
import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for step in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)


        if eos_id is not None and idx_next.item() == eos_id:
            print(f"Step {step}: EOS token generated. Stopping early.")
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_stream(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None, tokenizer=None, debug=None):
    """Generator that yields each new token as it's generated"""
    original_length = idx.shape[1]
    
    for step in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            if debug:
                print(f"Step {step}: EOS token generated. Stopping early.")
            break

        idx = torch.cat((idx, idx_next), dim=1)
        
        # Yield the new token (streaming)
        if tokenizer is not None:
            # Yield the decoded token text
            new_token_text = tokenizer.decode([idx_next.item()])
            yield new_token_text
        else:
            # Yield the token ID
            yield idx_next.item()
    
    # Return complete sequence at the end
    return idx
