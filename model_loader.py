# Define the standard GPT-2 model configs with params
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

def model_name_from_config(emb_dim, n_layers, n_heads, configs=MODEL_CONFIGS):
    for name, cfg in configs.items():
        if (
            cfg["emb_dim"] == emb_dim and
            cfg["n_layers"] == n_layers and
            cfg["n_heads"] == n_heads
        ):
            return name
    return f"Custom GPT model ({emb_dim=} {n_layers=} {n_heads=})"


def detect_model_from_state_dict(state_dict):
    """
    Auto-detects a matching GPT configuration from state_dict by heuristic:
    - Looks at keys like h.0.attn.c_attn.weight shape to infer emb_dim
    - Infers number of layers by counting h.* keys
    - Infers number of heads by dividing emb_dim by head_dim from the c_attn weight shape
    """
    # Identify transformer block keys
    block_keys = [k for k in state_dict.keys() if k.startswith("h.")]
    # Find distinct layers (assumes format h.{layer_number}.something)
    layers = set()
    for k in block_keys:
        parts = k.split('.')
        if len(parts) > 1 and parts[0] == 'h':
            try:
                layers.add(int(parts[1]))
            except:
                pass
    n_layers = len(layers)

    # Get one c_attn weight for inference
    sample_key = f"h.0.attn.c_attn.weight"
    if sample_key not in state_dict:
        raise ValueError(f"Cannot find key {sample_key} in checkpoint. Unable to infer model config.")

    c_attn_weight = state_dict[sample_key]
    # c_attn weight shape is (in_features, 3 * out_features) or (out_features, 3 * in_features)
    # Because of transpose, check which shape is plausible:
    # We expect: weight shape ~ (emb_dim, 3*emb_dim)
    shape = c_attn_weight.shape

    # Heuristic for embedding dim and number of heads:
    # Usually shape is (emb_dim, 3*emb_dim) or transposed (3*emb_dim, emb_dim)
    # We'll try both permutations and pick the probable emb_dim

    if shape[1] % 3 == 0:
        emb_dim = shape[0]
        triple_dim = shape[1]
    elif shape[0] % 3 == 0:
        emb_dim = shape[1]
        triple_dim = shape[0]
    else:
        raise ValueError(f"Unexpected shape of c_attn weight: {shape}")

    # emb_dim should equal triple_dim / 3
    if triple_dim // 3 != emb_dim:
        emb_dim = triple_dim // 3  # fallback

    # Try to infer n_heads by common divisors of emb_dim
    # We check possible heads from known configs
    possible_heads = []
    for cfg in MODEL_CONFIGS.values():
        if cfg['emb_dim'] == emb_dim:
            possible_heads.append(cfg["n_heads"])
    # Pick the first if any, else default to 12
    n_heads = possible_heads[0] if possible_heads else 12

    ## print values
    # print(f"Auto-detected model config: emb_dim={emb_dim}, n_layers={n_layers}, n_heads={n_heads}")
    model_name = model_name_from_config(cfg['emb_dim'], cfg['n_layers'], cfg['n_heads'])
    print(f"Detected model: {model_name}")

    return {
        "emb_dim": emb_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "Detected model" :model_name,
    }

def smart_load_state_dict(model, state_dict):
    """
    Loads weights into model. Automatically transposes nn.Linear/nn.Embedding 
    weights if model and checkpoint shapes are reversed (in_features vs out_features).
    """
    model_params = dict(model.named_parameters())
    new_state = {}
    for k, v in state_dict.items():
        if k in model_params:
            model_shape = model_params[k].shape
            if v.ndim == 2 and v.shape[::-1] == model_shape:  # shape reversed
                new_state[k] = v.T
            else:
                new_state[k] = v
        else:
            new_state[k] = v
    # strict=False to allow missing non-critical keys (like causal mask buffers)
    return model.load_state_dict(new_state, strict=False)

