# GPT-2 from Scratch in PyTorch

This repository contains a complete **implementation of the GPT architecture** in **PyTorch**, built entirely from scratch.

✅ **Supports inference** for:
- `gpt2-small`
- `gpt2-medium`
- `gpt2-large`
- `gpt2-xl`

📌 No external model libraries like Hugging Face are used — the model is built using **pure PyTorch**, closely following the original GPT-2 paper and architecture.

---

## Features

- [x] GPT-2 architecture implemented from scratch
- [x] Multi-head self-attention, layer normalization, GELU activation
- [x] Positional encodings and token embeddings
- [x] Fully modular and readable code
- [x] Load pretrained GPT-2 weights (`small` to `xl`)
- [x] Inference with raw input text
- [X] Supervised Fine-Tuning (SFT) on custom datasets

🔧 **Coming Soon**:
- [ ] RLHF alignment training
- [ ] Mixed precision training for faster performance

---

## Installation

```bash
git clone https://github.com/Himanshu-00/gpt2-from-scratch-pytorch.git
cd gpt2-from-scratch-pytorch
pip install -r requirements.txt
```

## Usage
#### Run Inference
```bash
python3 inference.py
```
#### Fine-Tuning
```bash
python3 train.py
```

## Try it instantly — no local setup needed.

🔹 Fine-Tuning Notebook & Inference → [Google colab](https://colab.research.google.com/drive/1BrpBqLnS2Gy5fhIaT3lTVJqQYtp6580r?usp=sharing)


## A huge thanks to the following open-source projects that inspired and guided this work:  
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch.git)  
- [openai/gpt-2](https://github.com/openai/gpt-2)  



