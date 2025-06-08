# NLP_Final_Project

# GPT-small & Retrieval-Augmented Generation (RAG)

This repository contains the implementation of a lightweight Transformer-based language model (GPT-small) and a retrieval-augmented generation (RAG) system for open-domain question answering. The project was completed as the final assignment for CSE402 at UNIST.

---

## 📌 Overview

The project consists of two main components:

1. **GPT-small**  
   - A decoder-only Transformer model built from scratch using PyTorch.
   - Includes modern architectural features:
     - Rotary Positional Embedding (RoPE)
     - Grouped Query Attention (GQA)
     - RMSNorm
   - Pretrained using causal language modeling on a curated version of the SmolLM-12.5 Cosmopedia corpus.
   - Fine-tuned on two downstream tasks:
     - **Text Summarization** (XSum)
     - **News Classification** (AG News)

2. **Retrieval-Augmented Generation (RAG)**  
   - Combines sparse retrieval (BM25 via Pyserini) with generative models.
   - Supports both:
     - GPT-small (fine-tuned)
     - LLaMA-3.2-1B-Instruct (zero-shot)
   - Enhanced using:
     - Prompt engineering (e.g., few-shot examples)
     - Answer extraction via post-processing (e.g., JSON parsing)

---

## 🧠 Model Details

- Architecture: Decoder-only Transformer
- Parameters: ~63.58M
- Pretraining objective: Causal Language Modeling (CLM)
- Optimizations:
  - RoPE for positional encoding
  - GQA for efficient attention
  - RMSNorm for stable training

---

## 📁 Project Structure
```
.
├── model/
│   ├── model.py               # GPT-small model with RoPE, GQA, RMSNorm
│   └── rag.py                 # RAG model: retrieval + generation logic
├── dataset/
│   ├── summary.py             # XSum summarization preprocessing
│   ├── classification.py      # AG News classification preprocessing
│   └── rag.py                 # NQ dataset loading for RAG
├── cache/                     # Local dataset/model cache (auto-generated)
├── main.ipynb                 # Main notebook for pretraining, finetuning, RAG
├── README.md                  # This file
└── requirements.txt           # Dependencies (e.g., torch, transformers, pyserini)
```

## 🚀 How to Run

### 1. Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
2. Run Each Stage
모든 실행은 main.ipynb에서 플래그만 설정하면 됩니다.

✅ Pretraining GPT-small
DO_PRETRAIN = True
DO_FINETUNE_SM = False
DO_FINETUNE_CF = False
DO_FINETUNE_RAG = False
DO_ZEROSHOT_RAG = False
DO_SUBMISSION = False

✅ Fine-tuning on Summarization
DO_PRETRAIN = False
DO_FINETUNE_SM = True
DO_FINETUNE_CF = False
DO_FINETUNE_RAG = False
DO_ZEROSHOT_RAG = False
DO_SUBMISSION = False

✅ Fine-tuning on Classification
DO_PRETRAIN = False
DO_FINETUNE_SM = False
DO_FINETUNE_CF = True
DO_FINETUNE_RAG = False
DO_ZEROSHOT_RAG = False
DO_SUBMISSION = False

✅ RAG Fine-tuning with GPT-small
DO_PRETRAIN = False
DO_FINETUNE_SM = False
DO_FINETUNE_CF = False
DO_FINETUNE_RAG = True
DO_ZEROSHOT_RAG = False
DO_SUBMISSION = False

✅ Zero-shot RAG with LLaMA-3.2-1B-Instruct
DO_PRETRAIN = False
DO_FINETUNE_SM = False
DO_FINETUNE_CF = False
DO_FINETUNE_RAG = False
DO_ZEROSHOT_RAG = True
DO_SUBMISSION = False

✅ Final Submission Package
DO_PRETRAIN = False
DO_FINETUNE_SM = False
DO_FINETUNE_CF = False
DO_FINETUNE_RAG = False
DO_ZEROSHOT_RAG = False
DO_SUBMISSION = True
```

## 📦 Outputs

defaultproject_report.pdf — Final report (6~8 pages, ACL style)
defaultproject_codes.zip — All code + README
defaultproject_supplementaries.zip — Pretrained + fine-tuned model weights
Please ensure your code runs successfully in main.ipynb.

## 📚 Reference Papers

Transformer: Vaswani et al., 2017
RoPE: Su et al., 2023
GQA: Ainslie et al., 2023
RMSNorm: Zhang & Sennrich, 2019
RAG: Lewis et al., 2020
Self-RAG: Asai et al., 2024
PCW: Ratner et al., 2023
CoT Prompting: Wei et al., 2022
