# Implementing BERT from Scratch for Fake News Classification

**Course:** NLP – CS60075, Spring 2026, IIT Kharagpur
**Authors:** Ritvik Sahu (25CS60R19), Manav Chaniyara (25CS60R87)
**Instructor:** Prof. Saptarshi Ghosh

## Overview

This project implements an encoder-only Transformer (BERT) from scratch in PyTorch to build intuition for its internal mechanics, then uses a pre-trained `bert-base-uncased` model to classify Reddit posts from the **Fakeddit** dataset as fake or not fake. Model predictions are explained at the token level using Integrated Gradients (Captum).

The work is organized into four tasks, all contained in a single notebook: `nlp_assignment_3.ipynb`.

## Contents

| File | Description |
|---|---|
| `nlp_assignment_3.ipynb` | Full implementation: scratch BERT, data preprocessing, fine-tuning, and Integrated Gradients |
| `Report.pdf` | Written report with results, tables, and figures |

## Task Breakdown

### Task 1 — BERT from Scratch
Implements the core components of BERT using PyTorch, built up as an object-oriented stack:
- `MultiHeadSelfAttention` — scaled dot-product attention across multiple heads
- `FeedForwardLayer` — position-wise feed-forward network with GELU activation
- `TransformerEncoderLayer` — self-attention + FFN with residual connections and layer norm
- `BertEmbeddings` — token, positional, and token-type embeddings
- `BertModel` — stack of encoder layers with a pooler and binary classification head

**Configuration:** embedding dim = 512, heads = 8, layers = 2, feed-forward dim = 2048, vocab size = 30,522 (from `bert-base-uncased` tokenizer).

The model is run on a sample sentence (tokenized with the Hugging Face `bert-base-uncased` tokenizer) to verify tensor shapes at every stage — token/positional embeddings, attention output, feed-forward output, encoder output, and classifier probabilities — and to report the shape of every learnable parameter (~22.46M total).

### Task 2 — Data Preprocessing
Prepares the **Fakeddit** dataset for classification:
1. Loads and combines `all_train.tsv`, `all_test_public.tsv`, and `all_validate.tsv`, keeping only `clean_title`, `2_way_label`, and `id`.
2. Builds a class-balanced subset by sampling 2,500 fake and 2,500 non-fake posts.
3. Splits the balanced set into 80% train / 20% test with stratification.
4. Reports dataset statistics (original size, class balance, train/test distribution).

### Task 3 — Fine-Tuning BERT
Fine-tunes `BertForSequenceClassification` (`bert-base-uncased`) on the balanced dataset.

**Hyperparameters:**
| Setting | Value |
|---|---|
| Max token length | 128 (trailing tokens truncated) |
| Batch size | 32 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight decay 0.01 on non-bias params) |
| Scheduler | Linear warmup (10%) + linear decay |
| Loss | Cross-entropy |
| Gradient clipping | Max norm 1.0 |

Reports training/validation loss and accuracy curves, plus test-set accuracy, precision, recall, F1-score, and a confusion matrix.

### Task 4 — Integrated Gradients for Token Attribution
Explains predictions of the fine-tuned model using Captum's `LayerIntegratedGradients` applied to the `bert.embeddings.word_embeddings` layer:
- Baseline: all-zero token IDs
- Target: true class label per example
- Attribution aggregated via L2 norm across the embedding dimension, then min-max normalized
- Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) excluded from interpretation

Run on 3 fake and 3 not-fake test examples, printing and plotting the top-5 attributed tokens per example, along with the Integrated Gradients convergence delta.

## Results Summary

| Metric | Value |
|---|---|
| Test Accuracy | 0.8050 |
| Test Precision | 0.8032 |
| Test Recall | 0.8080 |
| Test F1-score | 0.8056 |
| Scratch BERT parameters | 22,459,906 |

Full metrics tables, confusion matrix, training curves, and Integrated Gradients visualizations are in `Report.pdf`.

## Requirements

- Python 3.x
- `torch`
- `transformers` (Hugging Face)
- `captum`
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn`

Install with:
```bash
pip install torch transformers captum numpy pandas matplotlib scikit-learn
```

## Data

This project uses the [Fakeddit](https://github.com/entitize/Fakeddit) dataset (`all_train.tsv`, `all_test_public.tsv`, `all_validate.tsv`). Update the `DATA_DIR` path in the notebook to point to your local copy of the dataset before running Task 2 onward.

## Running the Notebook

1. Set `DATA_DIR` in the Task 2 section to the folder containing the Fakeddit TSV files.
2. Run all cells sequentially — Task 1 is self-contained (no dataset needed), while Tasks 2–4 depend on the Fakeddit files.
3. A GPU (e.g., via Google Colab or Kaggle) is recommended for fine-tuning in Task 3 and running Integrated Gradients in Task 4.

## Notes

- The scratch BERT implementation (Task 1) is used only to verify architecture and tensor shapes; it is **not** the model fine-tuned in Task 3. Fine-tuning uses the pre-trained `BertForSequenceClassification` from Hugging Face.
- Convergence deltas for Integrated Gradients were not close to zero at `n_steps=50` for several examples; increasing `n_steps` (e.g., to 100–200) would improve attribution stability, as noted in the report.
