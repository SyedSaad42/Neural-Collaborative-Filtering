# Neural Collaborative Filtering (NCF)

&#x20;&#x20;

## Overview

**Neural Collaborative Filtering (NCF)** is a neural-network-based framework that replaces traditional matrix factorization with flexible, learnable models to capture **nonlinear user–item interactions**. The NeuMF architecture fuses a **Generalized Matrix Factorization (GMF)** module with a **Multi-Layer Perceptron (MLP)** to model complementary linear and nonlinear signals from **implicit feedback** (clicks, views, purchases).

This repository contains a concise summary, evaluation protocol, and practical notes for reproducing results and integrating NCF ideas into recommendation projects.

---

## Why it matters

* Handles **implicit feedback** and extreme **sparsity** common in real-world interactions.
* Learns **nonlinear user–item relationships** beyond dot-product MF.
* Demonstrates measurable gains in **Top-K ranking** metrics (HR\@K, NDCG\@K).

**Keywords:** Neural Collaborative Filtering, NeuMF, GMF, MLP, implicit feedback, embeddings, recommender systems, HR, NDCG, deep learning, personalization.

---

## Quick start

1. Clone this repo: `git clone https://github.com/SyedSaad42/Neural-Collaborative-Filtering.git `
2. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

3. Prepare your dataset (MovieLens / Pinterest style implicit interactions). Use the provided `scripts/preprocess.py` to convert raw logs to `(user, item, timestamp)` format.
4. Train a baseline NeuMF model:

```bash
python train_neumf.py --data data/prepared --epochs 20 --batch-size 256 --embed-size 64
```

5. Evaluate with the leave-one-out protocol and negative sampling to compute HR\@10 and NDCG\@10.

---

## Architecture (high-level)

* **GMF:** Learns user & item embeddings and computes a learnable interaction (generalized inner product) to capture linear signals.
* **MLP:** Learns separate embeddings and passes the concatenated vectors through multiple dense layers (ReLU) to model higher-order nonlinear interactions.
* **NeuMF:** Concatenates/fuses outputs from GMF and MLP and applies a final prediction layer with a sigmoid for probabilistic ranking.

> Design choices: separate embeddings for GMF/MLP allow different capacities; pretraining GMF and MLP independently and then fine-tuning NeuMF improves convergence.

---

## Datasets & Evaluation

**Recommended datasets:** MovieLens (implicit), Pinterest (implicit), or any user–item interaction logs converted to implicit signals.

**Evaluation protocol:**

* Leave-one-out: hold out each user's latest interaction as the test item.
* For each test item, randomly sample 99 negative items and rank 100 items.
* Metrics: **Hit Ratio (HR\@K)** and **NDCG\@K** (commonly K=10).

---

## Results (what to expect)

* NeuMF typically **outperforms** single-component baselines (pure MF/BPR, pure MLP) on HR and NDCG for sparse implicit datasets.
* Pretraining and careful hyperparameter tuning (embedding size, depth, regularization, learning rate) significantly affect results.

---

## Strengths

* Expressive model for nonlinear patterns.
* Flexible fusion of linear (GMF) and nonlinear (MLP) signals.
* Works well on top-K recommendation tasks with implicit feedback.

---

## Limitations

* Heavier compute than simple MF for large catalogs.
* Sensitive to negative sampling strategy and hyperparameters.
* Does not natively use side information (reviews, metadata) — requires extensions.

---

## Extensions & Research Ideas

* Combine **eALS-style** element-wise weighting for negatives with learned embeddings to better handle implicit negatives.
* Use attention mechanisms or cross layers to improve fusion between GMF and MLP.
* Incorporate side information (text, images, knowledge graphs) via multimodal embeddings.
* Explore pairwise/hinge or Bayesian ranking objectives instead of pointwise log loss.

---

## Citation

If you use ideas from this summary or reproduction, cite the original paper:

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural Collaborative Filtering. *Proceedings of WWW 2017*.

