## Scripts

| File                          | Backbone                              | Notes                                                                                 |
|-------------------------------|----------------------------------------|---------------------------------------------------------------------------------------|
| `DeBERTa-base.py`             | `microsoft/deberta-base`               | CLS pooling → Dropout → Dense(256, ReLU) → Dense(6); Smooth L1; cosine schedule       |
| `DeBERTa-v3-base.py`          | `microsoft/deberta-v3-base`            | Same head as above; prints MCRMSE and saves best checkpoint                           |
| `DeBERTa-v3-base_SiFT.py`     | `microsoft/deberta-v3-base` (SiFT)     | Self-attentive pooling (attention weights over tokens) before the regression head     |
| `RoBERTa-base.py`             | `roberta-base`                         | Standard CLS pooling head; Smooth L1; cosine schedule                                 |
| `ELECTRA-base.py`             | `google/electra-base-discriminator`    | Standard CLS head; Smooth L1; cosine schedule                                         |
| `XLNet_base.py`               | `xlnet-base-cased`                     | Uses `token_type_ids` and CLS-style pooling; Smooth L1; cosine schedule               |

### Data and outputs
Set paths via env vars (defaults shown in code):
- `DATA_DIR` (default `/cluster/datastore/abdelazq`) must contain `train.csv`, `test.csv`, `sample_submission.csv`
- `OUTPUT_DIR` (default `/cluster/datastore/abdelazq/AES`) will store checkpoints and logs

Features

End-to-end training and evaluation pipelines (PyTorch + Hugging Face Transformers)

Multi-trait regression head for six analytic scores
cohesion, syntax, vocabulary, phraseology, grammar, conventions

Metric: MCRMSE with per-trait RMSE reporting

Reproducible configs per backbone (DeBERTa-base, DeBERTa-v3-base, DeBERTa-v3+SiFT, RoBERTa-base, ELECTRA-base, XLNet-base)

Mixed precision, cosine LR schedule, AdamW, fixed seeds

EDA notebooks: length, complexity, polarity/subjectivity, token distributions

Optional Weights & Biases (W&B) run tracking


### One-liners

```bash
# DeBERTa-base
python DeBERTa-base.py

# DeBERTa-v3-base
python DeBERTa-v3-base.py

# DeBERTa-v3 with SiFT pooling
python DeBERTa-v3-base_SiFT.py

# RoBERTa-base
python RoBERTa-base.py

# ELECTRA-base
python ELECTRA-base.py

# XLNet-base
python XLNet_base.py
