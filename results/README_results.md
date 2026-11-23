**Images used:** 1658 · **Exact duplicates removed:** 0 · **Classes:** 9

#### Split per class
| Variety | train | val | test | total |
|---|---|---|---|---|
| Ajwa | 122 | 27 | 26 | 175 |
| Galaxy | 133 | 29 | 28 | 190 |
| Medjool | 94 | 20 | 21 | 135 |
| Meneifi | 162 | 35 | 35 | 232 |
| Nabtat Ali | 124 | 26 | 27 | 177 |
| Rutab | 102 | 22 | 22 | 146 |
| Shaishe | 120 | 25 | 26 | 171 |
| Sokari | 185 | 40 | 39 | 264 |
| Sugaey | 118 | 25 | 25 | 168 |
| TOTAL | 1160 | 249 | 249 | 1658 |

#### Training setup
| Setting | Value |
|---|---|
| Model | ResNet-18 (ImageNet weights) + new Linear(512 → 9) head |
| Input size | 224×224 |
| Batch size | 32 |
| Epochs | 12 |
| Learning rate | 0.0001 (backbone) / 0.001 (head) |
| Optimizer | AdamW (weight decay 0.0001) + cosine schedule |
| Loss | CrossEntropyLoss |
| Weights kept | best validation epoch (11) |
| Training time | 1.5 min on CUDA |

#### Before vs after training (same test set, n = 249)
| Metric (test set) | Before training | After training | Change |
|---|---|---|---|
| Loss | 2.3584 | 0.1064 | -2.2520 |
| Accuracy | 0.1285 | 0.9679 | 0.8394 |
| Precision (macro) | 0.1521 | 0.9675 | 0.8154 |
| Recall (macro) | 0.1088 | 0.9713 | 0.8624 |
| F1 (macro) | 0.0646 | 0.9683 | 0.9037 |

#### Transfer learning vs training from scratch
| Model | Pretrained | Parameters (M) | Epochs | Train time (min) | Test accuracy | Test F1 (macro) |
|---|---|---|---|---|---|---|
| SmallCNN (from scratch) | no | 0.39 | 20 | 2.3 | 0.8273 | 0.8288 |
| ResNet-18 (fine-tuned) | yes (ImageNet) | 11.18 | 12 | 1.5 | 0.9679 | 0.9683 |
