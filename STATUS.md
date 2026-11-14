# Work completed — status against the 13 Nov 2026 MoM

Last updated: 14 Nov 2026. Deliverable due 18 Nov.

## Done

| MoM item | Where | Notes |
|---|---|---|
| 1-3. Common/shared feature map, shared backbone | Part 2, Step 6b | `FeatureExtractor` wraps any timm backbone into one pooled-embedding contract |
| 4, 5, 7. Percentage labelled, varying it, train/test accuracy | Part 3, Steps 13-14 | nested stratified sweep, 1% to 100%, 3 seeds at the small fractions |
| 6. Validation and test accuracy | Parts 2-3 | macro-F1, balanced accuracy, macro P/R, per-class table |
| 8. "Take as it is and check" | Part 2, Step 8 | full-data supervised baseline |
| 9. Training-set distribution, then benchmark | Parts 1-3 | class distribution in Part 1; benchmark in Part 2; accuracy vs labelled fraction in Part 3 |
| 10. Literature, models from the last year | `LITERATURE.md` | 4 papers read, 9 more located and listed |
| Model selection: 2 base + 3 new | `MODELS.md` | ResNet50, EfficientNet-B0 + MobileNetV4 (2024), DINOv3 ViT-S and ConvNeXt-T (2025) |
| Unlabelled remainder: use it or discard it | Part 3, Step 15 | both run at matched budgets — pseudo-labelling vs supervised-only |

Also complete, though not in the MoM: the data pipeline — integrity audit, exact (MD5) and
near-duplicate (dHash) removal, stratified 80/10/10 split with a hard leakage check, augmentation
policy, and a CSV audit trail for every rejected image.

## Not done

| Item | Note |
|---|---|
| Grad-CAM++ explainability | Block 8 of the original architecture. Not started. Joseph et al. (2026) in `LITERATURE.md` is the reference to work from |
| Reading the 9 located papers | `LITERATURE.md` marks exactly which were read and which were only located — the unread ones must not be cited as if read |

## Not yet run

**No model has been trained on the real dataset.** All three notebooks are written and smoke-tested
end to end on a synthetic fixture, so the code paths work, but every results table is empty until
they run. There are no checkpoints, no metrics and no figures.

Estimated T4 time:

| Stage | Cost |
|---|---|
| Part 2 benchmark, 5 backbones | 2-3 h |
| Part 2 winner on the full training split | ~1.5 h |
| Part 3 feature caching, 5 backbones | ~15 min |
| Part 3 probe sweep (all fractions x seeds) | a few minutes — runs on cached features |
| Part 3 fine-tuned points, 3 fractions x 5 backbones | 2-3 h |
| Part 3 pseudo-labelling | a few minutes |

The probe sweep being nearly free is the reason the label-efficiency curve is affordable at all.

## Dataset

PlantVillage (updated), Kaggle `tushar5harma/plant-village-dataset-updated`, CC0.
After cleaning: 16 classes, 53,590 train / 6,699 val / 6,699 test.
