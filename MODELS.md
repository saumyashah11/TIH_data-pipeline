# Model roster

Backbones loaded through `timm` 1.0.31 and wrapped in the shared `FeatureExtractor`, which gives
every architecture one contract: image batch in, pooled `(B, feat_dim)` embedding out.

Selection follows the 13 Nov MoM — 4-5 models, two base and three recent — and the reasoning is in
`LITERATURE.md`. Figures were measured by instantiating each model at 224x224, not copied from
papers.

## Active roster

| Role | Model | timm id | Year | Params (M) | feat_dim | Unfrozen (M) | % of trunk |
|---|---|---|---|---|---|---|---|
| Base | ResNet50 | `resnet50` | 2015 | 23.51 | 2048 | 14.96 | 63.7% |
| Base | EfficientNet-B0 | `efficientnet_b0` | 2019 | 4.01 | 1280 | 3.16 | 78.9% |
| New | MobileNetV4-Conv-Small | `mobilenetv4_conv_small.e2400_r224_in1k` | 2024 | 2.49 | 1280 | 2.18 | 87.3% |
| New | DINOv3 ViT-S/16 | `vit_small_patch16_dinov3.lvd1689m` | 2025 | 21.59 | 384 | 7.10 | 32.9% |
| New | DINOv3 ConvNeXt-Tiny | `convnext_tiny.dinov3_lvd1689m` | 2025 | 27.82 | 768 | 15.47 | 55.6% |

The two DINOv3 entries share a pretraining recipe and differ in architecture. DINOv3
ConvNeXt-Tiny is also architecturally identical to the ImageNet-supervised ConvNeXt-Tiny in the
extras below, so running the pair isolates **pretraining recipe** from **architecture** — a
cleaner result than a plain ranking of five models.

## Extras (in the registry, off by default)

Set `RUN_EXTRAS = True` in Step 12.

| Model | timm id | Year | Why kept |
|---|---|---|---|
| ConvNeXt-Tiny | `convnext_tiny` | 2022 | supervised twin of DINOv3 ConvNeXt-Tiny — the ablation control |
| MobileNetV3-Small | `mobilenetv3_small_100` | 2019 | superseded by MobileNetV4, kept for continuity |
| DeiT-Small | `deit_small_patch16_224` | 2021 | supervised ViT reference against DINOv3 ViT-S |

## Which blocks unfreeze

Declared per backbone rather than by a depth percentage, so "unfreeze the top stage" is the same
architectural decision each time.

| Model | Parameter-name fragments |
|---|---|
| ResNet50 | `layer4` |
| EfficientNet-B0 | `blocks.5`, `blocks.6`, `conv_head`, `bn2` |
| MobileNetV4-Conv-Small | `blocks.3`, `blocks.4`, `conv_head`, `norm_head` |
| DINOv3 ViT-S/16 | `blocks.8` to `blocks.11`, `norm.` |
| DINOv3 ConvNeXt-Tiny | `stages.3`, `head.` |

Note for the write-up: the same phrase means very different things by parameter count. Unfreezing
the top stage moves 33% of DINOv3 ViT-S but 87% of MobileNetV4-Conv-Small, because the small CNNs
concentrate their parameters near the head. Any claim about "fine-tuning the last stage" should
carry this table beside it.

## Learning rates

The DINOv3 trunks fine-tune at half the usual backbone rate (`LR_MULT` in Step 12). Their weights
are self-supervised rather than tuned for classification, and at small label budgets the standard
rate disturbs more than it repairs. Heads are unaffected.

## Three implementation details that bit us

1. **`trunk.num_features` is not the embedding width.** MobileNetV3-Small reports 576 but emits
   1024 with `num_classes=0`. The extractor measures the width with a dummy forward instead of
   trusting the attribute; trusting it builds a head of the wrong size and fails at the first batch.
2. **Frozen BatchNorm still moves.** A BN layer with frozen weights keeps updating its running
   mean and variance in `train()` mode, drifting the frozen trunk while the top is fine-tuned.
   `freeze_frozen_bn()` returns every fully-frozen BN to `eval()`. Affects the CNNs; the ViT uses
   LayerNorm and is unaffected.
3. **Normalisation is per checkpoint.** Each model keeps its own mean/std and `crop_pct`, read
   from `timm.data.resolve_model_data_config`. Forcing one normalisation across all five would
   handicap whichever models were not trained under it and invalidate the comparison.

## Licensing

DINOv3 weights carry Meta's own model licence, which was not confirmed while compiling
`LITERATURE.md`. **Check it before redistributing any derived checkpoint**, and before including
DINOv3-derived weights in anything submitted or published.
