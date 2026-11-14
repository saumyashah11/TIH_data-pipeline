# TIH Data Pipeline — plant disease classification

Draft. Written 14 Nov 2026 as a starting point; edit freely. Anything marked TODO is a number
that does not exist yet because the pipeline has not been run on the full dataset.

Agricultural image pipeline and transfer-learning benchmark for plant disease classification,
built under the TIH-IoT fellowship. Two halves: a reproducible data pipeline that turns the raw
PlantVillage archive into clean, leakage-free splits, and a two-stage transfer-learning harness
that benchmarks several ImageNet backbones under one identical recipe.

**Team:** Anushree Devarashetty, Saumya Shah — MPSTME, NMIMS Mumbai

## Dataset

PlantVillage (updated) — Kaggle `tushar5harma/plant-village-dataset-updated`, CC0.
After cleaning: **16 classes**, 53,590 train / 6,699 val / 6,699 test.

## Part 1 — Data pipeline (Steps 0-5)

Nothing is deleted in place. Every rejected image is logged to CSV with its reason, so the
cleaning decisions stay auditable and reproducible (seed = 42).

| Step | What it does |
|---|---|
| 0 | Environment and global config |
| 1 | Extract the archive (refusing any entry that escapes the target folder), index images, class = parent folder |
| 2a | Integrity audit — verify + full decode, catching empty, truncated, corrupt and undersized files |
| 2b | Exact duplicates by MD5, near-duplicates by dHash; flags byte-identical images filed under different classes as label noise |
| 2c | Class distribution, rare-class drop, cleaning report |
| 3 | Stratified 80/10/10 split on the cleaned index, materialised at 256 px |
| 4 | Per-split transforms, `ImageFolder` datasets, class-balanced `WeightedRandomSampler` |
| 5 | Sanity checks — tensor shape, value range, hard leakage check, visual grid |

Two choices worth noting. The split happens **after** de-duplication, so a test image can never be
a duplicate of a training image — the most common cause of inflated PlantVillage accuracy. And
images are stored at 256 px rather than the 224 px the model consumes, so `RandomResizedCrop` has
real pixels to work with at train time and evaluation still gets a clean deterministic centre crop.

## Part 2 — Transfer learning and benchmark (Steps 6-11)

Every backbone goes through two stages under an identical recipe.

| | Stage 1 — frozen feature extraction | Stage 2 — fine-tuning |
|---|---|---|
| Trunk | frozen | top blocks unfrozen |
| Input | deterministic, no augmentation | full augmentation |
| Trains | classifier head only | head + top blocks, discriminative LRs |
| Cost | one forward pass, cached; seconds per epoch | full backward pass; minutes per epoch |
| Measures | how separable the classes already are in ImageNet feature space | what the backbone gains by adapting |

The shared layer is `FeatureExtractor`: it wraps any timm backbone so all of them emit a pooled
`(B, feat_dim)` embedding, and nothing downstream branches on architecture. Stage 1 also pays for
itself twice — the head it trains is the warm start for stage 2, so fine-tuning never begins with
a random classifier pushing large gradients into pre-trained weights.

Deliberately **not** shared: each checkpoint's own normalisation and crop ratio, and which blocks
unfreeze. Sharing either would invalidate the comparison. See `MODELS.md`.

Stage 2 ranks the backbones on a stratified 25% slice for 5 epochs each; the winner is then
retrained on 100% of train. Model selection is on validation macro-F1, not accuracy — the classes
are imbalanced. Results are written after each backbone, so a Colab disconnect costs one model
rather than the run.

## Models

Five ImageNet-pretrained backbones: ResNet50, EfficientNet-B0, MobileNetV3-Small, DeiT-Small,
ConvNeXt-Tiny. Full table with parameter counts, embedding widths and unfreeze sets in
`MODELS.md`.

## Results

TODO — the pipeline has not been run on the full dataset yet. The notebook is smoke-tested end to
end; every results table in it is empty until it runs.

## Outputs the run produces

`features/` cached frozen embeddings · `audit/10_linear_probe.csv` · `audit/11_benchmark.json`
and `.csv` · `audit/12_per_class_test.csv` · `audit/12_final_report.json` (reproduces the whole
run) · `checkpoints/final_<winner>.pt` · four figures for the paper.

## Status and what is next

See `STATUS.md` for the full mapping against the 13 Nov MoM. In short: the shared backbone and
the benchmark harness are built; the labelled-fraction sweep (1%-50%) and the literature review
of recent models are not. Grad-CAM++ explainability (block 8 of the architecture) is also
outstanding.

## Running it

Open `notebooks/tih.ipynb` in Colab with a GPU runtime and run top to bottom. Part 2 can also be
run standalone from `notebooks/tih_part2_transfer_learning.ipynb` — set `PROCESSED_ZIP` at the top
of Step 6a to the processed dataset saved to Drive.

Roughly 2-3 h on a T4 for the five-backbone benchmark, plus about 1.5 h for the final model.
