# Literature review — recent models and label-efficient plant disease classification

Compiled 14 Nov 2026, covering roughly Nov 2025 to Nov 2026 plus the backbone releases the
benchmark depends on.

**Reading status is marked honestly.** "Read" means the paper text was retrieved and the numbers
below come from it. "Abstract only" means the headline figures are confirmed but the result tables
were not. "Located" means the reference is real and relevant but has not been read yet — these are
listed so they can be worked through, not cited as if they had been.

---

## 1. Where the field is: comparison papers on plant disease classification

### Hasani, Fondaj, Bytyqi & Misini (2026) — read
*Comparative evaluation of deep learning models for plant disease classification with edge-aware
performance analysis.* PLOS ONE 21(6):e0349901. Received 7 Jan 2026, published 18 Jun 2026.

The closest published work to our benchmark, and the most useful comparison point.

| Model | PlantVillage accuracy | Latency |
|---|---|---|
| YOLOv8-cls (proposed) | 99.64% ± 0.09 | 3.3 ms |
| DenseNet121 | 99.48% ± 0.12 | 18.2 ms |
| ResNet50 (baseline) | not stated in abstract | > 14 ms |
| Binarized NN | 88.31% ± 0.45 | 1.1 ms |

Dataset: a 15-class PlantVillage subset, 20,639 images (tomato, potato, pepper), stratified
70/15/15. Cross-domain check on 100 PlantDoc images across 5 classes.

Conclusion: YOLOv8-cls gives the best accuracy/latency balance; DenseNet121 matches it on accuracy
at roughly 5x the latency. Removing augmentation costs 1.8% accuracy.

**Two things to note before citing.** First, the reported PlantDoc accuracy for YOLOv8 (99.64%) is
identical to its PlantVillage accuracy while the other three collapse to 58-65%; that is unusual
enough to verify against the full tables. Second — and this is the opening for our work —
**the paper does not study limited labels at all.** Every model sees the full training set.

### Papers located but not yet read

Worth working through before the write-up; all are 2026 and on topic.

- *Evaluation of lightweight and efficient deep learning models for plant disease classification* —
  Discover IoT (Springer), `10.1007/s43926-026-00310-0`. Publisher blocked automated retrieval;
  needs manual access.
- Frontiers in Plant Science 2026, `10.3389/fpls.2026.1703449`
- Frontiers in Artificial Intelligence 2026, `10.3389/frai.2026.1773329`
- *Comparative deep learning approaches for bean leaf disease recognition*, PMC13168014
- *AI-Driven Plant Disease Detection and Diagnosis: A Comprehensive Review*, PMC13066816 — review
  article, useful for framing the introduction

---

## 2. The low-label question

This is the axis our experiment occupies, and the existing work approaches it through few-shot
meta-learning rather than through a label-fraction sweep on a standard split.

### Babu, Nair, Balusamy, Khoh & Nair (2025) — abstract only
*AgriFewNet — A Lightweight RGB Few-Shot Learning Model for Efficient Plant Disease
Classification.* Applied Sciences 15(23):12787, 3 Dec 2025. CC BY.

Hierarchical attention over a ResNet-18 backbone, MAML meta-learning, prototype classifier.
PlantVillage (54,303 images, 38 classes) split into 23 meta-train / 8 meta-val / 7 meta-test
classes.

| Setting | Accuracy |
|---|---|
| 1-shot | 87.3% |
| 5-shot | 94.8% |
| 10-shot | 97.1% |

Claims up to 7.9% over competing few-shot baselines. Result tables not yet read.

### Joseph, Pawar, Muthalagu & Mukharjee (2026) — read
*XAI and Few-shot-based Hybrid Classification Model for Plant Leaf Disease Prognosis.*
arXiv:2603.06676, 3 Mar 2026. BITS Pilani Dubai.

Hybrid Siamese-Prototypical network on a shared ResNet-18 encoder, trained episodically, with
Grad-CAM, Grad-CAM++ and EigenCAM for explanation. Baselines: Siamese (triplet), Relation Network,
Matching Networks, Prototypical Network. Episodes are 4-way, 5-support, 10-query. Classes are
**disease stages** (early / advancing / severe / healthy) across rice, wheat and maize rather than
disease identities. Reported accuracy, precision, recall and F1 "frequently exceeding 92%".

Caution: the per-class hybrid figures in the later tables repeat 0.9750 across several different
diseases and conflict with the surrounding text in places. Verify against the PDF tables before
citing any specific number.

Relevant to us twice over: it pairs few-shot learning with Grad-CAM++ explainability, which is
exactly block 8 of our architecture.

### Also located

- *Few-shot crop pests and diseases recognition based on adversarial augmentation and task
  interpolation*, Plant Methods, `10.1186/s13007-026-01530-9`
- *FewShotCropNet: Real-Time Detection of Emerging Crop Diseases with Limited Labels*, AJERD
- Frontiers in Agronomy 2026, `10.3389/fagro.2026.1918022`

### Where our experiment sits

The few-shot literature measures K-shot episodic accuracy on **held-out classes**. We measure
something different and, for a deployment argument, more directly useful: accuracy on **all 16
classes** as the labelled fraction of a fixed training set rises from 1% to 100%. That answers the
question a grower or an annotation budget actually poses — *how many labelled leaves do we need
before this is good enough* — which episodic accuracy does not. No paper found above runs that
sweep on PlantVillage with modern backbones.

---

## 3. Backbones released in the last year

The benchmark can only use what `timm` ships, so this section is restricted to architectures with
pretrained weights available in `timm` 1.0.31 (the version the notebook installs).

### DINOv3 — Siméoni et al., Meta AI Research / WRI / Inria — read (partially)
arXiv:2508.10104, 13 Aug 2025.

A self-supervised vision foundation model trained on LVD-1689M (1.689B curated images). The 7B-
parameter teacher (ViT, patch 16, RoPE, embed dim 4096, 40 blocks) is distilled into ViT-S, ViT-B,
ViT-L and ConvNeXt variants. The central contribution is **Gram anchoring**, which stops dense
features degrading over long training runs.

The claim that matters for us is that DINOv3 is designed to be used **frozen**. With a frozen
backbone the paper reports COCO detection mAP 66.1 and ADE20k mIoU 63.0, described as beating
specialised fine-tuned pipelines. Its data-curation ablation (200k iterations) gives:

| Data | IN1k k-NN | IN1k linear | ObjectNet | iNat-2021 | Paris retrieval |
|---|---|---|---|---|---|
| Raw | 80.1 | 84.8 | 70.3 | 70.1 | 63.3 |
| Clustering | 79.4 | 85.4 | 72.3 | 81.3 | 85.2 |
| Retrieval | 84.0 | 86.7 | 70.7 | 86.0 | 82.7 |
| LVD-1689M | 84.6 | 87.2 | 72.8 | 87.0 | 85.9 |

Section 6.2.1 (linear probing results) was not retrievable from the HTML version; read it from the
PDF before quoting final numbers. The paper reports no explicit few-shot results in the readable
text. Model-weight licence not confirmed — **check it before redistributing anything derived from
the weights**.

Why it belongs in this benchmark: a backbone built to work frozen is precisely what a
label-scarcity curve tests. If DINOv3 features are as strong as claimed, its advantage over
ImageNet-supervised backbones should be largest at 1% labels and shrink towards 100% — a
hypothesis our sweep can confirm or refute directly.

### MobileNetV4 — Qin et al., 2024
Available as `mobilenetv4_conv_small` (2.49M params, 1280-d features) and larger variants.
Replaces MobileNetV3-Small as the edge-deployment baseline; strictly newer and designed for
hardware-agnostic latency, which matters for the TIH-IoT framing.

### Other recent options in `timm` 1.0.31, not selected
`convnextv2` (FCMAE), `efficientvit`, `fastvit`, `repvit`, `starnet`, `shvit`, `mambaout`,
`rdnet`, `hiera`, `eva02`, `aimv2`, `naflexvit`, SigLIP-2 ViTs. Any of these is a defensible swap;
they were left out to keep the roster at the 4-5 models the MoM asks for.

---

## 4. What this means for the model roster

The MoM asks for 4-5 models, two base and two or three recent. Final selection:

| Role | Model | Year | Why |
|---|---|---|---|
| Base | ResNet50 | 2015 | universal reference point; used by Hasani et al. |
| Base | EfficientNet-B0 | 2019 | standard efficiency baseline |
| New | MobileNetV4-Conv-Small | 2024 | edge deployment, TIH-IoT relevance |
| New | DINOv3 ViT-S/16 | 2025 | self-supervised, built to be used frozen |
| New | DINOv3 ConvNeXt-Tiny | 2025 | same, convolutional |

The last two share a pretraining recipe but differ in architecture, and DINOv3 ConvNeXt-Tiny is
architecturally identical to the ImageNet-supervised ConvNeXt-Tiny kept in the registry as an
optional extra. Running both isolates **pretraining recipe** from **architecture** — a clean
ablation, and a stronger result than simply reporting which of five models wins.

---

## Sources

- [Hasani et al., PLOS ONE 2026](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0349901)
- [AgriFewNet, Applied Sciences 2025](https://doi.org/10.3390/app152312787)
- [Joseph et al., arXiv:2603.06676](https://arxiv.org/abs/2603.06676)
- [DINOv3, arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
- [Evaluation of lightweight models, Discover IoT](https://link.springer.com/article/10.1007/s43926-026-00310-0)
- [Frontiers in Plant Science 2026](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1703449/full)
- [Comparative deep learning for bean leaf disease, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13168014/)
- [AI-driven plant disease detection review, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13066816/)
- [Few-shot crop pests, Plant Methods](https://link.springer.com/article/10.1186/s13007-026-01530-9)
