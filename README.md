# StructDiff-SAR

### Physics-Aware Transformer-Guided Diffusion for SAR Image Despeckling

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.x-ee4c2c.svg)](#)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#)
[![Status](https://img.shields.io/badge/status-research--in--progress-yellow.svg)](#)

StructDiff-SAR is a research framework for Synthetic Aperture Radar (SAR)
image despeckling built on top of a Denoising Diffusion Probabilistic
Model (DDPM) backbone. It extends the original SAR-DDPM formulation with
a modular conditioning system (multi-look, structure-tensor, multi-scale
structure-tensor, spectral, and wavelet features), a set of despeckling-
specific losses, a configurable multi-mode sampling framework, and a
physics-aware Transformer bottleneck in which self-attention is biased by
relation matrices derived from the local structure tensor of the input
scene. The goal of this repository is to explore whether attention that
is explicitly informed by classical, interpretable SAR structure priors
can improve despeckling quality and edge/structure preservation relative
to a purely learned attention mechanism.

This repository is an independent research codebase, not a fork; it
builds on and credits the original SAR-DDPM and OpenAI guided-diffusion
projects (see [Acknowledgements](#acknowledgements)).

---

## Table of Contents

- [Highlights](#highlights)
- [Motivation](#motivation)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Implemented Modules](#implemented-modules)
- [Physics-Aware Attention](#physics-aware-attention)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Testing](#testing)
- [Configuration](#configuration)
- [Repository Timeline](#repository-timeline)
- [Current Status](#current-status)
- [Future Work](#future-work)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Modular conditioning framework** — multi-look, structure-tensor,
  multi-scale structure-tensor, spectral-tensor, and wavelet conditioning
  signals are each implemented as independent, swappable encoders that
  contribute to the diffusion timestep embedding.
- **Despeckling-specific losses** — an edge-aware loss and a structure
  consistency loss (computed via a zero-cost `x0`-prediction hook into
  the existing UNet forward pass, rather than a second forward pass) are
  available in addition to the standard diffusion objective.
- **Configurable multi-mode sampling framework** — a `sampling/` package
  implementing several inference-time schedule and aggregation
  strategies, including a spatially-aware cycle-spinning aggregator.
- **Physics-aware Transformer bottleneck** — a Transformer block
  appended to the UNet's middle block, whose self-attention accepts an
  optional additive bias term derived from the input structure tensor.
- **Structure-tensor-derived attention bias** — orientation and
  coherence are extracted from the structure tensor, converted into a
  pairwise relation matrix, and fused (via learnable, zero-initialized
  gates) into an attention bias that is spatially aligned to the
  Transformer's token grid before injection.
- **Checkpoint-compatible integration** — the physics-aware pathway is
  additive and zero-initialized end-to-end, so it activates as a no-op
  extension of the existing architecture rather than a breaking change.

## Motivation

**SAR speckle.** Synthetic Aperture Radar imagery is corrupted by
speckle, a multiplicative, spatially correlated noise pattern that
arises from coherent imaging. Unlike additive sensor noise in optical
imagery, speckle interacts with local texture and structure in ways that
make naive denoising prone to over-smoothing edges and fine linear
features (roads, field boundaries, coastlines) that carry most of the
scene's interpretable content.

**Why diffusion.** Diffusion probabilistic models have shown strong
results for image restoration tasks by learning to iteratively reverse a
noising process, which gives them an implicit, learned prior over
natural image statistics without requiring an explicit noise model. This
makes them a natural fit for despeckling, where the degradation process
is well understood physically but difficult to invert in closed form.

**Why Transformers.** Convolutional denoising networks have a limited
receptive field per layer and rely on depth to aggregate long-range
context. SAR structures such as roads, rivers, and coastlines are
long-range and often separated by speckle-dominated regions, which is
exactly the kind of long-range, content-addressable dependency that
self-attention is well suited to model directly.

**Why structure tensors.** The 2-D structure tensor is a classical,
well-understood, physically interpretable descriptor of local image
structure: its eigenvectors and eigenvalues directly encode orientation,
coherence, and anisotropy of local gradient energy. It provides a
principled way to describe "does this pixel lie on an oriented edge, and
how confidently" without requiring any learned parameters.

**Why physics-aware attention.** Combining the two: if two spatial
locations lie along the same physically coherent oriented structure, it
is plausible that attention between them should be encouraged, and
attention between locations that lie on unrelated, incoherent, or
orthogonal structures should not be favored purely on the basis of
learned content similarity. Physics-aware attention operationalizes this
by adding a structure-tensor-derived bias directly to the pre-softmax
attention scores, alongside — not in place of — ordinary learned
attention.

## Method Overview

At a high level, the pipeline is:

```
SAR Image
     │
     ▼
Multi-look / Structure / Multi-scale Structure / Spectral / Wavelet
Conditioning Encoders
     │
     ▼
Timestep + Conditioning Embedding
     │
     ▼
Diffusion UNet (input blocks → bottleneck → output blocks)
     │
     ▼
UNet Bottleneck:
  ResBlock → AttentionBlock → Physics-Aware Transformer Block → ResBlock
     │
     ▼
Cycle-Spinning Aggregation (inference)
     │
     ▼
Despeckled SAR Image
```

The physics-aware pathway that feeds the Transformer block is itself a
short pipeline, run once per forward pass when a structure tensor is
available:

```
Raw Structure Tensor (input resolution)
     │
     ▼
Spatial alignment to bottleneck resolution (bilinear interpolation)
     │
     ▼
Orientation / Coherence / Anisotropy Extraction
     │
     ▼
Pairwise Orientation Relation Matrix
     │
     ▼
Gated Fusion (+ optional wavelet / spectral / confidence relations)
     │
     ▼
Physics Attention Bias
     │
     ▼
Injected additively into Physics-Aware Attention (pre-softmax)
```

*An architecture diagram will be added to this section in a future
update.*

## Repository Structure

```
.
├── guided_diffusion/       # Diffusion UNet, training loop, and sampling
│                            # infrastructure (based on OpenAI's
│                            # guided-diffusion), including the UNet
│                            # bottleneck integration of the
│                            # physics-aware Transformer block.
├── structdiff/
│   ├── conditioning/        # Multi-look, structure-tensor, multi-scale
│   │                        # structure-tensor, spectral-tensor, and
│   │                        # wavelet conditioning encoders.
│   ├── transformer/          # OrientationExtractor, PhysicsRelationMatrix,
│   │                        # PhysicsBiasFusion, PhysicsAttentionBiasBuilder,
│   │                        # PhysicsAwareAttention, TransformerBlock, and
│   │                        # PhysicsTransformerBlock.
│   ├── losses/               # Edge-aware loss, structure consistency loss.
│   └── sampling/             # Ultimate Sampling Framework (USF): schedule
│                             # generators, spatial feature extractors,
│                             # SamplingController, CycleSpinningEngine /
│                             # CycleSpinAggregator.
├── scripts/                  # Training / inference entry points.
├── weights/                  # Model checkpoints (not tracked in git).
└── archive/                  # Deprecated or superseded implementations,
                              # kept for reference during development.
```

*Exact folder contents may evolve; this section reflects the structure
as of the current physics-aware Transformer integration phase.*

## Implemented Modules

| Module | Description | Status |
|---|---|---|
| Multi-look conditioning | Embeds the number of SAR looks into the diffusion timestep embedding. | Implemented |
| Structure tensor conditioning | Embeds a single-scale structure tensor into the timestep embedding. | Implemented |
| Multi-scale structure tensor conditioning | Embeds structure tensors computed at multiple scales. | Implemented |
| Spectral tensor conditioning | Embeds spectral-domain tensor features. | Implemented |
| Wavelet conditioning | Embeds wavelet-domain features. | Implemented |
| Edge-aware loss | Auxiliary loss term encouraging edge preservation. | Implemented |
| Structure consistency loss (A33) | Auxiliary loss comparing predicted structure to target structure, computed via a zero-cost `x0`-prediction hook (`EpsInterceptHook`) into the existing UNet forward pass. | Implemented |
| Advanced DDIM schedules | Alternative deterministic sampling schedules. | Implemented |
| Adaptive beta controller | Adaptive noise-schedule control at inference. | Implemented |
| Ultimate Sampling Framework (USF) | Modular `sampling/` package unifying schedule generation, spatial feature extraction, and a `SamplingController` with multiple orchestration modes. | Implemented |
| Confidence-guided cycle spinning | Cycle-spinning aggregation weighted by per-pixel confidence. | Implemented |
| Learnable / adaptive cycle spinning | Cycle-spinning variants with learned or adaptive shift selection. | Implemented |
| RL-based sampling orchestration | Reinforcement-learning-guided sampling mode. | Reserved (not yet active) |
| `OrientationExtractor` | Converts a raw structure tensor into orientation, coherence, and anisotropy maps. | Implemented |
| `PhysicsRelationMatrix` | Computes a pairwise orientation relation matrix from orientation and coherence. | Implemented |
| `PhysicsBiasFusion` | Fuses one or more relation matrices into a single attention bias via learnable, zero-initialized gates. | Implemented |
| `PhysicsAttentionBiasBuilder` | Orchestrates the above three modules into a single call from a raw structure tensor to a fused attention bias. | Implemented |
| `PhysicsAwareAttention` | Multi-head self-attention accepting an optional additive attention bias. | Implemented |
| `TransformerBlock` / `PhysicsTransformerBlock` | Pre-norm Transformer encoder blocks operating on `[B, C, H, W]` feature maps; the physics-aware variant threads the physics attention bias through `PhysicsAwareAttention`. | Implemented |
| UNet bottleneck integration | `PhysicsTransformerBlock` appended after the existing `AttentionBlock` in the UNet middle block; `TimestepEmbedSequential` extended to route an optional physics attention bias alongside the existing timestep embedding routing. | Implemented |
| Spatial alignment of physics bias | Structure tensor is bilinearly resized to the bottleneck's spatial resolution before the attention bias is built, so the bias is computed over the same token grid the Transformer attends over. | Implemented |
| Wavelet / spectral / confidence relation fusion | `PhysicsBiasFusion` already exposes gated slots for these relation sources. | Interface implemented; relation-matrix sources not yet wired in |

## Physics-Aware Attention

The physics-aware attention pathway is designed so that each stage owns
exactly one piece of domain logic, and no stage duplicates another's
computation:

```
Structure Tensor
       │
       ▼
OrientationExtractor        →  orientation, coherence, anisotropy
       │
       ▼
PhysicsRelationMatrix        →  pairwise orientation relation matrix [B, N, N]
       │
       ▼
PhysicsBiasFusion             →  gated sum of available relation matrices
       │
       ▼
Physics Attention Bias        →  [B, N, N], additive, pre-softmax
       │
       ▼
PhysicsAwareAttention          →  standard multi-head attention + bias
```

`PhysicsAttentionBiasBuilder` orchestrates the first three stages behind
a single call, so the UNet only needs to know that a structure tensor
goes in and a bias comes out — it does not need to know about
orientation extraction or relation-matrix construction individually.

Each modality contributed to `PhysicsBiasFusion` (currently orientation;
wavelet, spectral, and confidence relations are accepted by the
interface but not yet populated by an upstream relation-matrix source)
is weighted by an independent, learnable scalar gate initialized to
zero. At initialization, the physics attention bias is therefore exactly
zero for any input, and `PhysicsTransformerBlock` reduces to ordinary
self-attention — the physics pathway can only begin to influence
attention once training moves the corresponding gate(s) away from zero.
This is a deliberate zero-init warm-start design choice, not an
incidental one: it means the physics-aware pathway can be enabled on top
of an existing, previously-trained checkpoint without changing that
checkpoint's forward pass at the moment of insertion.

Because the Transformer bottleneck attends over the spatial resolution
of the bottleneck feature map (not the input image resolution), the
structure tensor is bilinearly resized to that resolution before the
bias is built, so the physics prior and the Transformer's token grid are
in spatial correspondence.

No claim is made here about the *effect* of this mechanism on
despeckling quality; that is an empirical question this repository has
not yet evaluated (see [Current Status](#current-status)).

## Installation

### Linux

```bash
git clone <repository-url>
cd StructDiff-SAR

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Windows (via WSL)

```bash
wsl --install
# inside the WSL shell:
git clone <repository-url>
cd StructDiff-SAR

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Python version:** 3.x (see `requirements.txt` for the exact pinned
version and dependency set, including the PyTorch build required for
your CUDA toolkit if training on GPU).

## Dataset

The framework is developed and evaluated against SAR imagery in the
**DSIFN** dataset layout. Expected directory structure:

```
data/
└── DSIFN/
    ├── train/
    │   ├── images/
    │   └── ... (per-sample structure-tensor / auxiliary conditioning inputs, if precomputed)
    ├── val/
    └── test/
```

Exact subfolder naming should match whatever your dataset loader in
`guided_diffusion/` or `scripts/` expects — consult the loader
implementation for the authoritative layout. No dataset download links
are included here; obtain DSIFN (or your SAR dataset of choice) through
its original distribution channel.

## Training

```bash
python scripts/train.py \
    --data_dir /path/to/DSIFN/train \
    --image_size 256 \
    --batch_size <batch_size> \
    --lr <learning_rate> \
    --num_res_blocks <num_res_blocks> \
    --attention_resolutions <attention_resolutions>
```

Adjust flags to match the arguments exposed by your training entry
point and the model configuration you intend to run (baseline vs.
physics-aware Transformer bottleneck, which conditioning encoders are
active, which losses are enabled, etc.). See
[Configuration](#configuration) for where these are defined.

## Testing

```bash
python scripts/sample.py \
    --model_path /path/to/checkpoint.pt \
    --data_dir /path/to/DSIFN/test \
    --image_size 256 \
    --num_samples <num_samples>
```

Use the appropriate sampling entry point for the schedule / aggregation
mode you want to evaluate (standard DDIM, cycle-spinning aggregation,
etc.), consistent with the `sampling/` package's supported modes.

## Configuration

Model, training, and sampling hyperparameters are defined via the
argument defaults in `guided_diffusion/script_util.py` (e.g.
`sr_model_and_diffusion_defaults`) and overridable via command-line
flags to the scripts in `scripts/`. Conditioning-specific settings live
alongside each encoder in `structdiff/conditioning/`; loss weighting and
schedule/sampling configuration live in `structdiff/losses/` and
`structdiff/sampling/` respectively.

## Repository Timeline

```
Baseline SAR-DDPM reproduction
        │
        ▼
Multi-look conditioning
        │
        ▼
Structure tensor conditioning (single-scale, then multi-scale)
        │
        ▼
Spectral and wavelet conditioning
        │
        ▼
Despeckling-specific losses (edge-aware, structure consistency)
        │
        ▼
Ultimate Sampling Framework (schedules, cycle spinning, orchestration)
        │
        ▼
Transformer-enhanced diffusion bottleneck
        │
        ▼
Physics-aware attention framework
(OrientationExtractor → PhysicsRelationMatrix → PhysicsBiasFusion
 → PhysicsAttentionBiasBuilder → PhysicsAwareAttention)
        │
        ▼
UNet integration + spatial alignment of the physics attention bias
```

## Current Status

- Architecture implementation of the physics-aware Transformer
  bottleneck is complete and structurally integrated into the UNet.
- The physics attention bias is spatially aligned to the bottleneck
  resolution and injected additively into self-attention.
- The training pipeline is operational for both the baseline and
  physics-aware configurations.
- Quantitative experimental evaluation (ablations isolating each
  conditioning signal, loss term, and the physics-aware attention
  pathway) is ongoing. This README will be updated with results once
  experiments are complete — no benchmark numbers are reported here yet.

## Future Work

**Implemented, evaluation pending:**
- Ablation study isolating the contribution of the physics-aware
  Transformer bottleneck versus the baseline attention block.
- Ablation study isolating the structure consistency loss.

**Planned, not yet implemented:**
- Wiring wavelet- and spectral-domain relation matrices into
  `PhysicsBiasFusion`'s existing (currently unused) fusion slots.
- Physics-guided pairwise relation learning beyond orientation
  agreement (e.g. confidence-weighted cross-modal relations).
- Dynamic / learned physics priors that adapt across diffusion
  timesteps rather than a single static bias per forward pass.
- Cross-scale physics-aware attention spanning multiple bottleneck
  resolutions.
- Lightweight / deployment-oriented inference variant.
- Evaluation on additional SAR datasets beyond DSIFN.

## Citation

A citation entry will be added upon publication. In the meantime:

```bibtex
@misc{structdiffsar,
  title        = {StructDiff-SAR: Physics-Aware Transformer-Guided Diffusion for SAR Image Despeckling},
  author       = {TODO: Author list},
  year         = {2026},
  howpublished = {\url{TODO: repository URL}},
  note         = {Work in progress}
}
```

## Acknowledgements

This repository builds on and gratefully acknowledges:

- **[Guided Diffusion](https://github.com/openai/guided-diffusion)**
  (OpenAI) — the diffusion UNet and training infrastructure this
  repository's `guided_diffusion/` package is based on.
- **SAR-DDPM** — the original diffusion-based SAR despeckling
  formulation this project extends.
- **SAR-DDPM Aggregation** — prior work on cycle-spinning aggregation
  for diffusion-based SAR despeckling that informed this repository's
  sampling framework.

StructDiff-SAR extends these works with a modular multi-signal
conditioning framework, despeckling-specific losses, a configurable
multi-mode sampling framework, and a physics-aware Transformer
bottleneck driven by structure-tensor-derived attention bias.
