import torch
import torch.nn.functional as F
from pathlib import Path

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
)
from guided_diffusion.unet import UNetModel
from guided_diffusion.test_util import _prepare_conditions, _build_model_kwargs

CKPT = "logs/train_2026-09-29-09-55-11/model_best.pt"
DATASET = "/dgxa_home/se24ucse043/datasets/SAR_despeckling_dataset/DSIFN/DSIFN/val"

print("=" * 80)
print("PHYSICS CONTRIBUTION DIAGNOSTIC")
print("=" * 80)
print("Checkpoint:", CKPT)

# ------------------------------------------------------------------
# 1. Build EXACT training architecture
# ------------------------------------------------------------------
defaults = sr_model_and_diffusion_defaults()

cfg = dict(defaults)
cfg.update({
    "large_size": 256,
    "small_size": 256,
    "in_channels": 3,
    "num_channels": 192,
    "num_res_blocks": 2,
    "num_heads": 4,
    "num_heads_upsample": -1,
    "num_head_channels": -1,
    "attention_resolutions": "32,16,8",
    "learn_sigma": True,
    "resblock_updown": True,
    "use_scale_shift_norm": True,
})

model, diffusion = sr_create_model_and_diffusion(**cfg)

state = torch.load(CKPT, map_location="cpu")
model.load_state_dict(state, strict=True)

device = dist_util.dev()
model.to(device)
model.eval()

print("\n[1] MODEL")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Checkpoint load: PASS")

# ------------------------------------------------------------------
# 2. Locate physics module
# ------------------------------------------------------------------
builder = getattr(model, "physics_attention_bias_builder", None)

if builder is None:
    raise RuntimeError("physics_attention_bias_builder NOT FOUND")

fusion = getattr(builder, "physics_bias_fusion", None)

if fusion is None:
    raise RuntimeError("physics_bias_fusion NOT FOUND")

print("\n[2] PHYSICS MODULE")
print("Builder:", type(builder).__name__)
print("Fusion:", type(fusion).__name__)

# ------------------------------------------------------------------
# 3. Reset diagnostic state
# ------------------------------------------------------------------
for obj in [builder, fusion]:
    for name in [
        "last_relation_norms",
        "last_contribution_norms",
        "last_attention_shift",
        "last_bias_norm",
    ]:
        if hasattr(obj, name):
            setattr(obj, name, None)

# ------------------------------------------------------------------
# 4. Get one real DSIFN sample
# ------------------------------------------------------------------
from scripts.datasets import SynthSARDataset

dataset = SynthSARDataset(
    DATASET,
    train=False,
    num_channels=3,
    crop_size=(256, 256),
    seed=42,
)

item = dataset[0]

print("\n[3] DATASET")
print("Returned type:", type(item))

if isinstance(item, dict):
    print("Keys:", item.keys())
    clean = item.get("clean")
    noisy = item.get("noisy")
    conditions = item.get("conditions", item)

elif isinstance(item, (tuple, list)):
    print("Length:", len(item))
    for i, x in enumerate(item):
        if torch.is_tensor(x):
            print(i, "tensor", tuple(x.shape))
        else:
            print(i, type(x))

    clean = item[0]
    noisy = item[1]
    conditions = item[3]

else:
    raise RuntimeError("Unexpected dataset item format")

if clean is None or noisy is None:
    raise RuntimeError(
        "Could not identify clean/noisy tensors from dataset item."
    )

if clean.ndim == 3:
    clean = clean.unsqueeze(0)

if noisy.ndim == 3:
    noisy = noisy.unsqueeze(0)

clean = clean.to(device)
noisy = noisy.to(device)

print("Clean:", tuple(clean.shape))
print("Noisy:", tuple(noisy.shape))

# ------------------------------------------------------------------
# TRACE THE REAL PHYSICS PATH
# ------------------------------------------------------------------
print("\n[TRACE] Installing runtime hooks...")

orig_builder_forward = builder.forward
orig_fusion_forward = fusion.forward

def traced_builder_forward(*args, **kwargs):
    print("\n>>> PHYSICS BIAS BUILDER CALLED <<<")

    print("Builder positional args:", len(args))
    for i, x in enumerate(args):
        if torch.is_tensor(x):
            print(
                f"  arg[{i}] shape={tuple(x.shape)} "
                f"dtype={x.dtype} "
                f"norm={float(x.detach().float().norm()):.6f}"
            )
        else:
            print(f"  arg[{i}] type={type(x).__name__}")

    for k, x in kwargs.items():
        if torch.is_tensor(x):
            print(
                f"  kw[{k}] shape={tuple(x.shape)} "
                f"dtype={x.dtype} "
                f"norm={float(x.detach().float().norm()):.6f}"
            )
        else:
            print(f"  kw[{k}] type={type(x).__name__}")

    out = orig_builder_forward(*args, **kwargs)

    print(">>> BUILDER OUTPUT <<<")
    if torch.is_tensor(out):
        print(
            "shape =", tuple(out.shape),
            "norm =", float(out.detach().float().norm()),
            "finite =", bool(torch.isfinite(out).all())
        )
    else:
        print("type =", type(out).__name__)

    return out


def traced_fusion_forward(*args, **kwargs):
    print("\n>>> PHYSICS BIAS FUSION CALLED <<<")

    print("Fusion positional args:", len(args))
    for i, x in enumerate(args):
        if torch.is_tensor(x):
            print(
                f"  arg[{i}] shape={tuple(x.shape)} "
                f"norm={float(x.detach().float().norm()):.6f}"
            )
        elif isinstance(x, dict):
            print("  arg[{}] dict keys={}".format(i, list(x.keys())))
        else:
            print(f"  arg[{i}] type={type(x).__name__}")

    for k, x in kwargs.items():
        if torch.is_tensor(x):
            print(
                f"  kw[{k}] shape={tuple(x.shape)} "
                f"norm={float(x.detach().float().norm()):.6f}"
            )
        elif isinstance(x, dict):
            print(f"  kw[{k}] dict keys={list(x.keys())}")
        else:
            print(f"  kw[{k}] type={type(x).__name__}")

    out = orig_fusion_forward(*args, **kwargs)

    print(">>> FUSION OUTPUT <<<")
    if torch.is_tensor(out):
        print(
            "shape =", tuple(out.shape),
            "norm =", float(out.detach().float().norm()),
            "finite =", bool(torch.isfinite(out).all())
        )
    else:
        print("type =", type(out).__name__)

    return out


builder.forward = traced_builder_forward
fusion.forward = traced_fusion_forward

print("[TRACE] Hooks installed.")

# ------------------------------------------------------------------
# 5. Run ACTUAL model forward
# ------------------------------------------------------------------
print("\n[4] RUNNING REAL MODEL FORWARD")

t = torch.tensor([1], device=device, dtype=torch.long)

# ============================================================
# EXACT REAL EVALUATION CONDITIONING
# This is the same path used by guided_diffusion.test_util.py.
# ============================================================

look_num, struct_tensors, spectral_tensor, wavelet_tensor =     _prepare_conditions(conditions, device)

# dataset[0] returns scalar look_num; model requires [B].
if look_num.ndim == 0:
    look_num = look_num.unsqueeze(0)

print("\n[TRACE] REAL CONDITIONS")
print(
    "look_num:",
    tuple(look_num.shape),
    look_num.detach().cpu().tolist()
)

for i, st in enumerate(struct_tensors, 1):
    print(
        f"struct_s{i}:",
        tuple(st.shape),
        "norm=",
        float(st.detach().float().norm())
    )

print(
    "spectral:",
    tuple(spectral_tensor.shape),
    "norm=",
    float(spectral_tensor.detach().float().norm())
)

print(
    "wavelet:",
    tuple(wavelet_tensor.shape),
    "norm=",
    float(wavelet_tensor.detach().float().norm())
)


# ============================================================
# FIX: dataset[0] gives single-sample conditioning tensors.
# The real model expects the DataLoader-style batch dimension.
# ============================================================

if look_num.ndim == 0:
    look_num = look_num.unsqueeze(0)

if isinstance(struct_tensors, (tuple, list)):
    struct_tensors = tuple(
        x.unsqueeze(0) if torch.is_tensor(x) and x.ndim == 3 else x
        for x in struct_tensors
    )
else:
    raise RuntimeError(
        f"Expected struct_tensors tuple/list, got {type(struct_tensors)}"
    )

# The first structure scale is also passed separately as struct_tensor.
# Required shape: [B, C, H, W]
struct_tensor = struct_tensors[0]

if spectral_tensor.ndim == 3:
    spectral_tensor = spectral_tensor.unsqueeze(0)

if wavelet_tensor.ndim == 3:
    wavelet_tensor = wavelet_tensor.unsqueeze(0)

print("\\n[TRACE] BATCHED REAL CONDITIONS")
print("look_num:", tuple(look_num.shape), look_num.detach().cpu().tolist())
for i, st in enumerate(struct_tensors):
    print(
        f"struct_s{i+1}:",
        tuple(st.shape),
        "norm=",
        float(st.detach().float().norm())
    )
print(
    "spectral:",
    tuple(spectral_tensor.shape),
    "norm=",
    float(spectral_tensor.detach().float().norm())
)
print(
    "wavelet:",
    tuple(wavelet_tensor.shape),
    "norm=",
    float(wavelet_tensor.detach().float().norm())
)

# Build model kwargs AFTER all condition tensors have been batched.
model_kwargs = _build_model_kwargs(
    noisy,
    look_num,
    struct_tensors,
    spectral_tensor,
    wavelet_tensor,
)

print("\n[TRACE] MODEL KWARGS")
print("keys:", list(model_kwargs.keys()))

# ------------------------------------------------------------
# Run EXACT conditioned UNet path.
# ------------------------------------------------------------
with torch.no_grad():
    print("\nCalling model with REAL CONDITIONS...")

    output = model(
        noisy,
        t,
        **model_kwargs,
    )

print("Forward PASS")
print("Output:", tuple(output.shape))
print("Finite:", bool(torch.isfinite(output).all()))

# ------------------------------------------------------------------
# 6. Inspect physics diagnostics
# ------------------------------------------------------------------
print("\n[5] PHYSICS DIAGNOSTICS")

def show(name, value):
    print(f"\n{name}:")
    if value is None:
        print("  NONE")
        return

    if torch.is_tensor(value):
        v = value.detach().float()
        print("  shape:", tuple(v.shape))
        print("  norm :", float(v.norm()))
        print("  mean :", float(v.mean()))
        print("  min  :", float(v.min()))
        print("  max  :", float(v.max()))
        print("  finite:", bool(torch.isfinite(v).all()))
    elif isinstance(value, dict):
        for k, x in value.items():
            if torch.is_tensor(x):
                x = x.detach().float()
                print(
                    f"  {k}: shape={tuple(x.shape)} "
                    f"norm={float(x.norm()):.6f} "
                    f"mean={float(x.mean()):.6f}"
                )
            else:
                print(f"  {k}: {x}")
    else:
        print(" ", value)

for obj_name, obj in [
    ("BUILDER", builder),
    ("FUSION", fusion),
]:
    print("\n" + "-" * 60)
    print(obj_name)

    for name in [
        "last_relation_norms",
        "last_contribution_norms",
        "last_attention_shift",
        "last_bias_norm",
    ]:
        if hasattr(obj, name):
            show(name, getattr(obj, name))

# ------------------------------------------------------------------
# 7. Gate values
# ------------------------------------------------------------------
print("\n[6] LEARNED PHYSICS GATES")

for name in [
    "alpha_orientation",
    "alpha_wavelet",
    "alpha_spectral",
    "alpha_confidence",
]:
    if hasattr(fusion, name):
        value = getattr(fusion, name)
        if torch.is_tensor(value):
            print(name, "=", float(value.detach().cpu()))

# ------------------------------------------------------------------
# 8. Relation modules
# ------------------------------------------------------------------
print("\n[7] RELATION MODULES")

relation = getattr(builder, "physics_relation_matrix", None)

if relation is not None:
    print("Relation module:", type(relation).__name__)

    for name in [
        "last_orientation_relation",
        "last_wavelet_relation",
        "last_spectral_relation",
    ]:
        if hasattr(relation, name):
            show(name, getattr(relation, name))
else:
    print("PhysicsRelationMatrix not directly exposed.")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
