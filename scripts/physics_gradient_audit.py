import torch

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
)
from guided_diffusion.test_util import (
    _prepare_conditions,
    _build_model_kwargs,
)

from scripts.datasets import SynthSARDataset


CKPT = "logs/train_2026-09-29-09-55-11/model_best.pt"
DATASET = "/dgxa_home/se24ucse043/datasets/SAR_despeckling_dataset/DSIFN/DSIFN/val"


print("=" * 80)
print("9-PATH PHYSICS + CONDITIONING GRADIENT AUDIT")
print("=" * 80)
print("Checkpoint:", CKPT)


# ============================================================
# 1. EXACT MODEL ARCHITECTURE
# ============================================================

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
model.train()

print("\n[1] MODEL")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Checkpoint load: PASS")
print("Device:", device)


# ============================================================
# 2. LOCATE MODULES
# ============================================================

builder = model.physics_attention_bias_builder
fusion = builder.physics_bias_fusion


print("\n[2b] PHYSICS ATTENTION OUTPUT SCALE AUDIT")
print("=" * 80)

physics_block_count = 0

for name, module in model.named_modules():
    if module.__class__.__name__ == "PhysicsTransformerBlock":
        physics_block_count += 1

        print(f"\nPhysicsTransformerBlock: {name}")

        attn = getattr(module, "attn", None)

        if attn is not None and hasattr(attn, "out_proj"):
            w = attn.out_proj.weight.detach().float()

            print(f"  out_proj weight norm : {w.norm().item():.12e}")
            print(f"  out_proj weight RMS  : {w.pow(2).mean().sqrt().item():.12e}")
            print(f"  out_proj weight max  : {w.abs().max().item():.12e}")

        for attr in ["gamma1", "gamma_attn", "layer_scale"]:
            if hasattr(module, attr):
                value = getattr(module, attr)

                if torch.is_tensor(value):
                    value = value.detach().float()

                    print(f"  {attr} norm           : {value.norm().item():.12e}")
                    print(f"  {attr} mean           : {value.mean().item():.12e}")
                    print(f"  {attr} min/max        : "
                          f"{value.min().item():.12e} / "
                          f"{value.max().item():.12e}")

print(f"\nPhysicsTransformerBlock count: {physics_block_count}")
print("=" * 80)


print("\n[2] MODULES")
print("Builder:", type(builder).__name__)
print("Fusion:", type(fusion).__name__)


# ============================================================
# 3. REAL DSIFN SAMPLE
# ============================================================

dataset = SynthSARDataset(
    DATASET,
    train=False,
    num_channels=3,
    crop_size=(256, 256),
    seed=42,
)

clean, noisy, filename, conditions = dataset[0]

clean = clean.unsqueeze(0).to(device)
noisy = noisy.unsqueeze(0).to(device)

look_num, struct_tensors, spectral_tensor, wavelet_tensor = \
    _prepare_conditions(conditions, device)

if look_num.ndim == 0:
    look_num = look_num.unsqueeze(0)

struct_tensors = tuple(
    x.unsqueeze(0) if x.ndim == 3 else x
    for x in struct_tensors
)

if spectral_tensor.ndim == 3:
    spectral_tensor = spectral_tensor.unsqueeze(0)

if wavelet_tensor.ndim == 3:
    wavelet_tensor = wavelet_tensor.unsqueeze(0)

print("\n[3] REAL CONDITIONS")
print("Clean:", tuple(clean.shape))
print("Noisy:", tuple(noisy.shape))
print("Look:", tuple(look_num.shape))

for i, st in enumerate(struct_tensors, 1):
    print(f"Struct_s{i}:", tuple(st.shape))

print("Spectral:", tuple(spectral_tensor.shape))
print("Wavelet:", tuple(wavelet_tensor.shape))


# ============================================================
# 4. BUILD MODEL KWARGS
# ============================================================

model_kwargs = _build_model_kwargs(
    noisy,
    look_num,
    struct_tensors,
    spectral_tensor,
    wavelet_tensor,
)


# ============================================================
# 5. RETAIN GRADIENTS ON CONDITION INPUTS
# ============================================================

struct_tensors = tuple(
    st.float().requires_grad_(True)
    for st in struct_tensors
)

spectral_tensor = spectral_tensor.float().requires_grad_(True)
wavelet_tensor = wavelet_tensor.float().requires_grad_(True)

model_kwargs = _build_model_kwargs(
    noisy,
    look_num,
    struct_tensors,
    spectral_tensor,
    wavelet_tensor,
)


# ============================================================
# 6. FORWARD
# ============================================================


# Capture gradients of the four physics relation tensors.
physics_relation_refs = {}

attention_scale_refs = {}

# Instrument the live PhysicsAwareAttention forward to measure
# QK/sqrt(d) versus the physics bias before any modification.
for module in model.modules():
    if module.__class__.__name__ == "PhysicsAwareAttention":
        original_attention_forward = module.forward

        def audited_attention_forward(
            x,
            physics_attention_bias=None,
            return_attention=False,
            _original=original_attention_forward,
            _module=module,
        ):
            if physics_attention_bias is not None:
                B, N, D = x.shape
                H = _module.num_heads
                Hd = _module.head_dim

                q = _module.q_proj(x)
                k = _module.k_proj(x)

                q = q.view(B, N, H, Hd).transpose(1, 2)
                k = k.view(B, N, H, Hd).transpose(1, 2)

                qk = torch.matmul(
                    q.float(),
                    k.transpose(-2, -1).float(),
                ) / (Hd ** 0.5)

                bias = physics_attention_bias.unsqueeze(1).float()

                attention_scale_refs["qk_norm"] = float(
                    qk.detach().norm()
                )
                attention_scale_refs["bias_norm"] = float(
                    bias.detach().norm()
                )
                attention_scale_refs["qk_mean"] = float(
                    qk.detach().mean()
                )
                attention_scale_refs["qk_std"] = float(
                    qk.detach().std()
                )
                attention_scale_refs["bias_mean"] = float(
                    bias.detach().mean()
                )
                attention_scale_refs["bias_std"] = float(
                    bias.detach().std()
                )
                attention_scale_refs["qk_max"] = float(
                    qk.detach().abs().max()
                )
                attention_scale_refs["bias_max"] = float(
                    bias.detach().abs().max()
                )

            return _original(
                x,
                physics_attention_bias=physics_attention_bias,
                return_attention=return_attention,
            )

        module.forward = audited_attention_forward
        break



fusion = model.physics_attention_bias_builder.physics_bias_fusion
_original_fusion_forward = fusion.forward

def _audited_fusion_forward(
    orientation_relation,
    wavelet_relation=None,
    spectral_relation=None,
    confidence_relation=None,
):
    relations = {
        "orientation": orientation_relation,
        "wavelet": wavelet_relation,
        "spectral": spectral_relation,
        "confidence": confidence_relation,
    }

    for name, relation in relations.items():
        if relation is not None and relation.requires_grad:
            relation.retain_grad()
            physics_relation_refs[name] = relation

    return _original_fusion_forward(
        orientation_relation,
        wavelet_relation,
        spectral_relation,
        confidence_relation,
    )

fusion.forward = _audited_fusion_forward

print("\n[4] FORWARD + BACKWARD")

t = torch.tensor([1], device=device, dtype=torch.long)

model.zero_grad(set_to_none=True)

output = model(
    noisy,
    t,
    **model_kwargs,
)

print("Output:", tuple(output.shape))
print("Output finite:", bool(torch.isfinite(output).all()))


# ============================================================
# 7. SIMPLE SCALAR LOSS
#
# Use a scalar directly connected to the model output.
# We deliberately avoid diffusion sampling here.
# ============================================================

loss = output.float().mean()

print("Loss:", float(loss.detach()))

loss.backward()

print("Backward PASS")


# ============================================================
# 8. CONDITIONING GRADIENT AUDIT
# ============================================================


print("\n" + "=" * 80)
print("ATTENTION SCALE AUDIT")
print("=" * 80)

if attention_scale_refs:
    qk_norm = attention_scale_refs["qk_norm"]
    bias_norm = attention_scale_refs["bias_norm"]

    print(f"QK/sqrt(d) norm : {qk_norm:.12e}")
    print(f"Physics bias norm: {bias_norm:.12e}")
    print(f"Bias/QK ratio    : {bias_norm / max(qk_norm, 1e-12):.12e}")
    print(f"QK mean          : {attention_scale_refs['qk_mean']:.12e}")
    print(f"QK std           : {attention_scale_refs['qk_std']:.12e}")
    print(f"Bias mean        : {attention_scale_refs['bias_mean']:.12e}")
    print(f"Bias std         : {attention_scale_refs['bias_std']:.12e}")
    print(f"QK |max|         : {attention_scale_refs['qk_max']:.12e}")
    print(f"Bias |max|       : {attention_scale_refs['bias_max']:.12e}")
else:
    print("No PhysicsAwareAttention call captured.")

print("\n" + "=" * 80)
print("CONDITIONING GRADIENT AUDIT")
print("=" * 80)

def grad_norm(x):
    if x is None:
        return 0.0
    return float(x.detach().float().norm())

conditioning_grads = {
    "Struct": grad_norm(struct_tensors[0].grad),
    "MSStruct": grad_norm(struct_tensors[1].grad),
    "Spectral": grad_norm(spectral_tensor.grad),
    "Wavelet": grad_norm(wavelet_tensor.grad),
}

# Look uses integer embedding lookup, so inspect the embedding parameters.
look_embedding_grad = model.look_emb.embedding.weight.grad
conditioning_grads["LookEmbedding"] = grad_norm(look_embedding_grad)

for name, value in conditioning_grads.items():
    print(f"{name:15s} gradient = {value:.12e}")


print("\n" + "=" * 80)
print("PHYSICS GATE GRADIENT AUDIT")
print("=" * 80)

physics_params = {
    "Orientation": fusion.alpha_orientation,
    "Wavelet": fusion.alpha_wavelet,
    "Spectral": fusion.alpha_spectral,
    "Confidence": fusion.alpha_confidence,
}

for name, param in physics_params.items():
    g = param.grad

    if g is None:
        value = 0.0
    else:
        value = float(g.detach().float().abs().item())

    print(
        f"{name:12s} "
        f"alpha={float(param.detach()):+.9f} "
        f"|grad|={value:.12e}"
    )


# ============================================================
# 10. RELATION OUTPUT GRADIENTS
#
# Capture gradients through the four relation matrices.
# ============================================================

print("\n" + "=" * 80)
print("PHYSICS RELATION GRADIENT AUDIT")
print("=" * 80)

physics_relation_grads = {}

for name in ["orientation", "wavelet", "spectral", "confidence"]:
    relation = physics_relation_refs.get(name)

    if relation is None:
        physics_relation_grads[name] = 0.0
        print(f"{name.capitalize():12s}: NOT CAPTURED")
        continue

    grad = relation.grad
    value = 0.0 if grad is None else float(
        grad.detach().float().norm()
    )

    physics_relation_grads[name] = value
    print(f"{name.capitalize():12s}: {value:.12e}")

print("\n========== SUMMARY ==========")

print("\nPhysics relations:")
for name, value in physics_relation_grads.items():
    status = "ACTIVE" if value > 1e-12 else "ZERO"
    print(f"{name.capitalize():12s}: {value:.6e} -> {status}")

print("\nConditioning:")
for name, value in conditioning_grads.items():
    status = "ACTIVE" if value > 1e-12 else "ZERO"
    print(f"{name:10s}: {value:.6e} -> {status}")

print("\nPhysics gates:")
for name, param in physics_params.items():
    g = param.grad
    value = 0.0 if g is None else float(g.detach().float().abs().item())
    status = "ACTIVE" if value > 1e-12 else "ZERO"
    print(f"{name:10s}: {value:.6e} -> {status}")

print("\n" + "=" * 80)
print("GRADIENT AUDIT COMPLETE")
print("=" * 80)
