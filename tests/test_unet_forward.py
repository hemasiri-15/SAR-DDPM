import torch as th

from guided_diffusion.script_util import sr_create_model

device = th.device("cuda" if th.cuda.is_available() else "cpu")

model = sr_create_model(
    large_size=64,
    small_size=64,
    in_channels=1,
    num_channels=64,
    num_res_blocks=2,
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16,8",
    num_heads=4,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0.0,
    resblock_updown=True,
    use_fp16=False,
).to(device)

model.eval()

B = 2

x = th.randn(B, 1, 64, 64, device=device)
noisy = th.randn(B, 1, 64, 64, device=device)
t = th.randint(0, 1000, (B,), device=device)

look_num = th.tensor([1, 4], device=device)
struct_tensor = th.randn(B, 3, 64, 64, device=device)

with th.no_grad():
    y = model(
        x,
        t,
        noisy=noisy,
        look_num=look_num,
        struct_tensor=struct_tensor,
    )

print("Output:", y.shape)
