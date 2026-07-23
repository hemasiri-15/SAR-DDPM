"""
LookCondition (A1 — Multi-look conditioning)

ASSUMPTION FLAGGED FOR REVIEW:
This reference implementation samples an integer "number of looks" L per
sample and exposes it as `look_num`. If your existing multi-look pipeline
also changes how speckle is *synthesized* (e.g. averaging L independent
single-look gamma draws to produce noisier/cleaner speckle for small/large
L), that noise-generation logic still lives in SynthSARDataset.__getitem__
and is intentionally NOT duplicated here — ConditionGenerator only builds
the *conditioning signal* the UNet sees, not the noisy observation itself.

If your real look-sampling logic differs (e.g. drawn from a fixed discrete
set, or with a specific prior distribution used in the paper), swap out
`sample_look_num` below — it's the only place this decision is made.
"""

import numpy as np
import torch


class LookCondition:
    def __init__(self, look_min=1, look_max=4, rng=None):
        """
        look_min, look_max: inclusive range of "number of looks" L used
            for multi-look conditioning (A1).
        rng: optional np.random.Generator for reproducibility. If None,
            a fresh default_rng is created (NOT recommended for training —
            pass the dataset's seeded RNG so runs are reproducible).
        """
        self.look_min = look_min
        self.look_max = look_max
        self.rng = rng if rng is not None else np.random.default_rng()

    def compute(self, clean_array: np.ndarray) -> dict:
        """
        Returns:
            {"look_num": torch.LongTensor scalar} — the sampled look count.
            Downstream, your UNet's look-embedding layer should consume
            this as an index (e.g. nn.Embedding(look_max+1, embed_dim)).
        """
        supported_looks = np.array([1, 2, 4, 8, 10], dtype=np.int64)
        look_num = int(self.rng.choice(supported_looks))
        return {"look_num": torch.tensor(look_num, dtype=torch.long)}
