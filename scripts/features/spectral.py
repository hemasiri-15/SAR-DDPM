import numpy as np
import torch

from structdiff.utils.spectral_tensor_features import (
    compute_spectral_features_multiscale,
)


class SpectralCondition:
    """
    A11 tensor-eigenvalue spectral conditioning.

    Converts the three multi-scale structure tensors:
        s1, s2, s3: [3, H, W]
    into:
        spectral_tensor: [12, H, W]

    Each scale contributes:
        lambda1
        lambda2
        anisotropy
        coherence
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def compute(
        self,
        clean_array: np.ndarray,
        struct_tensors=None,
    ) -> dict:
        if struct_tensors is None:
            raise ValueError(
                "SpectralCondition requires struct_tensors "
                "(s1, s2, s3) to generate the 12-channel "
                "tensor-spectral representation."
            )

        if len(struct_tensors) != 3:
            raise ValueError(
                f"Expected 3 structure tensors, got {len(struct_tensors)}."
            )

        # StructureCondition returns torch tensors [3,H,W].
        # Convert them to NumPy for the pure-NumPy spectral feature utility.
        structure_numpy = []

        for i, tensor in enumerate(struct_tensors):
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()

            tensor = np.asarray(tensor, dtype=np.float32)

            if tensor.ndim != 3 or tensor.shape[0] != 3:
                raise ValueError(
                    f"struct_tensors[{i}] must have shape [3,H,W], "
                    f"got {tensor.shape}."
                )

            structure_numpy.append(tensor)

        s1, s2, s3 = structure_numpy

        spectral_features = compute_spectral_features_multiscale(
            s1,
            s2,
            s3,
            eps=self.eps,
        )

        # [12,H,W]
        spectral_tensor = torch.from_numpy(
            spectral_features.astype(np.float32)
        )

        return {
            "spectral_tensor": spectral_tensor,
        }
