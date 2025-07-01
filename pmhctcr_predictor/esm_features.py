"""Utilities for embedding sequences using pretrained ESM models.

The functions in this module require both ``torch`` and ``fair-esm`` to be
installed. These heavy dependencies are optional and only needed when running
the ESM-based model.
"""

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    import esm
except ImportError:  # pragma: no cover - optional dependency
    esm = None


class ESMEmbedder:
    """Embed sequences using a pretrained ESM model."""

    def __init__(self, model_name: str = "esm2_t6_8M_UR50D"):
        if esm is None:
            raise ImportError("esm package is required for ESM features")
        if torch is None:
            raise ImportError("torch is required for ESM features")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

    def embed(self, seq: str) -> np.ndarray:
        """Return the average embedding of a sequence."""
        if torch is None:
            raise ImportError("torch is required for ESM features")
        _, _, tokens = self.batch_converter([("seq", seq)])
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        rep = out["representations"][self.model.num_layers].squeeze(0)
        return rep.mean(dim=0).cpu().numpy()

    def pair_embedding(self, seq1: str, seq2: str) -> np.ndarray:
        if torch is None:
            raise ImportError("torch is required for ESM features")
        emb1 = self.embed(seq1)
        emb2 = self.embed(seq2)
        return np.concatenate([emb1, emb2])
