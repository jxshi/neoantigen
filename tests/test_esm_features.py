import pytest

esm = pytest.importorskip('esm')

from pmhctcr_predictor.esm_features import ESMEmbedder


def test_pair_embedding_shape():
    embedder = ESMEmbedder()
    vec = embedder.pair_embedding("ACD", "EFG")
    assert vec.ndim == 1
    assert vec.size > 0
