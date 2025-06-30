import pytest
from urllib.error import URLError

esm = pytest.importorskip('esm')

from pmhctcr_predictor.esm_features import ESMEmbedder


def test_pair_embedding_shape():
    try:
        embedder = ESMEmbedder()
    except URLError:
        pytest.skip("ESM weights unavailable")
    vec = embedder.pair_embedding("ACD", "EFG")
    assert vec.ndim == 1
    assert vec.size > 0
