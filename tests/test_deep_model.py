import pandas as pd
import torch

from pmhctcr_predictor.deep_model import (
    seq_to_tensor,
    collate_batch,
    SequenceDataset,
    SequencePairClassifier,
)


def test_seq_to_tensor():
    t = seq_to_tensor("AC")
    assert t.dtype == torch.long
    assert t.tolist() == [0, 1]


def test_deep_model_forward(tmp_path):
    df = pd.DataFrame({
        "tcr_sequence": ["AC"],
        "pmhc_sequence": ["DEF"],
        "label": [1],
    })
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    dataset = SequenceDataset(csv)
    batch = collate_batch([dataset[0]])
    tcr, tcr_len, pmhc, pmhc_len, label = batch
    model = SequencePairClassifier()
    out = model(tcr, tcr_len, pmhc, pmhc_len)
    assert out.shape == torch.Size([1])
