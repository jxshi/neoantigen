import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

from .features import AA_ALPHABET


_MAPPING = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}

def seq_to_tensor(seq: str) -> torch.Tensor:
    """Convert an amino-acid sequence to a tensor of indices."""
    indices = [_MAPPING[s] for s in seq if s in _MAPPING]
    return torch.tensor(indices, dtype=torch.long)


class SequenceDataset(Dataset):
    """Dataset for paired sequences.

    If the CSV contains a ``label`` column it will be returned as part of each
    item.
    """

    def __init__(self, csv_file: str, labeled: bool = True):
        self.df = pd.read_csv(csv_file)
        self.labeled = labeled and "label" in self.df.columns

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tcr = row["tcr_sequence"]
        pmhc = row["pmhc_sequence"]
        if self.labeled:
            return tcr, pmhc, float(row["label"])
        return tcr, pmhc


def collate_batch(batch):
    labeled = len(batch[0]) == 3
    if labeled:
        tcr_seqs, pmhc_seqs, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.float32)
    else:
        tcr_seqs, pmhc_seqs = zip(*batch)
        labels = None

    tcr_tensors = [seq_to_tensor(s) for s in tcr_seqs]
    pmhc_tensors = [seq_to_tensor(s) for s in pmhc_seqs]
    tcr_lengths = torch.tensor([len(t) for t in tcr_tensors])
    pmhc_lengths = torch.tensor([len(t) for t in pmhc_tensors])
    tcr_pad = pad_sequence(tcr_tensors, batch_first=True)
    pmhc_pad = pad_sequence(pmhc_tensors, batch_first=True)

    if labeled:
        return tcr_pad, tcr_lengths, pmhc_pad, pmhc_lengths, labels
    return tcr_pad, tcr_lengths, pmhc_pad, pmhc_lengths


class SequencePairClassifier(nn.Module):
    """Minimal neural model for sequence pairs."""

    def __init__(self, alphabet: str = AA_ALPHABET, embed_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(len(alphabet), embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, tcr, tcr_len, pmhc, pmhc_len):
        tcr_emb = self.embed(tcr).sum(dim=1) / tcr_len.unsqueeze(1)
        pmhc_emb = self.embed(pmhc).sum(dim=1) / pmhc_len.unsqueeze(1)
        x = torch.cat([tcr_emb, pmhc_emb], dim=1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)


def train_model(train_csv: str, model_path: str, epochs: int = 5, batch_size: int = 32, lr: float = 1e-3):
    dataset = SequenceDataset(train_csv, labeled=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    model = SequencePairClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for tcr, tcr_len, pmhc, pmhc_len, labels in loader:
            optimizer.zero_grad()
            logits = model(tcr, tcr_len, pmhc, pmhc_len)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), model_path)


def predict(predict_csv: str, model_path: str, output_csv: str, batch_size: int = 32):
    dataset = SequenceDataset(predict_csv, labeled=False)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)
    model = SequencePairClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for tcr, tcr_len, pmhc, pmhc_len in loader:
            logits = model(tcr, tcr_len, pmhc, pmhc_len)
            probs = torch.sigmoid(logits)
            preds.extend(probs.tolist())
    df = pd.read_csv(predict_csv)
    df["prediction"] = preds
    df.to_csv(output_csv, index=False)
