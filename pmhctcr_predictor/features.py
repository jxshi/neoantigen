import itertools
from collections import Counter
import numpy as np

# Standard amino acid alphabet
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

def all_kmers(k, alphabet=AA_ALPHABET):
    """Generate all possible k-mers for the given alphabet."""
    return [''.join(p) for p in itertools.product(alphabet, repeat=k)]

def kmer_vector(seq, k, kmers=None):
    """Return a vector of k-mer counts for the sequence."""
    if kmers is None:
        kmers = all_kmers(k)
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    return np.array([counts.get(kmer, 0) for kmer in kmers], dtype=float)
