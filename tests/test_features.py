from pmhctcr_predictor.features import kmer_vector, all_kmers

def test_kmer_vector_length():
    seq = "ACDE"
    kmers = all_kmers(2)
    vec = kmer_vector(seq, 2, kmers)
    assert len(vec) == len(kmers)
