import pytest
from main import load_and_preprocess_text, create_ngram_counts, convert_counts_to_probabilities, sample_next_token, generate_text

@pytest.fixture
def test_tokens():
    return ['to', 'be', 'or', 'not', 'to', 'be', 'that', 'is', 'the', 'question']

@pytest.fixture
def bigram_counts(test_tokens):
    return create_ngram_counts(test_tokens, 2)

@pytest.fixture
def trigram_counts(test_tokens):
    return create_ngram_counts(test_tokens, 3)

@pytest.fixture
def bigram_probs(bigram_counts):
    return convert_counts_to_probabilities(bigram_counts)

@pytest.fixture
def trigram_probs(trigram_counts):
    return convert_counts_to_probabilities(trigram_counts)

def test_load_and_preprocess_text():
    processed_tokens = load_and_preprocess_text('test_shakespeare.txt')
    assert isinstance(processed_tokens, list)
    assert len(processed_tokens) > 0

def test_create_ngram_counts(bigram_counts):
    assert ('to', 'be') in bigram_counts
    assert bigram_counts[('to', 'be')]['or'] == 1
    assert bigram_counts[('be', 'or')]['not'] == 1

def test_convert_counts_to_probabilities(bigram_probs, trigram_probs):
    assert pytest.approx(sum(bigram_probs[('to', 'be')].values())) == 1.0
    assert pytest.approx(sum(trigram_probs[('to', 'be', 'or')].values())) == 1.0

def test_sample_next_token(bigram_probs):
    token = sample_next_token(('to', 'be'), bigram_probs)
    assert token in bigram_probs[('to', 'be')]

def test_generate_text(bigram_probs):
    generated_text = generate_text(('to', 'be'), bigram_probs, 10)
    assert isinstance(generated_text, str)
    assert len(generated_text.split()) > 5
