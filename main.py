import nltk
import random
import string
from collections import defaultdict
from nltk.util import ngrams

# Ensure necessary NLTK resources are available
nltk.download('punkt')

def load_and_preprocess_text(file_path):
    #Loads Shakespeare's text, converts to lowercase, removes punctuation, and tokenizes
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize words
    return tokens

def create_ngram_counts(tokens, n):
    #Creates n-gram to next-token frequency dictionary
    ngram_counts = defaultdict(lambda: defaultdict(int))
    
    for ngram in ngrams(tokens, n):
        prefix, next_token = tuple(ngram[:-1]), ngram[-1]
        ngram_counts[prefix][next_token] += 1
    
    return ngram_counts

def convert_counts_to_probabilities(ngram_counts):
    #Converts frequency counts to probabilities
    ngram_probs = {}
    for ngram, next_token_counts in ngram_counts.items():
        total_count = sum(next_token_counts.values())
        ngram_probs[ngram] = {word: count / total_count for word, count in next_token_counts.items()}
    return ngram_probs

def sample_next_token(ngram, ngram_probs):
    #Samples the next token based on the probability distribution
    if ngram not in ngram_probs:
        return random.choice(random.choice(list(ngram_probs.keys())))  # Pick a random token if ngram not found
    
    words, probabilities = zip(*ngram_probs[ngram].items())
    return random.choices(words, probabilities)[0]

def generate_text(initial_ngram, ngram_probs, num_words):
    #Generates text given an initial n-gram and probability dictionary
    text = list(initial_ngram)
    for _ in range(num_words - len(initial_ngram)):
        next_token = sample_next_token(tuple(text[-(len(initial_ngram)):]), ngram_probs)
        text.append(next_token)
    return ' '.join(text)

if __name__ == "__main__":
    # Load and process Shakespeare's text
    tokens = load_and_preprocess_text('shakespeare.txt')
    
    # Create and compute probabilities for bigrams, trigrams, and quadgrams
    bigram_counts = create_ngram_counts(tokens, 2)
    trigram_counts = create_ngram_counts(tokens, 3)
    quadgram_counts = create_ngram_counts(tokens, 4)
    
    bigram_probs = convert_counts_to_probabilities(bigram_counts)
    trigram_probs = convert_counts_to_probabilities(trigram_counts)
    quadgram_probs = convert_counts_to_probabilities(quadgram_counts)
    
    # Generate sample text using bigrams, trigrams, and quadgrams
    print("Generated text using bigrams:")
    print(generate_text(('to', 'be'), bigram_probs, 50))
    
    print("\nGenerated text using trigrams:")
    print(generate_text(('to', 'be', 'or'), trigram_probs, 50))
    
    print("\nGenerated text using quadgrams:")
    print(generate_text(('to', 'be', 'or', 'not'), quadgram_probs, 50))
