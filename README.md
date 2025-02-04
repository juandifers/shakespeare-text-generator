# shakespeare-text-generator

## Overview
This project is a Shakespearean text generator that uses **N-grams (bigrams, trigrams, and quadgrams)** to generate text in the style of William Shakespeare. The model learns from a corpus of Shakespeare's works and generates text by predicting the next token based on previous tokens.

## Features
- **Preprocesses Shakespearean text** by tokenizing and removing punctuation.
- **Builds n-gram frequency distributions** (bigram, trigram, and quadgram models).
- **Calculates probability distributions** to determine the likelihood of the next token.
- **Generates text** based on learned probabilities using weighted random sampling.
- **Includes unit tests** implemented using `pytest` to validate the correctness of the functions.

## Installation
Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt

