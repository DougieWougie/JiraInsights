"""
Text processing utilities for cleaning and preparing comment text for analysis.
Handles tokenization, stopword removal, and text normalization.
"""

import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download required NLTK data (will be called once)
def download_nltk_data():
    """Download required NLTK datasets if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def get_custom_stopwords() -> Set[str]:
    """
    Get a set of custom stopwords including Jira-specific terms.

    Returns:
        Set of stopwords to filter out
    """
    # Start with English stopwords
    download_nltk_data()
    stop_words = set(stopwords.words('english'))

    # Add Jira-specific terms that don't carry meaningful theme information
    jira_terms = {
        'jira', 'ticket', 'issue', 'comment', 'updated', 'created',
        'user', 'admin', 'status', 'project', 'task', 'story',
        'epic', 'sprint', 'assignee', 'reporter', 'priority',
        'attachment', 'link', 'mentioned', 'edited'
    }

    # Add common words that don't add value
    common_filler = {
        'please', 'thanks', 'thank', 'hi', 'hello', 'regards',
        'would', 'could', 'should', 'may', 'might', 'must',
        'also', 'just', 'really', 'need', 'needs', 'want', 'wants'
    }

    return stop_words.union(jira_terms).union(common_filler)


def clean_text(text: str, remove_stopwords: bool = True, min_word_length: int = 3) -> str:
    """
    Clean and normalize text for analysis.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove Jira markup (mentions, links)
    5. Remove special characters and punctuation
    6. Tokenize
    7. Remove stopwords (optional)
    8. Lemmatize
    9. Filter by minimum word length

    Args:
        text: Input text to clean
        remove_stopwords: Whether to remove stopwords
        min_word_length: Minimum length of words to keep

    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove Jira mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove Jira markup patterns [~username], {quote}, etc.
    text = re.sub(r'\[~\w+\]', '', text)
    text = re.sub(r'\{[^}]+\}', '', text)

    # Remove numbers (often ticket IDs or counts that don't add theme value)
    text = re.sub(r'\b\d+\b', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    if remove_stopwords:
        stop_words = get_custom_stopwords()
        tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Filter by minimum length
    tokens = [word for word in tokens if len(word) >= min_word_length]

    # Join back to string
    return ' '.join(tokens)


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    """
    Extract n-grams from text.

    Args:
        text: Input text (should be cleaned first)
        n: Size of n-grams (2 for bigrams, 3 for trigrams)

    Returns:
        List of n-grams
    """
    tokens = text.split()
    ngrams = []

    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)

    return ngrams


def preprocess_for_lda(
    texts: List[str],
    remove_stopwords: bool = True,
    min_word_length: int = 3
) -> List[str]:
    """
    Preprocess a list of texts for LDA topic modeling.

    Args:
        texts: List of text documents
        remove_stopwords: Whether to remove stopwords
        min_word_length: Minimum word length to keep

    Returns:
        List of cleaned text documents
    """
    cleaned_texts = []

    for text in texts:
        cleaned = clean_text(text, remove_stopwords, min_word_length)
        if cleaned:  # Only include non-empty cleaned texts
            cleaned_texts.append(cleaned)

    return cleaned_texts


def get_word_frequencies(texts: List[str], top_n: int = 50) -> dict:
    """
    Get word frequency counts from a list of texts.

    Args:
        texts: List of cleaned text documents
        top_n: Number of top words to return

    Returns:
        Dictionary of {word: frequency} for top N words
    """
    word_freq = {}

    for text in texts:
        words = text.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and return top N
    sorted_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])

    return sorted_freq


def filter_short_texts(texts: List[str], min_words: int = 5) -> tuple:
    """
    Filter out texts that are too short for meaningful analysis.

    Args:
        texts: List of text documents
        min_words: Minimum number of words required

    Returns:
        Tuple of (filtered_texts, valid_indices)
    """
    filtered = []
    valid_indices = []

    for i, text in enumerate(texts):
        word_count = len(text.split())
        if word_count >= min_words:
            filtered.append(text)
            valid_indices.append(i)

    return filtered, valid_indices
