"""
Theme analysis using Latent Dirichlet Allocation (LDA) topic modeling.
Discovers hidden topics/themes in comment text.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class ThemeAnalyzer:
    """
    Analyzer for discovering topics/themes in text using LDA.
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_features: int = 1000,
        max_df: float = 0.95,
        min_df: int = 2,
        random_state: int = 42
    ):
        """
        Initialize the theme analyzer.

        Args:
            n_topics: Number of topics to discover
            max_features: Maximum number of features for TF-IDF
            max_df: Maximum document frequency for TF-IDF (ignore terms appearing in >95% of docs)
            min_df: Minimum document frequency for TF-IDF (ignore terms appearing in <2 docs)
            random_state: Random state for reproducibility
        """
        self.n_topics = n_topics
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.random_state = random_state

        # Initialize vectorizer and LDA model
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )

        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=50,
            learning_method='online',
            n_jobs=-1  # Use all CPU cores
        )

        # Will be populated after fitting
        self.doc_term_matrix = None
        self.feature_names = None
        self.topic_distributions = None

    def fit(self, texts: List[str]) -> 'ThemeAnalyzer':
        """
        Fit the LDA model on the provided texts.

        Args:
            texts: List of cleaned text documents

        Returns:
            Self for method chaining
        """
        # Create document-term matrix using TF-IDF
        self.doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Fit LDA model
        self.lda_model.fit(self.doc_term_matrix)

        # Get topic distributions for each document
        self.topic_distributions = self.lda_model.transform(self.doc_term_matrix)

        return self

    def get_top_words_per_topic(self, n_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get the top words for each topic with their weights.

        Args:
            n_words: Number of top words to return per topic

        Returns:
            Dictionary mapping topic_id -> list of (word, weight) tuples
        """
        topics = {}

        for topic_idx, topic in enumerate(self.lda_model.components_):
            # Get indices of top words
            top_indices = topic.argsort()[-n_words:][::-1]

            # Get words and weights
            top_words = [
                (self.feature_names[i], topic[i])
                for i in top_indices
            ]

            topics[topic_idx] = top_words

        return topics

    def get_topic_labels(self, n_words: int = 3) -> Dict[int, str]:
        """
        Generate human-readable labels for each topic based on top words.

        Args:
            n_words: Number of words to include in label

        Returns:
            Dictionary mapping topic_id -> label string
        """
        labels = {}
        top_words = self.get_top_words_per_topic(n_words)

        for topic_idx, words in top_words.items():
            # Create label from top words
            word_list = [word for word, _ in words]
            label = f"Topic {topic_idx}: {', '.join(word_list)}"
            labels[topic_idx] = label

        return labels

    def assign_topics_to_documents(self, threshold: float = 0.3) -> List[int]:
        """
        Assign the dominant topic to each document.

        Args:
            threshold: Minimum probability threshold for assignment

        Returns:
            List of topic IDs (one per document)
        """
        if self.topic_distributions is None:
            raise ValueError("Model must be fitted before assigning topics")

        # Get the topic with highest probability for each document
        dominant_topics = np.argmax(self.topic_distributions, axis=1)

        # Get the maximum probability for each document
        max_probs = np.max(self.topic_distributions, axis=1)

        # Set to -1 if probability is below threshold
        dominant_topics = [
            topic if prob >= threshold else -1
            for topic, prob in zip(dominant_topics, max_probs)
        ]

        return dominant_topics

    def get_topic_distribution(self) -> pd.DataFrame:
        """
        Get the distribution of topics across all documents.

        Returns:
            DataFrame with topic counts and percentages
        """
        dominant_topics = self.assign_topics_to_documents()

        # Count documents per topic
        topic_counts = pd.Series(dominant_topics).value_counts().sort_index()

        # Create DataFrame
        labels = self.get_topic_labels(n_words=3)
        distribution = pd.DataFrame({
            'Topic': [labels.get(t, f"Unassigned") for t in topic_counts.index],
            'Count': topic_counts.values,
            'Percentage': (topic_counts.values / len(dominant_topics) * 100).round(2)
        })

        return distribution

    def get_sample_documents_per_topic(
        self,
        texts: List[str],
        n_samples: int = 5
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get sample documents for each topic.

        Args:
            texts: Original text documents
            n_samples: Number of sample documents per topic

        Returns:
            Dictionary mapping topic_id -> list of (text, probability) tuples
        """
        samples = {}

        for topic_idx in range(self.n_topics):
            # Get probabilities for this topic across all documents
            topic_probs = self.topic_distributions[:, topic_idx]

            # Get indices of top documents
            top_indices = topic_probs.argsort()[-n_samples:][::-1]

            # Get texts and probabilities
            topic_samples = [
                (texts[i], topic_probs[i])
                for i in top_indices
            ]

            samples[topic_idx] = topic_samples

        return samples

    def get_topic_coherence_scores(self) -> Dict[int, float]:
        """
        Calculate coherence scores for topics (simplified version).
        Higher scores indicate more coherent topics.

        Returns:
            Dictionary mapping topic_id -> coherence score
        """
        coherence_scores = {}
        top_words_per_topic = self.get_top_words_per_topic(n_words=10)

        for topic_idx, words in top_words_per_topic.items():
            # Simple coherence: average weight of top words
            avg_weight = np.mean([weight for _, weight in words])
            coherence_scores[topic_idx] = float(avg_weight)

        return coherence_scores

    def get_topic_words_for_wordcloud(self, topic_idx: int, n_words: int = 50) -> Dict[str, float]:
        """
        Get word frequencies for a specific topic suitable for word cloud generation.

        Args:
            topic_idx: Index of the topic
            n_words: Number of words to return

        Returns:
            Dictionary of {word: weight} for word cloud
        """
        if topic_idx >= self.n_topics:
            raise ValueError(f"Topic index {topic_idx} out of range (max: {self.n_topics - 1})")

        topic = self.lda_model.components_[topic_idx]
        top_indices = topic.argsort()[-n_words:][::-1]

        word_weights = {
            self.feature_names[i]: float(topic[i])
            for i in top_indices
        }

        return word_weights

    def get_all_words_for_wordcloud(self, n_words: int = 100) -> Dict[str, float]:
        """
        Get overall word frequencies across all topics for word cloud.

        Args:
            n_words: Number of words to return

        Returns:
            Dictionary of {word: weight}
        """
        # Sum weights across all topics
        overall_weights = self.lda_model.components_.sum(axis=0)
        top_indices = overall_weights.argsort()[-n_words:][::-1]

        word_weights = {
            self.feature_names[i]: float(overall_weights[i])
            for i in top_indices
        }

        return word_weights

    def get_results_dataframe(
        self,
        original_df: pd.DataFrame,
        texts: List[str],
        comment_column: str
    ) -> pd.DataFrame:
        """
        Create a DataFrame with original data plus topic assignments.

        Args:
            original_df: Original DataFrame
            texts: List of text documents (same order as original_df)
            comment_column: Name of the comment column

        Returns:
            DataFrame with added topic columns
        """
        result_df = original_df.copy()

        # Assign topics
        dominant_topics = self.assign_topics_to_documents()
        max_probs = np.max(self.topic_distributions, axis=1)

        # Get topic labels
        labels = self.get_topic_labels(n_words=3)

        # Add columns
        result_df['topic_id'] = dominant_topics
        result_df['topic_label'] = [labels.get(t, "Unassigned") for t in dominant_topics]
        result_df['topic_probability'] = max_probs

        return result_df
