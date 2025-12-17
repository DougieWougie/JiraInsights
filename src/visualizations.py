"""
Visualization utilities for creating charts and plots.
Includes word clouds, bar charts, time series, and interactive plots.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional
import io


def create_wordcloud(
    word_weights: Dict[str, float],
    title: str = "Word Cloud",
    width: int = 800,
    height: int = 400,
    colormap: str = 'viridis'
) -> plt.Figure:
    """
    Create a word cloud from word frequencies.

    Args:
        word_weights: Dictionary of {word: weight}
        title: Title for the plot
        width: Width of the word cloud
        height: Height of the word cloud
        colormap: Matplotlib colormap name

    Returns:
        Matplotlib figure
    """
    if not word_weights:
        # Create empty figure with message
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.text(0.5, 0.5, 'No words to display', ha='center', va='center', fontsize=20)
        ax.axis('off')
        return fig

    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        colormap=colormap,
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_weights)

    # Create figure
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)

    return fig


def create_topic_distribution_chart(distribution_df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive bar chart of topic distribution.

    Args:
        distribution_df: DataFrame with columns: Topic, Count, Percentage

    Returns:
        Plotly figure
    """
    fig = px.bar(
        distribution_df,
        x='Topic',
        y='Count',
        title='Topic Distribution',
        labels={'Count': 'Number of Comments'},
        color='Percentage',
        color_continuous_scale='viridis',
        hover_data={'Percentage': ':.2f'}
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
        xaxis_title="Topic",
        yaxis_title="Number of Comments"
    )

    return fig


def create_top_keywords_chart(word_freq: Dict[str, int], top_n: int = 20) -> go.Figure:
    """
    Create a bar chart of top keywords.

    Args:
        word_freq: Dictionary of {word: frequency}
        top_n: Number of top keywords to display

    Returns:
        Plotly figure
    """
    # Get top N keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, frequencies = zip(*sorted_words) if sorted_words else ([], [])

    fig = go.Figure([go.Bar(
        x=list(frequencies),
        y=list(words),
        orientation='h',
        marker=dict(
            color=list(frequencies),
            colorscale='blues',
            showscale=False
        )
    )])

    fig.update_layout(
        title=f'Top {top_n} Keywords',
        xaxis_title='Frequency',
        yaxis_title='Keyword',
        height=max(400, top_n * 20),
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def create_topic_trends_chart(
    df: pd.DataFrame,
    date_column: str,
    period: str = 'W'
) -> go.Figure:
    """
    Create a time series chart showing topic trends over time.

    Args:
        df: DataFrame with topic_id and date column
        date_column: Name of the date column
        period: Resampling period ('D'=daily, 'W'=weekly, 'M'=monthly)

    Returns:
        Plotly figure
    """
    # Group by date and topic
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy[date_column])
    df_copy = df_copy.set_index('date')

    # Get unique topics
    topics = sorted([t for t in df_copy['topic_id'].unique() if t != -1])

    fig = go.Figure()

    period_names = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
    period_name = period_names.get(period, period)

    for topic in topics:
        # Filter by topic and resample
        topic_data = df_copy[df_copy['topic_id'] == topic]
        counts = topic_data.resample(period).size()

        topic_label = topic_data['topic_label'].iloc[0] if len(topic_data) > 0 else f"Topic {topic}"

        fig.add_trace(go.Scatter(
            x=counts.index,
            y=counts.values,
            mode='lines+markers',
            name=topic_label,
            line=dict(width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=f'{period_name} Topic Trends Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Comments',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        )
    )

    return fig


def create_topic_heatmap(
    df: pd.DataFrame,
    date_column: str,
    period: str = 'W'
) -> go.Figure:
    """
    Create a heatmap showing topic intensity over time.

    Args:
        df: DataFrame with topic_id and date column
        date_column: Name of the date column
        period: Resampling period ('D'=daily, 'W'=weekly, 'M'=monthly)

    Returns:
        Plotly figure
    """
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy[date_column])
    df_copy = df_copy.set_index('date')

    # Get unique topics
    topics = sorted([t for t in df_copy['topic_id'].unique() if t != -1])

    # Create pivot table
    pivot_data = []
    date_index = []

    for topic in topics:
        topic_data = df_copy[df_copy['topic_id'] == topic]
        counts = topic_data.resample(period).size()
        pivot_data.append(counts)
        if len(date_index) == 0:
            date_index = counts.index

    # Combine into matrix
    if pivot_data:
        matrix = np.column_stack([series.reindex(date_index, fill_value=0).values for series in pivot_data])
        topic_labels = [
            df_copy[df_copy['topic_id'] == t]['topic_label'].iloc[0]
            if len(df_copy[df_copy['topic_id'] == t]) > 0
            else f"Topic {t}"
            for t in topics
        ]
    else:
        matrix = np.array([])
        topic_labels = []

    fig = go.Figure(data=go.Heatmap(
        z=matrix.T if len(matrix) > 0 else [],
        x=[d.strftime('%Y-%m-%d') for d in date_index] if len(date_index) > 0 else [],
        y=topic_labels,
        colorscale='YlOrRd',
        hoverongaps=False
    ))

    period_names = {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}
    period_name = period_names.get(period, period)

    fig.update_layout(
        title=f'{period_name} Topic Heatmap',
        xaxis_title='Date',
        yaxis_title='Topic',
        height=max(400, len(topics) * 40)
    )

    return fig


def create_topic_scatter_plot(
    topic_distributions: np.ndarray,
    topic_labels: List[str],
    texts: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a 2D scatter plot of documents in topic space using dimensionality reduction.

    Args:
        topic_distributions: Array of topic distributions (n_docs x n_topics)
        topic_labels: List of topic labels for each document
        texts: Optional list of text snippets for hover

    Returns:
        Plotly figure
    """
    from sklearn.decomposition import PCA

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(topic_distributions)

    # Create hover text
    if texts:
        hover_text = [f"{label}<br>{text[:100]}..." for label, text in zip(topic_labels, texts)]
    else:
        hover_text = topic_labels

    fig = px.scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        color=topic_labels,
        title='Document Distribution in Topic Space',
        labels={'x': 'First Component', 'y': 'Second Component', 'color': 'Topic'},
        hover_data={'hover': hover_text}
    )

    fig.update_traces(marker=dict(size=8, opacity=0.6))
    fig.update_layout(height=600)

    return fig


def create_topic_pie_chart(distribution_df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart of topic distribution.

    Args:
        distribution_df: DataFrame with columns: Topic, Count, Percentage

    Returns:
        Plotly figure
    """
    fig = px.pie(
        distribution_df,
        values='Count',
        names='Topic',
        title='Topic Distribution',
        hover_data=['Percentage']
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)

    return fig


def create_topic_comparison_chart(
    topic_words: Dict[int, List[tuple]],
    topics_to_compare: List[int],
    n_words: int = 10
) -> go.Figure:
    """
    Create a grouped bar chart comparing top words across topics.

    Args:
        topic_words: Dictionary mapping topic_id -> list of (word, weight) tuples
        topics_to_compare: List of topic IDs to compare
        n_words: Number of top words to show per topic

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for topic_id in topics_to_compare:
        if topic_id in topic_words:
            words, weights = zip(*topic_words[topic_id][:n_words])
            fig.add_trace(go.Bar(
                name=f'Topic {topic_id}',
                x=list(words),
                y=list(weights)
            ))

    fig.update_layout(
        title='Topic Comparison - Top Words',
        xaxis_title='Words',
        yaxis_title='Weight',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )

    return fig


def create_coherence_chart(coherence_scores: Dict[int, float]) -> go.Figure:
    """
    Create a bar chart of topic coherence scores.

    Args:
        coherence_scores: Dictionary mapping topic_id -> coherence score

    Returns:
        Plotly figure
    """
    topics = list(coherence_scores.keys())
    scores = list(coherence_scores.values())

    fig = go.Figure([go.Bar(
        x=[f'Topic {t}' for t in topics],
        y=scores,
        marker=dict(
            color=scores,
            colorscale='rdylgn',
            showscale=False
        )
    )])

    fig.update_layout(
        title='Topic Coherence Scores',
        xaxis_title='Topic',
        yaxis_title='Coherence Score',
        height=400,
        xaxis_tickangle=-45
    )

    return fig


def fig_to_bytes(fig) -> bytes:
    """
    Convert a matplotlib figure to bytes for download.

    Args:
        fig: Matplotlib figure

    Returns:
        Bytes representation of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()
