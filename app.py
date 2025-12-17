"""
Jira Comments Theme Analysis Dashboard
Interactive Streamlit application for analyzing Jira comments using LDA topic modeling.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import prepare_data, get_data_summary, detect_comment_columns, detect_date_columns
from text_processor import preprocess_for_lda, get_word_frequencies, filter_short_texts
from theme_analyzer import ThemeAnalyzer
from visualizations import (
    create_wordcloud, create_topic_distribution_chart, create_top_keywords_chart,
    create_topic_trends_chart, create_topic_heatmap, create_topic_scatter_plot,
    create_topic_pie_chart, create_coherence_chart, fig_to_bytes
)


# Page configuration
st.set_page_config(
    page_title="Jira Theme Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(file_path, comment_col, date_col):
    """Load and prepare data with caching."""
    df, comment_column, date_column = prepare_data(file_path, comment_col, date_col)
    summary = get_data_summary(df, comment_column, date_column)
    return df, comment_column, date_column, summary


@st.cache_data
def analyze_themes(texts, n_topics, max_features, min_df):
    """Perform theme analysis with caching."""
    analyzer = ThemeAnalyzer(
        n_topics=n_topics,
        max_features=max_features,
        min_df=min_df
    )
    analyzer.fit(texts)
    return analyzer


def main():
    # Header
    st.markdown('<div class="main-header">📊 Jira Comments Theme Analysis</div>', unsafe_allow_html=True)
    st.markdown("Discover hidden themes in your Jira comments using AI-powered topic modeling")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        help="Upload your Jira export Excel file"
    )

    if uploaded_file is None:
        st.info("👈 Please upload an Excel file to begin analysis")
        st.markdown("""
        ### How to use this dashboard:

        1. **Upload** your Jira export Excel file (must contain a comments column)
        2. **Select** the column containing comments (auto-detected by default)
        3. **Configure** analysis parameters in the sidebar
        4. **Click** 'Run Analysis' to discover themes
        5. **Explore** the results in different tabs

        ### What you'll get:

        - **Word Clouds**: Visual representation of common terms
        - **Topic Distribution**: See how comments group into themes
        - **Trends Analysis**: Track how themes evolve over time
        - **Interactive Reports**: Explore and export your findings
        """)
        return

    # Save uploaded file temporarily
    temp_file = Path(f"/tmp/{uploaded_file.name}")
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load data
    try:
        with st.spinner("Loading data..."):
            # First, just load to get column options
            import pandas as pd
            preview_df = pd.read_excel(temp_file, engine='openpyxl')

            # Detect columns
            comment_cols = detect_comment_columns(preview_df)
            date_cols = detect_date_columns(preview_df)

            # Column selection
            st.sidebar.subheader("📋 Column Selection")

            all_cols = list(preview_df.columns)
            default_comment = comment_cols[0] if comment_cols else all_cols[0]

            selected_comment_col = st.sidebar.selectbox(
                "Comment Column",
                options=all_cols,
                index=all_cols.index(default_comment),
                help="Select the column containing comment text"
            )

            selected_date_col = None
            if date_cols:
                date_options = ["None"] + date_cols
                selected_date = st.sidebar.selectbox(
                    "Date Column (Optional)",
                    options=date_options,
                    help="Select a date column for trend analysis"
                )
                selected_date_col = selected_date if selected_date != "None" else None

            # Load with selected columns
            df, comment_column, date_column, summary = load_and_process_data(
                str(temp_file),
                selected_comment_col,
                selected_date_col
            )

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return

    # Analysis parameters
    st.sidebar.subheader("🔧 Analysis Parameters")

    n_topics = st.sidebar.slider(
        "Number of Topics",
        min_value=3,
        max_value=20,
        value=10,
        help="How many themes to discover"
    )

    max_features = st.sidebar.slider(
        "Max Features",
        min_value=500,
        max_value=3000,
        value=1000,
        step=100,
        help="Maximum number of words to consider"
    )

    min_df = st.sidebar.slider(
        "Min Document Frequency",
        min_value=1,
        max_value=10,
        value=2,
        help="Minimum number of documents a word must appear in"
    )

    # Run analysis button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Display data summary
    st.sidebar.subheader("📊 Data Summary")
    st.sidebar.metric("Total Comments", summary['total_comments'])
    st.sidebar.metric("Avg Comment Length", f"{summary['avg_comment_length']:.0f} chars")

    if date_column:
        st.sidebar.metric("Date Range", f"{summary['date_range_start'].date()} to {summary['date_range_end'].date()}")

    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'processed_texts' not in st.session_state:
        st.session_state.processed_texts = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None

    # Run analysis
    if run_analysis:
        with st.spinner("🔄 Analyzing themes... This may take a minute."):
            try:
                # Preprocess text
                progress_bar = st.progress(0)
                st.write("Step 1/3: Cleaning text...")
                raw_texts = df[comment_column].tolist()
                cleaned_texts = preprocess_for_lda(raw_texts)

                progress_bar.progress(33)
                st.write("Step 2/3: Filtering short texts...")
                filtered_texts, valid_indices = filter_short_texts(cleaned_texts, min_words=5)

                if len(filtered_texts) < 10:
                    st.error(f"Not enough valid comments after cleaning. Found {len(filtered_texts)}, need at least 10.")
                    return

                progress_bar.progress(66)
                st.write("Step 3/3: Discovering themes...")

                # Analyze themes
                analyzer = analyze_themes(filtered_texts, n_topics, max_features, min_df)

                # Filter original df to match valid texts
                filtered_df = df.iloc[valid_indices].reset_index(drop=True)

                # Get results
                results_df = analyzer.get_results_dataframe(
                    filtered_df,
                    filtered_texts,
                    comment_column
                )

                # Store in session state
                st.session_state.analyzer = analyzer
                st.session_state.processed_texts = filtered_texts
                st.session_state.results_df = results_df

                progress_bar.progress(100)
                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Display results
    if st.session_state.analyzer is not None:
        analyzer = st.session_state.analyzer
        processed_texts = st.session_state.processed_texts
        results_df = st.session_state.results_df

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview",
            "🎯 Topics",
            "📅 Trends",
            "🔍 Explore",
            "💡 Insights"
        ])

        # Tab 1: Overview
        with tab1:
            st.header("Overview")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Overall Word Cloud")
                word_weights = analyzer.get_all_words_for_wordcloud(n_words=100)
                wc_fig = create_wordcloud(word_weights, title="Most Common Terms")
                st.pyplot(wc_fig)

                # Download button
                wc_bytes = fig_to_bytes(wc_fig)
                st.download_button(
                    "📥 Download Word Cloud",
                    data=wc_bytes,
                    file_name="overall_wordcloud.png",
                    mime="image/png"
                )

            with col2:
                st.subheader("Top Keywords")
                word_freq = get_word_frequencies(processed_texts, top_n=50)
                keywords_chart = create_top_keywords_chart(word_freq, top_n=15)
                st.plotly_chart(keywords_chart, use_container_width=True)

        # Tab 2: Topics
        with tab2:
            st.header("Discovered Topics")

            # Topic distribution
            distribution = analyzer.get_topic_distribution()
            dist_chart = create_topic_distribution_chart(distribution)
            st.plotly_chart(dist_chart, use_container_width=True)

            # Pie chart
            col1, col2 = st.columns(2)
            with col1:
                pie_chart = create_topic_pie_chart(distribution)
                st.plotly_chart(pie_chart, use_container_width=True)

            with col2:
                coherence_scores = analyzer.get_topic_coherence_scores()
                coherence_chart = create_coherence_chart(coherence_scores)
                st.plotly_chart(coherence_chart, use_container_width=True)

            # Topic details
            st.subheader("Topic Details")

            top_words = analyzer.get_top_words_per_topic(n_words=15)
            samples = analyzer.get_sample_documents_per_topic(
                df[comment_column].iloc[analyzer.topic_distributions.shape[0]].tolist()[:len(processed_texts)],
                n_samples=3
            )

            for topic_id in range(n_topics):
                with st.expander(f"📌 Topic {topic_id}: {', '.join([w for w, _ in top_words[topic_id][:3]])}"):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.write("**Top Words:**")
                        words_df = pd.DataFrame(
                            top_words[topic_id][:10],
                            columns=['Word', 'Weight']
                        )
                        words_df['Weight'] = words_df['Weight'].round(3)
                        st.dataframe(words_df, hide_index=True, use_container_width=True)

                    with col2:
                        # Topic word cloud
                        topic_words = analyzer.get_topic_words_for_wordcloud(topic_id, n_words=50)
                        topic_wc = create_wordcloud(
                            topic_words,
                            title=f"Topic {topic_id}",
                            width=400,
                            height=300,
                            colormap='plasma'
                        )
                        st.pyplot(topic_wc)

                    st.write("**Sample Comments:**")
                    if topic_id in samples:
                        for i, (text, prob) in enumerate(samples[topic_id], 1):
                            st.text(f"{i}. [{prob:.2%}] {text[:200]}...")

        # Tab 3: Trends
        with tab3:
            st.header("Topic Trends Over Time")

            if date_column:
                period = st.selectbox(
                    "Time Period",
                    options=['D', 'W', 'M'],
                    format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x],
                    index=1
                )

                # Time series
                trends_chart = create_topic_trends_chart(results_df, date_column, period)
                st.plotly_chart(trends_chart, use_container_width=True)

                # Heatmap
                heatmap = create_topic_heatmap(results_df, date_column, period)
                st.plotly_chart(heatmap, use_container_width=True)

            else:
                st.info("📅 Date column not available. Upload data with dates to see trends.")

        # Tab 4: Explore
        with tab4:
            st.header("Explore Data")

            # Scatter plot
            st.subheader("Document Distribution in Topic Space")
            topic_labels = results_df['topic_label'].tolist()
            scatter = create_topic_scatter_plot(
                analyzer.topic_distributions,
                topic_labels,
                processed_texts
            )
            st.plotly_chart(scatter, use_container_width=True)

            # Filterable data table
            st.subheader("Comments with Topic Assignments")

            topic_filter = st.multiselect(
                "Filter by Topic",
                options=sorted(results_df['topic_label'].unique()),
                default=None
            )

            display_df = results_df.copy()
            if topic_filter:
                display_df = display_df[display_df['topic_label'].isin(topic_filter)]

            # Select columns to display
            display_cols = [comment_column, 'topic_label', 'topic_probability']
            if date_column and date_column in display_df.columns:
                display_cols.insert(0, date_column)

            st.dataframe(
                display_df[display_cols],
                use_container_width=True,
                height=400
            )

            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Results (CSV)",
                data=csv,
                file_name="jira_theme_analysis_results.csv",
                mime="text/csv"
            )

        # Tab 5: Insights
        with tab5:
            st.header("Key Insights")

            # Summary statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Comments Analyzed", len(results_df))

            with col2:
                most_common = distribution.iloc[0]['Topic']
                most_common_pct = distribution.iloc[0]['Percentage']
                st.metric("Most Common Topic", f"{most_common_pct:.1f}%", most_common)

            with col3:
                avg_coherence = sum(coherence_scores.values()) / len(coherence_scores)
                st.metric("Avg Topic Coherence", f"{avg_coherence:.3f}")

            # Insights
            st.subheader("📊 Analysis Summary")

            st.write(f"""
            **Dataset Overview:**
            - Analyzed **{len(results_df)}** comments across **{n_topics}** discovered topics
            - Average comment length: **{summary['avg_comment_length']:.0f}** characters
            - Total unique words: **{len(analyzer.feature_names)}**

            **Topic Distribution:**
            - Most prevalent topic: **{distribution.iloc[0]['Topic']}** ({distribution.iloc[0]['Percentage']:.1f}%)
            - Most coherent topic: **Topic {max(coherence_scores, key=coherence_scores.get)}**
            """)

            if date_column:
                date_span = (summary['date_range_end'] - summary['date_range_start']).days
                st.write(f"- Time span: **{date_span}** days ({summary['date_range_start'].date()} to {summary['date_range_end'].date()})")

            st.subheader("💡 Recommendations")
            st.write("""
            1. **Review high-coherence topics** - These represent well-defined themes worth investigating
            2. **Monitor trend changes** - Look for topics that spike suddenly, indicating emerging issues
            3. **Focus on dominant topics** - Address the most common themes to impact the majority of comments
            4. **Investigate low-probability assignments** - Comments that don't fit well may require manual review
            """)


if __name__ == "__main__":
    main()
