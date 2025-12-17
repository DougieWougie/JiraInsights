# Jira Comments Theme Analysis Dashboard

## Overview
Build an interactive web dashboard using Streamlit that analyzes Jira comment data from Excel files, performs LDA topic modeling to identify themes, and provides multiple visualization types.

## Architecture

### Technology Stack
- **Web Framework**: Streamlit (simple, interactive, perfect for data apps)
- **Data Processing**: pandas, openpyxl
- **Text Analysis**: scikit-learn (TfidfVectorizer, LatentDirichletAllocation), nltk (stopwords, tokenization)
- **Visualizations**:
  - plotly (interactive charts)
  - wordcloud (word clouds)
  - matplotlib/seaborn (static charts as needed)

### Application Structure
```
insights/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── src/
│   ├── data_loader.py     # Excel file reading and preprocessing
│   ├── text_processor.py  # Text cleaning and preprocessing
│   ├── theme_analyzer.py  # LDA topic modeling
│   └── visualizations.py  # All visualization functions
└── README.md             # Setup and usage instructions
```

## Implementation Steps

### 1. Project Setup
- Create directory structure
- Create requirements.txt with dependencies:
  - streamlit
  - pandas
  - openpyxl
  - scikit-learn
  - nltk
  - plotly
  - wordcloud
  - matplotlib

### 2. Data Loader (src/data_loader.py)
- Function to read Excel files (support .xlsx, .xls)
- Auto-detect potential comment columns (look for columns with text data, keywords like "comment", "description", "notes")
- Allow user to select specific column
- Extract date columns for trend analysis
- Handle missing/null values
- Return cleaned DataFrame

### 3. Text Processor (src/text_processor.py)
- Text cleaning function:
  - Remove URLs, emails, special characters
  - Convert to lowercase
  - Remove stopwords (English + custom Jira terms like "jira", "ticket")
  - Tokenization and lemmatization
- N-gram extraction (unigrams, bigrams, trigrams)
- Prepare text for LDA (create document-term matrix)

### 4. Theme Analyzer (src/theme_analyzer.py)
- LDA topic modeling implementation:
  - Configurable number of topics (default 5-10)
  - Use TF-IDF vectorization before LDA
  - Extract top words per topic
  - Assign topics to each comment
  - Calculate topic distributions
- Generate theme labels (automatic naming based on top words)
- Topic coherence scoring

### 5. Visualizations (src/visualizations.py)
**Word Clouds**:
- Overall word cloud from all comments
- Per-topic word clouds
- Customizable colors and sizes

**Bar Charts**:
- Top themes by frequency (document count per topic)
- Top keywords overall
- Topic distribution percentages
- Interactive plotly bar charts with hover details

**Trend Analysis**:
- Time series of topic prevalence (if date column available)
- Line charts showing theme evolution
- Heatmap of topics over time periods (weekly/monthly)

**Interactive Reports**:
- Plotly interactive scatter plots (comments in topic space)
- Topic distribution pie charts
- Data tables with filtering capabilities
- Export functionality for charts (PNG, HTML)

### 6. Main Dashboard (app.py)
**Layout**:
```
Header: "Jira Comments Theme Analysis"

Sidebar:
- File uploader (Excel)
- Column selector (auto-detected + manual)
- Parameters:
  - Number of topics (slider: 3-20)
  - Min word frequency threshold
  - Date column selector (for trends)
- "Run Analysis" button

Main Area:
Tabs:
1. Overview Tab:
   - Dataset summary (# comments, date range)
   - Overall word cloud
   - Top keywords bar chart

2. Topics Tab:
   - Topic distribution bar chart
   - Topic details (expandable sections)
   - Per-topic word clouds
   - Sample comments per topic

3. Trends Tab:
   - Time series plots (if date available)
   - Topic evolution heatmap
   - Insights and patterns

4. Explore Tab:
   - Interactive scatter plot (dimensionality reduction)
   - Searchable data table
   - Filter by topic
   - Export options

5. Insights Tab:
   - Key findings summary
   - Topic coherence scores
   - Recommendations
```

**Features**:
- Real-time analysis progress indicator
- Caching for performance (@st.cache_data)
- Download buttons for visualizations
- Export analyzed data with topic assignments

### 7. Documentation (README.md)
- Installation instructions
- Usage guide
- Example screenshots
- Troubleshooting section

## Key Implementation Details

### LDA Configuration
- Use TF-IDF vectorization (better than raw counts)
- Set `max_df=0.95, min_df=2` to filter common/rare terms
- Default 10 topics, allow 3-20 range
- Use `n_jobs=-1` for parallel processing

### Text Preprocessing
- Remove Jira-specific artifacts (mentions, markup)
- Custom stopwords: common Jira terms that don't add meaning
- Keep domain-specific terms that are meaningful
- Minimum word length: 3 characters

### Performance Optimization
- Streamlit caching for data loading and model training
- Lazy loading of visualizations (only generate when tab is viewed)
- Sample data for preview (if dataset is very large)

### Error Handling
- Graceful handling of malformed Excel files
- Validation that selected column contains text data
- Minimum data requirements (at least 10 comments)
- Clear error messages for users

## Critical Files
- `app.py` - Main entry point
- `src/theme_analyzer.py` - Core LDA implementation
- `src/visualizations.py` - All visualization functions

## Future Enhancements (Post-MVP)
- Support for CSV files
- Multiple language support
- Sentiment analysis integration
- Comparison between different time periods
- Export full report as PDF
