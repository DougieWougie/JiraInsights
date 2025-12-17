# Jira Comments Theme Analysis Dashboard

An interactive web application that analyzes Jira comment data from Excel files using AI-powered topic modeling (LDA - Latent Dirichlet Allocation) to discover hidden themes and trends.

## Features

- **Automated Theme Discovery**: Uses LDA to automatically identify topics/themes in your comments
- **Interactive Visualizations**:
  - Word clouds (overall and per-topic)
  - Topic distribution charts
  - Time-based trend analysis
  - Interactive scatter plots
  - Topic heatmaps
- **Flexible Data Handling**: Auto-detects comment and date columns
- **Export Capabilities**: Download visualizations and analyzed data
- **Real-time Analysis**: Streamlit-based dashboard with instant feedback

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup (Recommended)

Run the automated setup script:

```bash
./setup.sh
```

This script will:
- Check Python version (3.8+ required)
- Create a virtual environment
- Install all dependencies
- Download required NLTK data

### Manual Setup

If you prefer manual setup:

1. **Clone or navigate to the project directory**:
   ```bash
   cd /home/dougie/Projects/claude/insights
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**:
   ```python
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
   ```

## Usage

### Running the Dashboard

**Option 1: Using the run script (Recommended)**

```bash
./run.sh
```

**Option 2: Manual start**

```bash
source venv/bin/activate  # Activate virtual environment
streamlit run app.py
```

The dashboard will open in your default web browser (typically at `http://localhost:8501`).

### Using the Dashboard

1. **Upload Excel File**:
   - Click "Browse files" in the sidebar
   - Select your Jira export Excel file (.xlsx or .xls)

2. **Configure Settings**:
   - **Comment Column**: Select the column containing comment text (auto-detected by default)
   - **Date Column**: Optionally select a date column for trend analysis
   - **Number of Topics**: Adjust how many themes to discover (3-20, default: 10)
   - **Max Features**: Control vocabulary size (500-3000, default: 1000)
   - **Min Document Frequency**: Filter rare terms (1-10, default: 2)

3. **Run Analysis**:
   - Click "🚀 Run Analysis" button
   - Wait for processing to complete (typically 30-60 seconds)

4. **Explore Results**:
   - **Overview Tab**: See overall word cloud and top keywords
   - **Topics Tab**: Explore discovered topics with word clouds and sample comments
   - **Trends Tab**: View how topics evolve over time (requires date column)
   - **Explore Tab**: Interactive scatter plot and filterable data table
   - **Insights Tab**: Key findings and recommendations

### Excel File Requirements

Your Excel file should contain:
- **Required**: At least one column with text data (comments, descriptions, notes, etc.)
- **Optional**: A date/timestamp column for trend analysis
- **Minimum**: 10 comments for meaningful analysis (more is better)

Example column names that are auto-detected:
- Comment columns: "Comment", "Description", "Notes", "Text", "Summary", "Details"
- Date columns: "Date", "Created", "Updated", "Modified", "Timestamp"

## Project Structure

```
insights/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── src/
│   ├── data_loader.py     # Excel file reading and preprocessing
│   ├── text_processor.py  # Text cleaning and preprocessing
│   ├── theme_analyzer.py  # LDA topic modeling
│   └── visualizations.py  # Chart and plot generation
└── README.md             # This file
```

## How It Works

### 1. Data Loading
- Reads Excel files using pandas and openpyxl
- Auto-detects columns containing comments and dates
- Validates data quality and minimum requirements

### 2. Text Processing
- Cleans text: removes URLs, emails, special characters
- Removes stopwords (common English words + Jira-specific terms)
- Tokenizes and lemmatizes text
- Filters short or empty comments

### 3. Topic Modeling (LDA)
- Converts text to TF-IDF vectors
- Applies Latent Dirichlet Allocation to discover topics
- Each topic is characterized by a set of related words
- Each comment is assigned to one or more topics with probabilities

### 4. Visualization
- Generates interactive charts using Plotly
- Creates word clouds with matplotlib
- Provides downloadable PNG and CSV exports

## Customization

### Adjusting Topic Quality

If topics aren't meaningful:
- **Increase number of topics**: More granular themes
- **Decrease number of topics**: Broader themes
- **Adjust min_df**: Higher values filter more rare terms
- **Increase max_features**: Consider more vocabulary

### Adding Custom Stopwords

Edit `src/text_processor.py` and add words to the `jira_terms` or `common_filler` sets in the `get_custom_stopwords()` function.

## Troubleshooting

### "No comment columns detected"
- Manually select the comment column from the dropdown
- Ensure the column contains text data (not numbers or dates)

### "Insufficient data: Found only X comments"
- You need at least 10 valid comments
- Check that your comment column isn't mostly empty
- Try a different Excel file or sheet

### Topics don't make sense
- Increase the number of comments (100+ recommended)
- Adjust the number of topics
- Check that comments contain meaningful text (not just IDs or short phrases)

### Analysis is slow
- Reduce max_features (e.g., to 500)
- Reduce number of topics
- Use a smaller dataset for testing

## Technical Details

### Libraries Used
- **streamlit**: Web dashboard framework
- **pandas**: Data manipulation
- **scikit-learn**: LDA implementation and TF-IDF
- **nltk**: Natural language processing
- **plotly**: Interactive visualizations
- **wordcloud**: Word cloud generation
- **matplotlib**: Static plot generation

### Algorithm: Latent Dirichlet Allocation (LDA)
LDA is a probabilistic model that:
- Assumes each document is a mixture of topics
- Assumes each topic is a mixture of words
- Infers the topic structure from the data
- Widely used for text mining and information retrieval

## Future Enhancements

Potential improvements:
- CSV file support
- Multi-language support
- Sentiment analysis integration
- Topic comparison between time periods
- PDF report export
- API for programmatic access

## License

This project is provided as-is for analyzing Jira comment data.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your Excel file meets the requirements
4. Check Python version compatibility (3.8+)

## Contributing

To contribute improvements:
1. Test thoroughly with various Excel files
2. Ensure code follows existing patterns
3. Update documentation as needed
4. Add error handling for edge cases
