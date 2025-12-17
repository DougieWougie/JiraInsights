"""
Data loader for Excel files containing Jira comments.
Handles file reading, column detection, and data preprocessing.
"""

import pandas as pd
from typing import Tuple, List, Optional


def load_excel_file(file_path: str) -> pd.DataFrame:
    """
    Load an Excel file into a pandas DataFrame.

    Args:
        file_path: Path to the Excel file (.xlsx or .xls)

    Returns:
        DataFrame containing the Excel data

    Raises:
        ValueError: If file cannot be read or is empty
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl')

        if df.empty:
            raise ValueError("The Excel file is empty")

        return df
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")


def detect_comment_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect columns that likely contain comments or text data.

    Looks for:
    - Columns with names containing keywords: comment, description, notes, text, summary
    - Columns with mostly string data
    - Columns with longer text (average length > 20 characters)

    Args:
        df: DataFrame to analyze

    Returns:
        List of column names likely to contain comments
    """
    comment_keywords = ['comment', 'description', 'notes', 'text', 'summary',
                        'details', 'body', 'content', 'message']

    potential_columns = []

    for col in df.columns:
        # Check if column name contains keywords
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in comment_keywords):
            potential_columns.append(col)
            continue

        # Check if column contains mostly text data
        if df[col].dtype == 'object':
            # Calculate average text length (excluding nulls)
            non_null = df[col].dropna()
            if len(non_null) > 0:
                avg_length = non_null.astype(str).str.len().mean()
                # If average length > 20 chars, likely a comment field
                if avg_length > 20:
                    potential_columns.append(col)

    return potential_columns


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns that contain date/time information.

    Args:
        df: DataFrame to analyze

    Returns:
        List of column names containing dates
    """
    date_keywords = ['date', 'time', 'created', 'updated', 'modified', 'timestamp']
    date_columns = []

    for col in df.columns:
        # Check column name
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in date_keywords):
            date_columns.append(col)
            continue

        # Check data type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)

    return date_columns


def clean_text_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Clean a text column by removing null values and converting to string.

    Args:
        df: DataFrame containing the column
        column_name: Name of the column to clean

    Returns:
        DataFrame with cleaned text column
    """
    # Create a copy to avoid modifying original
    df = df.copy()

    # Convert to string and replace NaN with empty string
    df[column_name] = df[column_name].fillna('').astype(str)

    # Remove rows with empty strings
    df = df[df[column_name].str.strip() != '']

    return df


def prepare_data(
    file_path: str,
    comment_column: Optional[str] = None,
    date_column: Optional[str] = None
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """
    Main function to load and prepare Excel data for analysis.

    Args:
        file_path: Path to Excel file
        comment_column: Specific column to use (if None, auto-detect)
        date_column: Specific date column to use (if None, auto-detect)

    Returns:
        Tuple of (cleaned DataFrame, comment column name, date column name)

    Raises:
        ValueError: If no suitable comment column is found
    """
    # Load the Excel file
    df = load_excel_file(file_path)

    # Detect or validate comment column
    if comment_column is None:
        detected_cols = detect_comment_columns(df)
        if not detected_cols:
            raise ValueError(
                "No comment columns detected. Please specify a column manually."
            )
        comment_column = detected_cols[0]  # Use first detected column
    else:
        if comment_column not in df.columns:
            raise ValueError(f"Column '{comment_column}' not found in Excel file")

    # Clean the comment column
    df = clean_text_column(df, comment_column)

    # Detect or validate date column
    if date_column is None:
        detected_dates = detect_date_columns(df)
        date_column = detected_dates[0] if detected_dates else None
    else:
        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' not found in Excel file")

    # Convert date column if present
    if date_column:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            print(f"Warning: Could not convert {date_column} to datetime: {e}")
            date_column = None

    # Validate minimum data requirements
    if len(df) < 10:
        raise ValueError(
            f"Insufficient data: Found only {len(df)} comments. "
            "Need at least 10 comments for meaningful analysis."
        )

    return df, comment_column, date_column


def get_data_summary(df: pd.DataFrame, comment_column: str, date_column: Optional[str]) -> dict:
    """
    Generate summary statistics about the dataset.

    Args:
        df: DataFrame to summarize
        comment_column: Name of the comment column
        date_column: Name of the date column (if any)

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_comments': len(df),
        'comment_column': comment_column,
        'avg_comment_length': df[comment_column].str.len().mean(),
        'total_words': df[comment_column].str.split().str.len().sum(),
    }

    if date_column:
        summary['date_column'] = date_column
        summary['date_range_start'] = df[date_column].min()
        summary['date_range_end'] = df[date_column].max()

    return summary
