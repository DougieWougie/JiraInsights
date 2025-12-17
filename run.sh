#!/bin/bash

# Jira Comments Theme Analysis Dashboard - Run Script
# This script starts the Streamlit dashboard

set -e  # Exit on error

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found."
    echo "Please run ./setup.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "❌ Error: Dependencies not installed."
    echo "Please run ./setup.sh first to install dependencies."
    exit 1
fi

# Start the Streamlit app
echo "🚀 Starting Jira Theme Analysis Dashboard..."
echo ""
echo "The dashboard will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
