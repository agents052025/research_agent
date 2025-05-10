#!/bin/bash

# Research Agent Test Runner Script
# This script activates the Python virtual environment and runs the unit tests

# Directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one with Python 3.13..."
    python3.13 -m venv venv
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
    echo "Setup complete."
else
    echo "Using existing virtual environment."
fi

# Activate virtual environment
source venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version)
echo "Using $PYTHON_VERSION"

# Create test data directory if it doesn't exist
mkdir -p tests/data

# Install test dependencies if needed
pip install pytest pytest-cov

# Run the tests
if [ "$1" == "--coverage" ]; then
    # Run tests with coverage report
    python -m pytest tests/ -v --cov=agent --cov=core --cov=tools --cov-report=term-missing
elif [ -n "$1" ]; then
    # Run specific test file or directory
    python -m pytest "$1" -v
else
    # Run all tests
    python -m pytest tests/ -v
fi

# Deactivate the virtual environment when done
deactivate
