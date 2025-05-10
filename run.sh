#!/bin/bash

# Research Agent Runner Script
# This script activates the Python virtual environment and runs the research agent

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

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Please edit .env file to add your API keys."
    else
        echo "Error: .env.example not found. Please create a .env file manually."
        exit 1
    fi
fi

# Run the research agent with any provided arguments
if [ $# -eq 0 ]; then
    # No arguments provided, run in interactive mode
    echo "Starting Research Agent in interactive mode..."
    python main.py --interactive
else
    # Pass all arguments to the research agent
    echo "Starting Research Agent with provided arguments..."
    python main.py "$@"
fi

# Deactivate the virtual environment when done
deactivate
