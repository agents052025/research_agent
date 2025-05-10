#!/bin/bash

# Research Agent Runner Script
# This script activates the Python virtual environment and runs the research agent

# Directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating a new one..."
    
    # Function to check Python version
    get_python_version() {
        local cmd=$1
        if command -v $cmd &> /dev/null; then
            local version=$($cmd -c 'import sys; print("{}.{}.{}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))')
            echo $version
        else
            echo "0.0.0"
        fi
    }
    
    # Function to compare version strings
    version_gt() {
        test "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" != "$1";
    }
    
    # Find the best Python version
    best_cmd=""
    best_version="0.0.0"
    target_version="3.13.0"
    min_version="3.9.0"
    
    # Check specific Python versions
    for ver in 3.13 3.12 3.11 3.10 3.9; do
        cmd="python$ver"
        version=$(get_python_version $cmd)
        if [ "$version" != "0.0.0" ]; then
            if version_gt "$version" "$best_version" && version_gt "$target_version" "$version"; then
                best_cmd=$cmd
                best_version=$version
            fi
        fi
    done
    
    # Check generic python3 command
    version=$(get_python_version "python3")
    if [ "$version" != "0.0.0" ]; then
        if version_gt "$version" "$best_version" && version_gt "$target_version" "$version"; then
            best_cmd="python3"
            best_version=$version
        fi
    fi
    
    # Check generic python command
    version=$(get_python_version "python")
    if [ "$version" != "0.0.0" ]; then
        if version_gt "$version" "$best_version" && version_gt "$target_version" "$version"; then
            best_cmd="python"
            best_version=$version
        fi
    fi
    
    # Use the best version found
    if [ "$best_cmd" != "" ] && version_gt "$best_version" "$min_version"; then
        echo "Using $best_cmd (version $best_version)..."
        $best_cmd -m venv venv
    else
        echo "Error: No suitable Python version found. Please install Python 3.9 or newer."
        exit 1
    fi
    
    # Create data directory if it doesn't exist
    mkdir -p data
    
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
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
