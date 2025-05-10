#!/usr/bin/env python3
"""
Initialization script for the Research Agent.
Sets up the project environment and creates necessary files and directories.
"""

import os
import sys
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 7)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version is {current_version[0]}.{current_version[1]}")
        return False
        
    return True


def create_env_file():
    """Create .env file from .env.example if it doesn't exist."""
    env_example_path = Path(".env.example")
    env_path = Path(".env")
    
    if not env_path.exists() and env_example_path.exists():
        shutil.copy(env_example_path, env_path)
        print("Created .env file from .env.example. Please fill in your API keys.")
    elif not env_example_path.exists():
        print("Warning: .env.example file not found.")


def create_data_directories():
    """Create necessary data directories."""
    directories = [
        "data",
        "logs"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"Created directory: {directory}")


def check_dependencies():
    """Check if required Python packages are installed."""
    try:
        import smolagents
        print("SmolaGents framework is installed.")
    except ImportError:
        print("Warning: SmolaGents framework is not installed.")
        print("Please run: pip install -r requirements.txt")
        return False
        
    return True


def main():
    """Main initialization function."""
    print("Initializing Research Agent...")
    
    # Check Python version
    if not check_python_version():
        return
        
    # Create necessary directories
    create_data_directories()
    
    # Create .env file
    create_env_file()
    
    # Check dependencies
    check_dependencies()
    
    print("\nInitialization complete!")
    print("To run the agent, use: python main.py 'Your research query'")
    print("For interactive mode, use: python main.py --interactive")


if __name__ == "__main__":
    main()
