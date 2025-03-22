#!/bin/bash

# x_bootstrap.sh - Bootstrap script for setting up Python environment
# This script checks for Python 3.12+, creates a virtual environment,
# installs dependencies, and prepares the environment for running the process command.

# Function to check Python version
check_python_version() {
    if command -v python3 &>/dev/null; then
        python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        python_major=$(echo $python_version | cut -d. -f1)
        python_minor=$(echo $python_version | cut -d. -f2)

        if [ "$python_major" -ge 3 ] && [ "$python_minor" -ge 12 ]; then
            echo "✅ Python $python_version detected"
            return 0
        else
            echo "❌ Python $python_version detected, but version 3.12 or higher is required"
            return 1
        fi
    else
        echo "❌ Python 3 not found"
        return 1
    fi
}

# Function to print Python installation instructions
print_python_instructions() {
    echo "Please install Python 3.12 or higher:"
    echo "  • On Ubuntu/Debian: sudo apt-get install python3.12"
    echo "  • On macOS with Homebrew: brew install python@3.12"
    echo ""
    echo "After installation, make sure python3 is in your system PATH."
    echo "Then run this script again."
}


# Function to create and setup virtual environment
setup_virtual_env() {
    echo "🔄 Setting up Python virtual environment..."

    # Remove existing virtual environment if it exists
    if [ -d ".venv" ]; then
        echo "🗑️  Removing existing virtual environment"
        rm -rf .venv
    fi

    # Create new virtual environment
    python3 -m venv .venv

    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi

    echo "✅ Virtual environment created"

    # Activate virtual environment
    source .venv/bin/activate

    if [ $? -ne 0 ]; then
        echo "❌ Failed to activate virtual environment"
        exit 1
    fi

    echo "✅ Virtual environment activated"

    # Install dependencies
    echo "🔄 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi

    echo "✅ Dependencies installed successfully"
}

# Main script execution
echo "🚀 Starting Python environment setup..."

# Check Python version
if check_python_version; then
    # Setup virtual environment
    setup_virtual_env
    echo ""
    echo "✨ Python env ready."
    echo ""
    echo "📌 You can now run ./html_to_pdf_crawler.py"
else
    print_python_instructions
    exit 1
fi
