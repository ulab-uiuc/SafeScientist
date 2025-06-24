#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

# Check if required packages are installed
if ! python3 -c "import openai" &> /dev/null; then
    echo "Installing required packages..."
    pip install openai
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable is not set"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY=your-key-here"
    exit 1
fi

# Run the evaluation
echo "Starting safety evaluation experiment..."
python3 safety_evaluation.py

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "✅ Experiment completed successfully"
else
    echo "❌ Experiment failed"
    exit 1
fi 