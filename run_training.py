#!/usr/bin/env python
import subprocess
import sys

# Run training with output to file
with open('training_output.log', 'w') as f:
    result = subprocess.run(
        [sys.executable, '-m', 'backend.train_cnn', '--epochs', '30', '--batch-size', '32'],
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(f"Training completed with exit code: {result.returncode}")

# Read and print the log
with open('training_output.log', 'r') as f:
    print("=== Training Output ===")
    print(f.read())
