"""
Complete VQ-VAE Pipeline Script
Author: Shubh Gupta (s47019070)
Description: Master script to run the entire VQ-VAE training and evaluation pipeline
"""

import sys
import os
import argparse
from pathlib import Path
import subprocess

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")
    
    
def print_step(step_num, total_steps, description):
    """Print step information."""
    print(f"\n{'─' * 80}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'─' * 80}\n")

def run_command(description, command, python_script=False):
    """Run a command and handle errors."""
    print(f"Running: {description}...")
    
    if python_script:
        # Use the virtual environment Python
        venv_python = "/Users/shubhgupta/Library/CloudStorage/GoogleDrive-shubhgupta2510@gmail.com/My Drive/Shubh Gupta/University/YEAR 5/SEM 2/COMP3710/PatternAnalysis-2025/.venv/bin/python"
        full_command = [venv_python] + command
    else:
        full_command = command
    
    try:
        result = subprocess.run(
            full_command,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during {description}: {e}")
        return False
