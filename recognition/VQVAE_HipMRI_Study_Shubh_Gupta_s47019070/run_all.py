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