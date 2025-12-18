#!/usr/bin/env python3
"""Generate demo data for Foldex."""

import os
import sys
from pathlib import Path

def generate_demo_folder():
    """Generate a demo Google Drive folder structure."""
    print("Generating demo data...")
    
    # Create demo folder structure
    demo_path = Path("data/demo_folder")
    demo_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample text files
    (demo_path / "readme.txt").write_text(
        "This is a sample README file for demonstration purposes.\n"
        "Foldex can process various file types including text, PDF, and audio."
    )
    
    (demo_path / "notes.md").write_text(
        "# Project Notes\n\n"
        "## Overview\n"
        "This is a sample markdown file.\n\n"
        "## Features\n"
        "- Feature 1\n"
        "- Feature 2\n"
    )
    
    print(f"Demo folder created at: {demo_path}")
    print("You can use this folder for testing Foldex processing.")


if __name__ == "__main__":
    generate_demo_folder()

