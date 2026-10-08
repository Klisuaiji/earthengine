#!/usr/bin/env python
"""CLI entry point for the full three-phase planet pipeline.

This is a thin wrapper around ``tools/planet_pipeline.py`` so the command
matches the file name.

Usage:
    python tools/generate_planet.py --seed 1234567 --width 1024 --height 512
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planet_pipeline import main

if __name__ == "__main__":
    main()
