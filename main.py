#!/usr/bin/env python3
"""
Fake News Detection System - Main Entry Point
This is the main entry point for the fake news detection system.
"""

import sys
import os
from src.utils.logger import get_logger

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logger = get_logger(__name__)

try:
    from src.main import main as run_fake_news_detector
    logger.info("Starting Fake News Detection System..")
    run_fake_news_detector()
    
except ImportError as e:
    logger.error("Import Error: Make sure you're running from the correct directory")
    print(f"Error: {e}")
    print("\n💡 Try running: python src/main.py")
    
except Exception as e:
    print(f"Unexpected error: {e}")
    print("\n🔧 Make sure all dependencies are installed:")
    print("   pip install -r requirements.txt")