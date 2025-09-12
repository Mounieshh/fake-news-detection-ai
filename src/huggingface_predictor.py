#!/usr/bin/env python3
"""
HuggingFace Predictor Module
Handles loading and using the HuggingFace fake news detection model
"""

import os
from typing import Dict, Any, Optional
from transformers import pipeline

class HuggingFacePredictor:
    """Handles HuggingFace model loading and prediction for fake news detection"""
    
    def __init__(self, model_name: str = "dhruvpal/fake-news-bert"):
        """
        Initialize the HuggingFace predictor
        
        Args:
            model_name (str): Name of the HuggingFace model to use
        """
        from utils.logger import get_logger
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.pipe = None
        self.model_loaded = False
        
        self.logger.info(f"Initializing HuggingFace Predictor with model: {model_name}")
        
    def load_model(self) -> bool:
        """
        Load the HuggingFace model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Check if transformers is available
            try:
                import transformers
                self.logger.info("Transformers library found, loading HuggingFace model...")
            except ImportError:
                self.logger.error("Transformers library not installed. Please run: pip install transformers")
                self.model_loaded = False
                return False
            
            self.logger.info("Loading HuggingFace model...")
            self.pipe = pipeline("text-classification", model=self.model_name)
            self.model_loaded = True
            self.logger.info("HuggingFace model loaded successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {str(e)}")
            self.model_loaded = False
            return False
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict fake news using the loaded model
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict containing prediction results
        """
        if not self.model_loaded:
            if not self.load_model():
                return {
                    'status': 'error',
                    'error': 'Model not loaded and failed to load'
                }
        
        try:
            self.logger.info("Running fake news prediction...")
            
            # Run prediction
            prediction = self.pipe(text)
            
            # Extract results
            if isinstance(prediction, list) and len(prediction) > 0:
                result = prediction[0]
                label = result.get('label', 'UNKNOWN')
                score = result.get('score', 0.0)
            else:
                label = 'UNKNOWN'
                score = 0.0
            
            # Format the result
            prediction_result = {
                'status': 'success',
                'prediction': {
                    'label': label,
                    'score': score,
                    'confidence': score * 100  # Convert to percentage
                },
                'model_name': self.model_name,
                'text_analyzed': text[:200] + "..." if len(text) > 200 else text
            }
            
            self.logger.info(f"Prediction completed - Label: {label}, Score: {score:.4f}")
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {
                'status': 'error',
                'error': f'Prediction failed: {str(e)}'
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict containing model information
        """
        return {
            'model_name': self.model_name,
            'model_loaded': self.model_loaded,
            'model_type': 'text-classification',
            'status': 'ready' if self.model_loaded else 'not_loaded'
        }
    
    def is_ready(self) -> bool:
        """
        Check if the model is ready for prediction
        
        Returns:
            bool: True if model is loaded and ready
        """
        return self.model_loaded and self.pipe is not None
