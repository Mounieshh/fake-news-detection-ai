#!/usr/bin/env python3
"""
Hybrid Predictor Module
Combines BERT predictions with feature-based predictions for enhanced fake news detection
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

from huggingface_predictor import HuggingFacePredictor
from preprocessing.preprocessing_pipeline import PreprocessingPipeline

class HybridPredictor:
    """Combines BERT and feature-based predictions for enhanced fake news detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the hybrid predictor
        
        Args:
            model_path (str, optional): Path to saved feature-based model
        """
        from utils.logger import get_logger
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.bert_predictor = HuggingFacePredictor()
        self.preprocessing_pipeline = PreprocessingPipeline()
        
        # Initialize or load feature-based model
        self.feature_model = self._load_or_create_feature_model(model_path)
        
        self.logger.info("Hybrid Predictor initialized!")
        
    def _load_or_create_feature_model(self, model_path: Optional[str]) -> RandomForestClassifier:
        """Load existing model or create a new one"""
        if model_path and os.path.exists(model_path):
            try:
                self.logger.info(f"Loading feature-based model from {model_path}")
                return joblib.load(model_path)
            except Exception as e:
                self.logger.error(f"Failed to load model: {str(e)}")
        
        self.logger.info("Creating new feature-based model")
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _extract_numerical_features(self, preprocessed_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from preprocessed data"""
        features = []
        
        # Get features from preprocessed data
        stats = preprocessed_data.get('statistics', {})
        features.extend([
            stats.get('word_count', 0),
            stats.get('sentence_count', 0),
            stats.get('average_word_length', 0),
            stats.get('average_sentence_length', 0)
        ])
        
        # Add sentiment scores
        sentiment = preprocessed_data.get('sentiment', {})
        features.extend([
            sentiment.get('positive', 0),
            sentiment.get('negative', 0),
            sentiment.get('neutral', 0),
            sentiment.get('compound', 0)
        ])
        
        # Add readability scores
        readability = preprocessed_data.get('readability', {})
        features.extend([
            readability.get('flesch_reading_ease', 0),
            readability.get('flesch_kincaid_grade', 0)
        ])
        
        return features
    
    def predict(self, text: str, weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Make predictions using both BERT and feature-based models
        
        Args:
            text (str): Text to analyze
            weights (dict, optional): Custom weights for prediction combination
                                    {'bert': float, 'features': float}
        
        Returns:
            Dict containing combined prediction results
        """
        if weights is None:
            weights = {'bert': 0.7, 'features': 0.3}
            
        try:
            # 1. Get BERT prediction
            bert_result = self.bert_predictor.predict(text)
            if bert_result.get('status') != 'success':
                return bert_result  # Return error if BERT fails
            
            bert_score = bert_result['prediction']['score']
            
            # 2. Get preprocessed features and feature-based prediction
            preprocessed = self.preprocessing_pipeline.process_text(text)
            if preprocessed.get('status') != 'success':
                return {
                    'status': 'error',
                    'error': 'Preprocessing failed'
                }
            
            # Extract numerical features
            features = self._extract_numerical_features(preprocessed)
            
            # Get feature-based prediction
            try:
                feature_score = self.feature_model.predict_proba([features])[0][1]
            except Exception as e:
                self.logger.warning(f"Feature-based prediction failed: {str(e)}")
                feature_score = 0.5  # Neutral score if prediction fails
            
            # 3. Combine predictions
            combined_score = (
                bert_score * weights['bert'] +
                feature_score * weights['features']
            )
            
            # 4. Prepare final result
            return {
                'status': 'success',
                'prediction': {
                    'label': 'FAKE' if combined_score > 0.5 else 'REAL',
                    'score': combined_score,
                    'confidence': combined_score * 100,
                    'bert_score': bert_score,
                    'feature_score': feature_score
                },
                'features': preprocessed.get('features', {}),
                'bert_details': bert_result.get('prediction', {}),
                'weights_used': weights
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid prediction failed: {str(e)}")
            return {
                'status': 'error',
                'error': f'Prediction failed: {str(e)}'
            }
    
    def train_feature_model(self, texts: List[str], labels: List[int], save_path: Optional[str] = None):
        """
        Train the feature-based model
        
        Args:
            texts (List[str]): List of training texts
            labels (List[int]): List of labels (0 for real, 1 for fake)
            save_path (str, optional): Path to save the trained model
        """
        try:
            # Process all texts and extract features
            features = []
            for text in texts:
                preprocessed = self.preprocessing_pipeline.process_text(text)
                if preprocessed.get('status') == 'success':
                    features.append(self._extract_numerical_features(preprocessed))
            
            # Train the model
            self.feature_model.fit(features, labels)
            
            # Save the model if path provided
            if save_path:
                joblib.dump(self.feature_model, save_path)
                self.logger.info(f"Feature-based model saved to {save_path}")
            
            self.logger.info("Feature-based model training completed!")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
