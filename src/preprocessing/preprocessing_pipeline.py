"""
Preprocessing Pipeline
Main pipeline that integrates text cleaning, feature extraction, and validation
"""

import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .text_cleaner import TextCleaner
from .feature_extractor import FeatureExtractor
from .data_validator import DataValidator

class PreprocessingPipeline:
    """Main preprocessing pipeline for fake news detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize preprocessing pipeline
        
        Args:
            config (dict): Configuration options for preprocessing
        """
        from utils.logger import get_logger
        self.logger = get_logger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.text_cleaner = TextCleaner()
        self.feature_extractor = FeatureExtractor()
        self.data_validator = DataValidator()
        
        self.logger.info("Preprocessing Pipeline Initialized!")
        self.logger.info(f"   - Text Cleaner: {'OK' if self.text_cleaner.nltk_available or self.text_cleaner.spacy_available else 'DEGRADED'}")
        self.logger.info(f"   - Feature Extractor: {'OK' if self.feature_extractor.nltk_available or self.feature_extractor.spacy_available else 'DEGRADED'}")
        self.logger.info("   - Data Validator: OK")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'validation': {
                'enabled': True,
                'strict_mode': False
            },
            'cleaning': {
                'remove_html': True,
                'remove_urls': True,
                'remove_emails': True,
                'remove_phone_numbers': True,
                'expand_contractions': True,
                'remove_special_chars': True,
                'normalize_whitespace': True,
                'remove_stopwords': True,
                'lemmatize': True,
                'stem': False,
                'lowercase': True,
                'remove_punctuation': False,
                'remove_numbers': False,
                'min_word_length': 2,
                'max_word_length': 50
            },
            'feature_extraction': {
                'extract_statistical': True,
                'extract_linguistic': True,
                'extract_readability': True,
                'extract_sentiment': True,
                'extract_fake_news_indicators': True,
                'extract_ngrams': True,
                'extract_pos': True,
                'extract_ner': True,
                'extract_url_features': True,
                'extract_capitalization': True,
                'extract_punctuation': True
            },
            'output': {
                'include_original_text': True,
                'include_cleaned_text': True,
                'include_tokens': True,
                'include_metadata': True,
                'include_validation_results': True
            }
        }
    
    def process_text(self, text: str, input_type: str = "text") -> Dict[str, Any]:
        """
        Process text through the complete preprocessing pipeline
        
        Args:
            text (str): Text to process
            input_type (str): Type of input (text, url, image)
            
        Returns:
            Dict containing processed results
        """
        pipeline_result = {
            'status': 'success',
            'input_type': input_type,
            'processed_at': datetime.now().isoformat(),
            'pipeline_steps': [],
            'data': {}
        }
        
        try:
            # Step 1: Validation
            if self.config['validation']['enabled']:
                self.logger.info("Step 1: Validating input...")
                validation_result = self._validate_input(text, input_type)
                pipeline_result['validation'] = validation_result
                pipeline_result['pipeline_steps'].append('validation')
                
                if not validation_result['valid'] and self.config['validation']['strict_mode']:
                    pipeline_result['status'] = 'error'
                    pipeline_result['error'] = f"Validation failed: {validation_result['errors']}"
                    return pipeline_result
            
            # Step 2: Text Cleaning
            self.logger.info("Step 2: Cleaning text...")
            cleaning_result = self._clean_text(text)
            pipeline_result['cleaning'] = cleaning_result
            pipeline_result['pipeline_steps'].append('cleaning')
            
            # Step 3: Feature Extraction
            self.logger.info("Step 3: Extracting features...")
            feature_result = self._extract_features(text, cleaning_result.get('cleaned_text', ''))
            pipeline_result['features'] = feature_result
            pipeline_result['pipeline_steps'].append('feature_extraction')
            
            # Step 4: Prepare final output
            self.logger.info("Step 4: Preparing output...")
            final_data = self._prepare_final_output(text, cleaning_result, feature_result, validation_result)
            pipeline_result['data'] = final_data
            pipeline_result['pipeline_steps'].append('output_preparation')
            
            # Add summary
            pipeline_result['summary'] = self._generate_summary(pipeline_result)
            
            self.logger.info("Preprocessing completed successfully!")
            
        except Exception as e:
            pipeline_result['status'] = 'error'
            pipeline_result['error'] = str(e)
            self.logger.exception(f"Preprocessing failed: {str(e)}")
        
        return pipeline_result
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process file through the preprocessing pipeline
        
        Args:
            file_path (str): Path to file to process
            
        Returns:
            Dict containing processed results
        """
        # First validate the file
        file_validation = self.data_validator.validate_file_input(file_path)
        
        if not file_validation['valid']:
            return {
                'status': 'error',
                'error': f"File validation failed: {file_validation['errors']}",
                'validation': file_validation
            }
        
        # Read file content
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_ext == '.csv':
                content = self._read_csv_file(file_path)
            elif file_ext == '.json':
                content = self._read_json_file(file_path)
            else:
                return {
                    'status': 'error',
                    'error': f'Unsupported file format: {file_ext}'
                }
            
            # Process the content
            return self.process_text(content, "file")
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Error reading file: {str(e)}"
            }
    
    def process_url(self, url: str) -> Dict[str, Any]:
        """
        Process URL through the preprocessing pipeline
        
        Args:
            url (str): URL to process
            
        Returns:
            Dict containing processed results
        """
        # First validate the URL
        url_validation = self.data_validator.validate_url_input(url)
        
        if not url_validation['valid']:
            return {
                'status': 'error',
                'error': f"URL validation failed: {url_validation['errors']}",
                'validation': url_validation
            }
        
        # Extract content from URL (this would use the URL handler)
        # For now, we'll assume the URL content has been extracted
        # In a real implementation, you'd integrate with the URL handler here
        
        return {
            'status': 'error',
            'error': 'URL processing not yet integrated with URL handler',
            'validation': url_validation
        }
    
    def _validate_input(self, text: str, input_type: str) -> Dict[str, Any]:
        """Validate input based on type"""
        if input_type == "text":
            return self.data_validator.validate_text_input(text)
        elif input_type == "file":
            return self.data_validator.validate_file_input(text)
        elif input_type == "url":
            return self.data_validator.validate_url_input(text)
        else:
            return {'valid': True, 'errors': [], 'warnings': [], 'metadata': {}}
    
    def _clean_text(self, text: str) -> Dict[str, Any]:
        """Clean text using the text cleaner"""
        cleaning_options = self.config['cleaning']
        return self.text_cleaner.clean_text(text, cleaning_options)
    
    def _extract_features(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """Extract features using the feature extractor"""
        feature_config = self.config['feature_extraction']
        
        # Only extract enabled features
        if not any(feature_config.values()):
            return {}
        
        all_features = self.feature_extractor.extract_all_features(original_text, cleaned_text)
        
        # Filter features based on configuration
        filtered_features = {}
        for feature_type, enabled in feature_config.items():
            if enabled:
                # Map config keys to feature categories
                feature_mapping = {
                    'extract_statistical': ['word_count', 'sentence_count', 'character_count', 'avg_word_length', 'avg_sentence_length', 'lexical_diversity'],
                    'extract_linguistic': ['words_1_char', 'words_2_chars', 'words_3_chars', 'words_4_chars', 'words_5_chars', 'words_6_plus_chars', 'avg_syllables_per_word'],
                    'extract_readability': ['flesch_reading_ease', 'flesch_kincaid_grade', 'readability_level'],
                    'extract_sentiment': ['positive_word_count', 'negative_word_count', 'sentiment_polarity', 'sentiment_subjectivity'],
                    'extract_fake_news_indicators': ['emotional_words_count', 'urgency_words_count', 'exaggeration_words_count', 'conspiracy_words_count', 'clickbait_words_count', 'fake_news_indicator_score'],
                    'extract_ngrams': ['unique_bigrams', 'unique_trigrams', 'bigram_diversity', 'trigram_diversity'],
                    'extract_pos': ['noun_count', 'verb_count', 'adjective_count', 'adverb_count', 'pronoun_count'],
                    'extract_ner': ['person_count', 'organization_count', 'location_count', 'total_entities', 'entity_density'],
                    'extract_url_features': ['url_count', 'email_count', 'has_urls', 'has_emails'],
                    'extract_capitalization': ['uppercase_ratio', 'lowercase_ratio', 'all_caps_word_count', 'title_case_word_count'],
                    'extract_punctuation': ['exclamation_count', 'question_count', 'punctuation_density', 'multiple_exclamation_count']
                }
                
                if feature_type in feature_mapping:
                    for feature_name in feature_mapping[feature_type]:
                        if feature_name in all_features:
                            filtered_features[feature_name] = all_features[feature_name]
        
        return filtered_features
    
    def _prepare_final_output(self, original_text: str, cleaning_result: Dict, 
                            feature_result: Dict, validation_result: Dict) -> Dict[str, Any]:
        """Prepare final output data"""
        output_config = self.config['output']
        final_data = {}
        
        if output_config['include_original_text']:
            final_data['original_text'] = original_text
        
        if output_config['include_cleaned_text']:
            final_data['cleaned_text'] = cleaning_result.get('cleaned_text', '')
        
        if output_config['include_tokens']:
            final_data['tokens'] = cleaning_result.get('tokens', [])
        
        if output_config['include_metadata']:
            final_data['metadata'] = {
                'cleaning_metadata': cleaning_result.get('metadata', {}),
                'feature_count': len(feature_result),
                'processing_timestamp': datetime.now().isoformat()
            }
        
        if output_config['include_validation_results']:
            final_data['validation'] = validation_result
        
        # Add features
        final_data['features'] = feature_result
        
        # Add cleaning information
        final_data['cleaning_info'] = {
            'cleaning_steps': cleaning_result.get('cleaning_steps', []),
            'word_count': cleaning_result.get('word_count', 0),
            'char_count': cleaning_result.get('char_count', 0)
        }
        
        return final_data
    
    def _generate_summary(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of preprocessing results"""
        data = pipeline_result.get('data', {})
        features = data.get('features', {})
        cleaning_info = data.get('cleaning_info', {})
        
        return {
            'total_features_extracted': len(features),
            'cleaning_steps_applied': len(cleaning_info.get('cleaning_steps', [])),
            'final_word_count': cleaning_info.get('word_count', 0),
            'final_char_count': cleaning_info.get('char_count', 0),
            'pipeline_steps_completed': len(pipeline_result.get('pipeline_steps', [])),
            'processing_successful': pipeline_result.get('status') == 'success',
            'validation_passed': pipeline_result.get('validation', {}).get('valid', False)
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and capabilities"""
        return {
            'text_cleaner_status': {
                'nltk_available': self.text_cleaner.nltk_available,
                'spacy_available': self.text_cleaner.spacy_available
            },
            'feature_extractor_status': {
                'nltk_available': self.feature_extractor.nltk_available,
                'spacy_available': self.feature_extractor.spacy_available
            },
            'data_validator_status': {
                'available': True
            },
            'configuration': self.config,
            'supported_input_types': ['text', 'file', 'url'],
            'pipeline_version': '1.0.0'
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update pipeline configuration"""
        self.config.update(new_config)
        print("🔧 Pipeline configuration updated")
    
    def reset_config(self):
        """Reset configuration to defaults"""
        self.config = self._get_default_config()
        print("🔧 Pipeline configuration reset to defaults")
    
    def batch_process(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple inputs in batch
        
        Args:
            inputs (List[Dict]): List of input dictionaries with 'type' and 'data' keys
            
        Returns:
            List of processing results
        """
        results = []
        
        self.logger.info(f"Processing {len(inputs)} inputs in batch...")
        
        for i, input_data in enumerate(inputs):
            input_type = input_data.get('type', 'text')
            data = input_data.get('data', '')
            
            self.logger.info(f"   Processing input {i+1}/{len(inputs)} ({input_type})...")
            
            if input_type == 'text':
                result = self.process_text(data, input_type)
            elif input_type == 'file':
                result = self.process_file(data)
            elif input_type == 'url':
                result = self.process_url(data)
            else:
                result = {
                    'status': 'error',
                    'error': f'Unsupported input type: {input_type}'
                }
            
            results.append(result)
        
        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def _read_csv_file(self, file_path: str) -> str:
        """Read CSV file and convert to text"""
        try:
            import csv
            content_parts = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    content_parts.extend(row)
            return ' '.join(str(part) for part in content_parts if part)
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
    
    def _read_json_file(self, file_path: str) -> str:
        """Read JSON file and convert to text"""
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON data to text
            if isinstance(data, dict):
                content_parts = []
                for key, value in data.items():
                    content_parts.append(str(key))
                    if isinstance(value, (dict, list)):
                        content_parts.append(str(value))
                    else:
                        content_parts.append(str(value))
                return ' '.join(content_parts)
            elif isinstance(data, list):
                return ' '.join(str(item) for item in data)
            else:
                return str(data)
        except Exception as e:
            raise Exception(f"Error reading JSON file: {str(e)}")
