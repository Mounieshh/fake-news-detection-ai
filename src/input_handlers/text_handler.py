import os
import re
import json
import csv
from typing import Dict, Any, Optional

class TextHandler:
    """Handles text input processing for fake news detection"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.csv', '.json']
        self.max_text_length = 10000  # Maximum characters to process
        
    def process_text_input(self, text: str) -> Dict[str, Any]:
        """
        Process direct text input
        
        Args:
            text (str): Raw text input
            
        Returns:
            Dict containing processed text and metadata
        """
        try:
            # Basic text cleaning
            cleaned_text = self._clean_text(text)
            
            # Extract basic features
            features = self._extract_basic_features(cleaned_text)
            
            return {
                'status': 'success',
                'input_type': 'direct_text',
                'original_text': text,
                'cleaned_text': cleaned_text,
                'features': features,
                'length': len(cleaned_text),
                'word_count': len(cleaned_text.split())
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'input_type': 'direct_text'
            }
    
    def process_file_input(self, file_path: str) -> Dict[str, Any]:
        """
        Process text file input
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            Dict containing processed text and metadata
        """
        try:
            if not os.path.exists(file_path):
                return {
                    'status': 'error',
                    'error': f'File not found: {file_path}',
                    'input_type': 'file'
                }
            
            # Check file format
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                return {
                    'status': 'error',
                    'error': f'Unsupported file format: {file_ext}',
                    'input_type': 'file'
                }
            
            # Read file content
            if file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_ext == '.csv':
                content = self._read_csv_file(file_path)
            elif file_ext == '.json':
                content = self._read_json_file(file_path)
            
            # Process the content
            return self.process_text_input(content)
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'input_type': 'file'
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Truncate if too long
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "..."
        
        return text
    
    def _extract_basic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text features"""
        if not text:
            return {}
        
        sentences = text.split('.')
        words = text.split()
        
        return {
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': len(words) / max(len([s for s in sentences if s.strip()]), 1),
            'has_numbers': any(char.isdigit() for char in text),
            'has_urls': 'http' in text.lower() or 'www' in text.lower(),
            'has_emails': '@' in text and '.' in text,
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1)
        }
    
    def validate_input(self, text: str) -> Dict[str, Any]:
        """Validate text input"""
        if not text or not text.strip():
            return {
                'valid': False,
                'error': 'Text input is empty'
            }
        
        if len(text) > self.max_text_length:
            return {
                'valid': False,
                'error': f'Text too long. Maximum {self.max_text_length} characters allowed.'
            }
        
        return {
            'valid': True,
            'message': 'Text input is valid'
        }
    
    def _read_csv_file(self, file_path: str) -> str:
        """Read CSV file and convert to text"""
        try:
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
