"""
Data Validation Module
Validates input data and ensures quality for fake news detection
"""

import re
import os
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse
import mimetypes

class DataValidator:
    """Comprehensive data validation for fake news detection inputs"""
    
    def __init__(self):
        """Initialize data validator"""
        self.max_text_length = 100000  # 100KB of text
        self.min_text_length = 10
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.supported_text_formats = ['.txt', '.csv', '.json', '.xml', '.html']
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        self.suspicious_domains = [
            'fake', 'hoax', 'satire', 'parody', 'clickbait', 'scam', 'phishing',
            'malware', 'virus', 'spam', 'bot', 'troll'
        ]
        
        # Language detection patterns
        self.english_patterns = [
            r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
            r'\b(is|are|was|were|be|been|being)\b',
            r'\b(a|an|this|that|these|those)\b'
        ]
    
    def validate_text_input(self, text: str) -> Dict[str, Any]:
        """
        Validate text input
        
        Args:
            text (str): Text to validate
            
        Returns:
            Dict containing validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        # Check if text is provided
        if not text:
            validation_result['valid'] = False
            validation_result['errors'].append('Text input is empty')
            return validation_result
        
        # Check text length
        text_length = len(text)
        validation_result['metadata']['text_length'] = text_length
        
        if text_length < self.min_text_length:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Text too short. Minimum {self.min_text_length} characters required')
        
        if text_length > self.max_text_length:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Text too long. Maximum {self.max_text_length} characters allowed')
        
        # Check for suspicious content
        suspicious_checks = self._check_suspicious_content(text)
        validation_result['metadata'].update(suspicious_checks)
        
        if suspicious_checks.get('suspicious_score', 0) > 0.7:
            validation_result['warnings'].append('High suspicious content score detected')
        
        # Check language
        language_check = self._detect_language(text)
        validation_result['metadata']['language'] = language_check
        
        if language_check != 'english':
            validation_result['warnings'].append(f'Non-English content detected: {language_check}')
        
        # Check text quality
        quality_checks = self._check_text_quality(text)
        validation_result['metadata'].update(quality_checks)
        
        if quality_checks.get('readability_score', 0) < 0.3:
            validation_result['warnings'].append('Low readability score detected')
        
        return validation_result
    
    def validate_file_input(self, file_path: str) -> Dict[str, Any]:
        """
        Validate file input
        
        Args:
            file_path (str): Path to file to validate
            
        Returns:
            Dict containing validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        # Check if file exists
        if not os.path.exists(file_path):
            validation_result['valid'] = False
            validation_result['errors'].append(f'File not found: {file_path}')
            return validation_result
        
        # Get file information
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        validation_result['metadata'] = {
            'file_path': file_path,
            'file_size': file_size,
            'file_extension': file_ext,
            'file_name': os.path.basename(file_path)
        }
        
        # Check file size
        if file_size > self.max_file_size:
            validation_result['valid'] = False
            validation_result['errors'].append(f'File too large. Maximum {self.max_file_size} bytes allowed')
        
        if file_size == 0:
            validation_result['valid'] = False
            validation_result['errors'].append('File is empty')
        
        # Check file format
        if file_ext in self.supported_text_formats:
            validation_result['metadata']['file_type'] = 'text'
        elif file_ext in self.supported_image_formats:
            validation_result['metadata']['file_type'] = 'image'
        else:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Unsupported file format: {file_ext}')
        
        # Additional checks for text files
        if file_ext in self.supported_text_formats:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1000 characters
                    text_validation = self.validate_text_input(content)
                    validation_result['metadata']['text_validation'] = text_validation
                    
                    if not text_validation['valid']:
                        validation_result['warnings'].extend(text_validation['errors'])
            except Exception as e:
                validation_result['warnings'].append(f'Could not read file content: {str(e)}')
        
        return validation_result
    
    def validate_url_input(self, url: str) -> Dict[str, Any]:
        """
        Validate URL input
        
        Args:
            url (str): URL to validate
            
        Returns:
            Dict containing validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        # Check if URL is provided
        if not url:
            validation_result['valid'] = False
            validation_result['errors'].append('URL input is empty')
            return validation_result
        
        # Parse URL
        try:
            parsed_url = urlparse(url)
            validation_result['metadata'] = {
                'original_url': url,
                'scheme': parsed_url.scheme,
                'domain': parsed_url.netloc,
                'path': parsed_url.path,
                'query': parsed_url.query,
                'fragment': parsed_url.fragment
            }
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Invalid URL format: {str(e)}')
            return validation_result
        
        # Check URL scheme
        if parsed_url.scheme not in ['http', 'https']:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Unsupported URL scheme: {parsed_url.scheme}')
        
        # Check domain
        if not parsed_url.netloc:
            validation_result['valid'] = False
            validation_result['errors'].append('URL missing domain')
        
        # Check for suspicious domains
        domain_lower = parsed_url.netloc.lower()
        for suspicious in self.suspicious_domains:
            if suspicious in domain_lower:
                validation_result['warnings'].append(f'Suspicious domain detected: {suspicious}')
                break
        
        # Check for common fake news domains
        fake_news_domains = [
            'infowars.com', 'breitbart.com', 'naturalnews.com',
            'beforeitsnews.com', 'yournewswire.com'
        ]
        
        if domain_lower in fake_news_domains:
            validation_result['warnings'].append('Known fake news domain detected')
        
        # Check URL length
        if len(url) > 2048:
            validation_result['warnings'].append('URL is very long')
        
        return validation_result
    
    def validate_image_input(self, image_path: str) -> Dict[str, Any]:
        """
        Validate image input
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            Dict containing validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        # Check if file exists
        if not os.path.exists(image_path):
            validation_result['valid'] = False
            validation_result['errors'].append(f'Image file not found: {image_path}')
            return validation_result
        
        # Get file information
        file_size = os.path.getsize(image_path)
        file_ext = os.path.splitext(image_path)[1].lower()
        
        validation_result['metadata'] = {
            'image_path': image_path,
            'file_size': file_size,
            'file_extension': file_ext,
            'file_name': os.path.basename(image_path)
        }
        
        # Check file size
        if file_size > self.max_file_size:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Image file too large. Maximum {self.max_file_size} bytes allowed')
        
        # Check file format
        if file_ext not in self.supported_image_formats:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Unsupported image format: {file_ext}')
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type and not mime_type.startswith('image/'):
            validation_result['warnings'].append(f'File MIME type does not match image: {mime_type}')
        
        # Try to open image to verify it's valid
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                validation_result['metadata'].update({
                    'image_width': img.width,
                    'image_height': img.height,
                    'image_mode': img.mode,
                    'image_format': img.format
                })
                
                # Check image dimensions
                if img.width < 50 or img.height < 50:
                    validation_result['warnings'].append('Image dimensions are very small')
                
                if img.width > 10000 or img.height > 10000:
                    validation_result['warnings'].append('Image dimensions are very large')
                
        except Exception as e:
            validation_result['warnings'].append(f'Could not verify image: {str(e)}')
        
        return validation_result
    
    def _check_suspicious_content(self, text: str) -> Dict[str, Any]:
        """Check for suspicious content patterns"""
        text_lower = text.lower()
        
        # Suspicious patterns
        suspicious_patterns = [
            r'\b(click here|read more|find out|discover|revealed|exposed)\b',
            r'\b(you won\'t believe|shocking|amazing|incredible)\b',
            r'\b(doctors hate|this one trick|number \d+ will shock you)\b',
            r'\b(conspiracy|cover-up|secret|classified|leaked)\b',
            r'\b(urgent|breaking|immediate|critical|emergency)\b',
            r'\b(never|always|all|every|none|completely|totally)\b'
        ]
        
        suspicious_count = 0
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, text_lower)
            suspicious_count += len(matches)
        
        # Calculate suspicious score
        word_count = len(text.split())
        suspicious_score = suspicious_count / max(word_count, 1)
        
        return {
            'suspicious_pattern_count': suspicious_count,
            'suspicious_score': suspicious_score,
            'suspicious_level': self._classify_suspicious_level(suspicious_score)
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        text_lower = text.lower()
        
        # Count English patterns
        english_matches = 0
        for pattern in self.english_patterns:
            matches = re.findall(pattern, text_lower)
            english_matches += len(matches)
        
        # Simple heuristic: if we find many English patterns, it's likely English
        word_count = len(text.split())
        english_ratio = english_matches / max(word_count, 1)
        
        if english_ratio > 0.1:
            return 'english'
        elif english_ratio > 0.05:
            return 'likely_english'
        else:
            return 'non_english'
    
    def _check_text_quality(self, text: str) -> Dict[str, Any]:
        """Check text quality metrics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Basic readability metrics
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Calculate simple readability score
        readability_score = 1.0
        if avg_sentence_length > 20:
            readability_score -= 0.3
        if avg_word_length > 6:
            readability_score -= 0.2
        if len(sentences) < 3:
            readability_score -= 0.2
        
        readability_score = max(0, readability_score)
        
        return {
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'sentence_count': len(sentences),
            'readability_score': readability_score,
            'readability_level': self._classify_readability_level(readability_score)
        }
    
    def _classify_suspicious_level(self, score: float) -> str:
        """Classify suspicious content level"""
        if score >= 0.1:
            return 'high'
        elif score >= 0.05:
            return 'medium'
        elif score >= 0.02:
            return 'low'
        else:
            return 'very_low'
    
    def _classify_readability_level(self, score: float) -> str:
        """Classify readability level"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        elif score >= 0.2:
            return 'poor'
        else:
            return 'very_poor'
    
    def validate_batch_inputs(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate multiple inputs at once
        
        Args:
            inputs (List[Dict]): List of input dictionaries with 'type' and 'data' keys
            
        Returns:
            Dict containing batch validation results
        """
        batch_result = {
            'valid': True,
            'total_inputs': len(inputs),
            'valid_inputs': 0,
            'invalid_inputs': 0,
            'results': [],
            'summary': {}
        }
        
        for i, input_data in enumerate(inputs):
            input_type = input_data.get('type', 'unknown')
            data = input_data.get('data', '')
            
            # Validate based on type
            if input_type == 'text':
                result = self.validate_text_input(data)
            elif input_type == 'file':
                result = self.validate_file_input(data)
            elif input_type == 'url':
                result = self.validate_url_input(data)
            elif input_type == 'image':
                result = self.validate_image_input(data)
            else:
                result = {
                    'valid': False,
                    'errors': [f'Unknown input type: {input_type}'],
                    'warnings': [],
                    'metadata': {}
                }
            
            result['input_index'] = i
            result['input_type'] = input_type
            batch_result['results'].append(result)
            
            if result['valid']:
                batch_result['valid_inputs'] += 1
            else:
                batch_result['invalid_inputs'] += 1
                batch_result['valid'] = False
        
        # Generate summary
        batch_result['summary'] = {
            'success_rate': batch_result['valid_inputs'] / max(batch_result['total_inputs'], 1),
            'error_count': sum(len(r['errors']) for r in batch_result['results']),
            'warning_count': sum(len(r['warnings']) for r in batch_result['results'])
        }
        
        return batch_result
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get current validation rules and limits"""
        return {
            'text_limits': {
                'min_length': self.min_text_length,
                'max_length': self.max_text_length
            },
            'file_limits': {
                'max_size': self.max_file_size,
                'supported_text_formats': self.supported_text_formats,
                'supported_image_formats': self.supported_image_formats
            },
            'suspicious_domains': self.suspicious_domains,
            'validation_features': [
                'text_length_check',
                'suspicious_content_detection',
                'language_detection',
                'text_quality_assessment',
                'file_format_validation',
                'url_scheme_validation',
                'domain_reputation_check',
                'image_dimension_validation'
            ]
        }
