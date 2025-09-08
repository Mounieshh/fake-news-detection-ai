import os
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np
import io

class ImageHandler:
    """Handles image input processing for fake news detection"""
    
    def __init__(self):
        from utils.logger import get_logger
        self.logger = get_logger(__name__)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.max_image_size = 10 * 1024 * 1024  # 10MB max
        self.ocr_available = False
        
        # Try to import pytesseract (will be installed tomorrow)
        try:
            import pytesseract
            self.ocr_available = True
            self.pytesseract = pytesseract
            # Configure tesseract path: env override, else default Windows path
            tesseract_cmd = os.environ.get('TESSERACT_CMD', r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
            if os.path.exists(tesseract_cmd):
                self.pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                self.logger.info(f"Using Tesseract at: {tesseract_cmd}")
            else:
                self.logger.warning(f"Tesseract not found at '{tesseract_cmd}'. Install it or set TESSERACT_CMD.")
        except ImportError:
            self.logger.warning("pytesseract not available. Install with: pip install pytesseract")
    
    def process_image_input(self, image_path: str) -> Dict[str, Any]:
        """
        Process image input and extract text using OCR
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict containing extracted text and image metadata
        """
        try:
            if not os.path.exists(image_path):
                return {
                    'status': 'error',
                    'error': f'Image file not found: {image_path}',
                    'input_type': 'image'
                }
            
            # Check file format
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in self.supported_formats:
                return {
                    'status': 'error',
                    'error': f'Unsupported image format: {file_ext}',
                    'input_type': 'image'
                }
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size > self.max_image_size:
                return {
                    'status': 'error',
                    'error': f'Image file too large: {file_size} bytes. Max: {self.max_image_size} bytes',
                    'input_type': 'image'
                }
            
            # Load and process image
            image = Image.open(image_path)
            image_info = self._get_image_info(image)
            
            # Extract text using OCR
            extracted_text = self._extract_text_from_image(image_path)
            
            return {
                'status': 'success',
                'input_type': 'image',
                'image_path': image_path,
                'image_info': image_info,
                'extracted_text': extracted_text,
                'ocr_available': self.ocr_available
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'input_type': 'image'
            }
    
    def process_image_data(self, image_data: bytes, filename: str = "uploaded_image") -> Dict[str, Any]:
        """
        Process image data from memory (e.g., web upload)
        
        Args:
            image_data (bytes): Raw image data
            filename (str): Name of the uploaded file
            
        Returns:
            Dict containing extracted text and image metadata
        """
        try:
            # Check file size
            if len(image_data) > self.max_image_size:
                return {
                    'status': 'error',
                    'error': f'Image data too large: {len(image_data)} bytes. Max: {self.max_image_size} bytes',
                    'input_type': 'image_data'
                }
            
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data))
            image_info = self._get_image_info(image)
            
            # Save temporarily for OCR processing
            temp_path = f"temp_{filename}"
            image.save(temp_path)
            
            # Extract text
            extracted_text = self._extract_text_from_image(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                'status': 'success',
                'input_type': 'image_data',
                'filename': filename,
                'image_info': image_info,
                'extracted_text': extracted_text,
                'ocr_available': self.ocr_available
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'input_type': 'image_data'
            }
    
    def _get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic image information"""
        return {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'width': image.width,
            'height': image.height,
            'dpi': image.info.get('dpi', 'Unknown')
        }
    
    def _extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        if not self.ocr_available:
            return {
                'status': 'ocr_not_available',
                'text': '',
                'confidence': 0,
                'message': 'Install pytesseract for OCR functionality'
            }
        
        try:
            # Use pytesseract to extract text
            extracted_text = self.pytesseract.image_to_string(image_path)
            
            # Get confidence scores if available
            try:
                data = self.pytesseract.image_to_data(image_path, output_type=self.pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            except:
                avg_confidence = 0
            
            return {
                'status': 'success',
                'text': extracted_text.strip(),
                'confidence': avg_confidence,
                'word_count': len(extracted_text.split()),
                'has_text': bool(extracted_text.strip())
            }
            
        except Exception as e:
            return {
                'status': 'ocr_error',
                'text': '',
                'confidence': 0,
                'error': str(e)
            }
    
    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """Validate image input"""
        if not os.path.exists(image_path):
            return {
                'valid': False,
                'error': 'Image file not found'
            }
        
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in self.supported_formats:
            return {
                'valid': False,
                'error': f'Unsupported format: {file_ext}'
            }
        
        file_size = os.path.getsize(image_path)
        if file_size > self.max_image_size:
            return {
                'valid': False,
                'error': f'File too large: {file_size} bytes'
            }
        
        return {
            'valid': True,
            'message': 'Image is valid'
        }
