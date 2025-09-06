#!/usr/bin/env python3
"""
Data Manager Module
Handles saving and loading preprocessing results to/from JSON files
"""

import os
import json
import hashlib
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

class DataManager:
    """Manages storage and retrieval of preprocessing results"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data manager
        
        Args:
            data_dir (str): Directory to store JSON files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of data
        self.text_dir = self.data_dir / "text"
        self.image_dir = self.data_dir / "image"
        self.url_dir = self.data_dir / "url"
        
        for subdir in [self.text_dir, self.image_dir, self.url_dir]:
            subdir.mkdir(exist_ok=True)
        
        print(f"📁 Data Manager initialized with directory: {self.data_dir}")
    
    def generate_filename(self, text: str, input_type: str = "text") -> str:
        """
        Generate a filename based on the input text
        
        Args:
            text (str): Input text to generate filename from
            input_type (str): Type of input (text, image, url)
            
        Returns:
            str: Generated filename
        """
        # Clean the text for filename
        clean_text = self._clean_text_for_filename(text)
        
        # Create a hash of the text for uniqueness
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        if len(clean_text) > 50:
            clean_text = clean_text[:50]
        
        filename = f"{clean_text}_{text_hash}_{timestamp}.json"
        
        return filename
    
    def _clean_text_for_filename(self, text: str) -> str:
        """
        Clean text to make it suitable for filename
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove special characters and replace with underscores
        clean_text = re.sub(r'[^\w\s-]', '', text)
        
        # Replace spaces and multiple underscores with single underscore
        clean_text = re.sub(r'[\s_]+', '_', clean_text)
        
        # Remove leading/trailing underscores
        clean_text = clean_text.strip('_')
        
        # Ensure it's not empty
        if not clean_text:
            clean_text = "processed_data"
        
        return clean_text
    
    def save_preprocessing_result(self, result: Dict[str, Any], input_type: str = "text") -> str:
        """
        Save preprocessing result to JSON file
        
        Args:
            result (Dict[str, Any]): Preprocessing result to save
            input_type (str): Type of input (text, image, url)
            
        Returns:
            str: Path to saved file
        """
        try:
            # Extract text for filename generation
            text_for_filename = self._extract_text_for_filename(result)
            
            # Generate filename
            filename = self.generate_filename(text_for_filename, input_type)
            
            # Determine directory based on input type
            if input_type == "text":
                save_dir = self.text_dir
            elif input_type == "image":
                save_dir = self.image_dir
            elif input_type == "url":
                save_dir = self.url_dir
            else:
                save_dir = self.data_dir
            
            # Create full file path
            file_path = save_dir / filename
            
            # Add metadata to result
            result_with_metadata = result.copy()
            result_with_metadata['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'filename': filename,
                'input_type': input_type,
                'data_manager_version': '1.0.0'
            }
            
            # Save to JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_with_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved preprocessing result to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            print(f"❌ Error saving preprocessing result: {str(e)}")
            raise
    
    def _extract_text_for_filename(self, result: Dict[str, Any]) -> str:
        """
        Extract text from result for filename generation
        
        Args:
            result (Dict[str, Any]): Preprocessing result
            
        Returns:
            str: Text to use for filename
        """
        # Try different fields to get text
        text_fields = [
            'original_text',
            'cleaned_text',
            'extracted_text',
            'text',
            'content'
        ]
        
        for field in text_fields:
            if field in result and result[field]:
                text = result[field]
                if isinstance(text, str):
                    return text
                elif isinstance(text, dict) and 'text' in text:
                    return text['text']
        
        # If no text found, use a default
        return "processed_data"
    
    def load_preprocessing_result(self, file_path: str) -> Dict[str, Any]:
        """
        Load preprocessing result from JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            Dict[str, Any]: Loaded preprocessing result
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            print(f"📂 Loaded preprocessing result from: {file_path}")
            return result
            
        except Exception as e:
            print(f"❌ Error loading preprocessing result: {str(e)}")
            raise
    
    def list_saved_results(self, input_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all saved preprocessing results
        
        Args:
            input_type (Optional[str]): Filter by input type (text, image, url)
            
        Returns:
            List[Dict[str, Any]]: List of result metadata
        """
        results = []
        
        # Determine which directories to search
        if input_type:
            if input_type == "text":
                search_dirs = [self.text_dir]
            elif input_type == "image":
                search_dirs = [self.image_dir]
            elif input_type == "url":
                search_dirs = [self.url_dir]
            else:
                search_dirs = [self.data_dir]
        else:
            search_dirs = [self.text_dir, self.image_dir, self.url_dir, self.data_dir]
        
        # Search for JSON files
        for search_dir in search_dirs:
            if search_dir.exists():
                for json_file in search_dir.glob("*.json"):
                    try:
                        # Get file metadata
                        stat = json_file.stat()
                        
                        # Try to load metadata from file
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        result_info = {
                            'file_path': str(json_file),
                            'filename': json_file.name,
                            'file_size': stat.st_size,
                            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'input_type': data.get('_metadata', {}).get('input_type', 'unknown'),
                            'status': data.get('status', 'unknown'),
                            'has_features': 'features' in data.get('data', {}),
                            'feature_count': len(data.get('data', {}).get('features', {}))
                        }
                        
                        results.append(result_info)
                        
                    except Exception as e:
                        print(f"⚠️ Error reading {json_file}: {str(e)}")
        
        # Sort by creation time (newest first)
        results.sort(key=lambda x: x['created_at'], reverse=True)
        
        return results
    
    def get_result_summary(self) -> Dict[str, Any]:
        """
        Get summary of all saved results
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        all_results = self.list_saved_results()
        
        summary = {
            'total_files': len(all_results),
            'by_type': {},
            'by_status': {},
            'total_features': 0,
            'average_features_per_file': 0,
            'oldest_file': None,
            'newest_file': None
        }
        
        if all_results:
            # Count by type and status
            for result in all_results:
                input_type = result['input_type']
                status = result['status']
                
                summary['by_type'][input_type] = summary['by_type'].get(input_type, 0) + 1
                summary['by_status'][status] = summary['by_status'].get(status, 0) + 1
                
                summary['total_features'] += result['feature_count']
            
            # Calculate averages
            summary['average_features_per_file'] = summary['total_features'] / len(all_results)
            
            # Get oldest and newest files
            summary['oldest_file'] = all_results[-1]['created_at']
            summary['newest_file'] = all_results[0]['created_at']
        
        return summary
    
    def delete_result(self, file_path: str) -> bool:
        """
        Delete a preprocessing result file
        
        Args:
            file_path (str): Path to file to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️ Deleted file: {file_path}")
                return True
            else:
                print(f"⚠️ File not found: {file_path}")
                return False
        except Exception as e:
            print(f"❌ Error deleting file: {str(e)}")
            return False
    
    def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        Clean up old preprocessing result files
        
        Args:
            days_old (int): Delete files older than this many days
            
        Returns:
            int: Number of files deleted
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        all_results = self.list_saved_results()
        
        for result in all_results:
            created_at = datetime.fromisoformat(result['created_at'])
            if created_at < cutoff_date:
                if self.delete_result(result['file_path']):
                    deleted_count += 1
        
        print(f"🧹 Cleaned up {deleted_count} old files")
        return deleted_count
    
    def export_results_to_csv(self, output_file: str = None) -> str:
        """
        Export all results metadata to CSV file
        
        Args:
            output_file (str): Output CSV file path
            
        Returns:
            str: Path to exported CSV file
        """
        import csv
        
        if not output_file:
            output_file = self.data_dir / f"preprocessing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        all_results = self.list_saved_results()
        
        if not all_results:
            print("⚠️ No results to export")
            return str(output_file)
        
        # Write CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"📊 Exported {len(all_results)} results to: {output_file}")
        return str(output_file)
