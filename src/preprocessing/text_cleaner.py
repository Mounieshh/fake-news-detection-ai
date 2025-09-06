"""
Text Cleaning and Normalization Module
Uses NLTK and spaCy for advanced text preprocessing
"""

import re
import string
from typing import Dict, Any, List, Optional
import unicodedata

class TextCleaner:
    """Advanced text cleaning and normalization using NLTK and spaCy"""
    
    def __init__(self):
        """Initialize text cleaner with NLTK and spaCy"""
        self.nltk_available = False
        self.spacy_available = False
        self.spacy_model = None
        
        # Common stopwords (fallback if NLTK not available)
        self.fallback_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'this',
            'these', 'those', 'have', 'had', 'has', 'do', 'does', 'did',
            'can', 'could', 'would', 'should', 'may', 'might', 'must'
        }
        
        # Initialize NLTK
        self._init_nltk()
        
        # Initialize spaCy
        self._init_spacy()
        
        # Common contractions mapping
        self.contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
            "can't": "cannot", "couldn't": "could not", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
            "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have",
            "they've": "they have", "i'll": "i will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will", "i'd": "i would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would",
            "they'd": "they would"
        }
    
    def _init_nltk(self):
        """Initialize NLTK components"""
        try:
            import nltk
            self.nltk_available = True
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)
            
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                print("Downloading NLTK POS tagger...")
                nltk.download('averaged_perceptron_tagger', quiet=True)
            
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                print("Downloading NLTK punkt_tab...")
                nltk.download('punkt_tab', quiet=True)
            
            # Import NLTK components
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk.corpus import stopwords
            from nltk.tag import pos_tag
            from nltk.stem import WordNetLemmatizer, PorterStemmer
            
            self.word_tokenize = word_tokenize
            self.sent_tokenize = sent_tokenize
            self.pos_tag = pos_tag
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            
            print("✅ NLTK initialized successfully")
            
        except ImportError:
            print("⚠️ NLTK not available. Install with: pip install nltk")
            # Set fallback methods
            self.word_tokenize = None
            self.sent_tokenize = None
            self.pos_tag = None
            self.stopwords = self.fallback_stopwords
            self.lemmatizer = None
            self.stemmer = None
        except Exception as e:
            print(f"⚠️ NLTK initialization error: {e}")
            # Set fallback methods
            self.word_tokenize = None
            self.sent_tokenize = None
            self.pos_tag = None
            self.stopwords = self.fallback_stopwords
            self.lemmatizer = None
            self.stemmer = None
    
    def _init_spacy(self):
        """Initialize spaCy components"""
        try:
            import spacy
            self.spacy_available = True
            
            # Try to load English model
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
                print("✅ spaCy English model loaded successfully")
            except OSError:
                print("⚠️ spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                print("   Using basic spaCy without model...")
                self.spacy_model = spacy.blank("en")
            
        except ImportError:
            print("⚠️ spaCy not available. Install with: pip install spacy")
        except Exception as e:
            print(f"⚠️ spaCy initialization error: {e}")
    
    def clean_text(self, text: str, options: Dict[str, bool] = None) -> Dict[str, Any]:
        """
        Comprehensive text cleaning and normalization
        
        Args:
            text (str): Raw text to clean
            options (dict): Cleaning options
            
        Returns:
            Dict containing cleaned text and metadata
        """
        if not text or not text.strip():
            return {
                'cleaned_text': '',
                'original_text': text,
                'cleaning_steps': [],
                'metadata': {}
            }
        
        # Default cleaning options
        default_options = {
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
        }
        
        if options:
            default_options.update(options)
        
        original_text = text
        cleaned_text = text
        cleaning_steps = []
        
        # Step 1: Basic cleaning
        if default_options['remove_html']:
            cleaned_text = self._remove_html_tags(cleaned_text)
            cleaning_steps.append('html_removal')
        
        if default_options['remove_urls']:
            cleaned_text = self._remove_urls(cleaned_text)
            cleaning_steps.append('url_removal')
        
        if default_options['remove_emails']:
            cleaned_text = self._remove_emails(cleaned_text)
            cleaning_steps.append('email_removal')
        
        if default_options['remove_phone_numbers']:
            cleaned_text = self._remove_phone_numbers(cleaned_text)
            cleaning_steps.append('phone_removal')
        
        # Step 2: Text normalization
        if default_options['expand_contractions']:
            cleaned_text = self._expand_contractions(cleaned_text)
            cleaning_steps.append('contraction_expansion')
        
        if default_options['normalize_whitespace']:
            cleaned_text = self._normalize_whitespace(cleaned_text)
            cleaning_steps.append('whitespace_normalization')
        
        if default_options['lowercase']:
            cleaned_text = cleaned_text.lower()
            cleaning_steps.append('lowercasing')
        
        # Step 3: Advanced processing with NLTK/spaCy
        if self.nltk_available or self.spacy_available:
            # Tokenization
            tokens = self._tokenize_text(cleaned_text)
            cleaning_steps.append('tokenization')
            
            # Remove special characters and filter tokens
            if default_options['remove_special_chars']:
                tokens = self._remove_special_characters(tokens)
                cleaning_steps.append('special_char_removal')
            
            if default_options['remove_punctuation']:
                tokens = self._remove_punctuation(tokens)
                cleaning_steps.append('punctuation_removal')
            
            if default_options['remove_numbers']:
                tokens = self._remove_numbers(tokens)
                cleaning_steps.append('number_removal')
            
            # Filter by word length
            tokens = self._filter_by_length(tokens, 
                                          default_options['min_word_length'],
                                          default_options['max_word_length'])
            cleaning_steps.append('length_filtering')
            
            # Remove stopwords
            if default_options['remove_stopwords']:
                tokens = self._remove_stopwords(tokens)
                cleaning_steps.append('stopword_removal')
            
            # Lemmatization or stemming
            if default_options['lemmatize'] and self.nltk_available:
                tokens = self._lemmatize_tokens(tokens)
                cleaning_steps.append('lemmatization')
            elif default_options['stem'] and self.nltk_available:
                tokens = self._stem_tokens(tokens)
                cleaning_steps.append('stemming')
            
            # Join tokens back to text
            cleaned_text = ' '.join(tokens)
        
        # Step 4: Final cleanup
        cleaned_text = self._final_cleanup(cleaned_text)
        cleaning_steps.append('final_cleanup')
        
        # Extract metadata
        metadata = self._extract_cleaning_metadata(original_text, cleaned_text, cleaning_steps)
        
        return {
            'cleaned_text': cleaned_text,
            'original_text': original_text,
            'cleaning_steps': cleaning_steps,
            'metadata': metadata,
            'tokens': self._tokenize_text(cleaned_text) if self.nltk_available else cleaned_text.split(),
            'word_count': len(cleaned_text.split()),
            'char_count': len(cleaned_text)
        }
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        html_pattern = re.compile(r'<[^>]+>')
        return html_pattern.sub('', text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def _remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text"""
        phone_pattern = re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})')
        return phone_pattern.sub('', text)
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions in text"""
        for contraction, expansion in self.contractions.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using NLTK or basic splitting"""
        if self.nltk_available and self.word_tokenize is not None:
            return self.word_tokenize(text)
        else:
            return text.split()
    
    def _remove_special_characters(self, tokens: List[str]) -> List[str]:
        """Remove special characters from tokens"""
        return [token for token in tokens if token.isalnum() or token in string.punctuation]
    
    def _remove_punctuation(self, tokens: List[str]) -> List[str]:
        """Remove punctuation from tokens"""
        return [token for token in tokens if token not in string.punctuation]
    
    def _remove_numbers(self, tokens: List[str]) -> List[str]:
        """Remove numeric tokens"""
        return [token for token in tokens if not token.isdigit()]
    
    def _filter_by_length(self, tokens: List[str], min_length: int, max_length: int) -> List[str]:
        """Filter tokens by length"""
        return [token for token in tokens if min_length <= len(token) <= max_length]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        if self.nltk_available and hasattr(self, 'stopwords') and self.stopwords:
            stopwords_set = self.stopwords
        else:
            stopwords_set = self.fallback_stopwords
        
        return [token for token in tokens if token.lower() not in stopwords_set]
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens using NLTK"""
        if not self.nltk_available or self.lemmatizer is None:
            return tokens
        
        lemmatized = []
        for token in tokens:
            try:
                lemmatized.append(self.lemmatizer.lemmatize(token))
            except:
                lemmatized.append(token)
        return lemmatized
    
    def _stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem tokens using NLTK Porter Stemmer"""
        if not self.nltk_available or self.stemmer is None:
            return tokens
        
        return [self.stemmer.stem(token) for token in tokens]
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing spaces
        return text.strip()
    
    def _extract_cleaning_metadata(self, original: str, cleaned: str, steps: List[str]) -> Dict[str, Any]:
        """Extract metadata about the cleaning process"""
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'reduction_ratio': (len(original) - len(cleaned)) / max(len(original), 1),
            'cleaning_steps_count': len(steps),
            'nltk_used': self.nltk_available,
            'spacy_used': self.spacy_available
        }
    
    def get_cleaning_options(self) -> Dict[str, Any]:
        """Get available cleaning options"""
        return {
            'available_options': [
                'remove_html', 'remove_urls', 'remove_emails', 'remove_phone_numbers',
                'expand_contractions', 'remove_special_chars', 'normalize_whitespace',
                'remove_stopwords', 'lemmatize', 'stem', 'lowercase',
                'remove_punctuation', 'remove_numbers'
            ],
            'nltk_available': self.nltk_available,
            'spacy_available': self.spacy_available,
            'default_options': {
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
            }
        }
