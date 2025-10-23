"""
Gemini Predictor Module
Handles predictions using Google's Gemini AI while maintaining HuggingFace-like interface
"""

from google import genai
from google.genai import types
import os
from typing import Dict, Any, Union, List
import json
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

class GeminiPredictor:
    def __init__(self):
        """Initialize Gemini predictor"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Configure the client with API key (correct modern syntax)
        self.client = genai.Client(api_key=self.api_key)
        
        # Model names for text and vision
        self.text_model_name = 'gemini-2.5-flash'
        self.vision_model_name = 'gemini-2.5-flash'  
        
    def predict_text(self, text: str) -> Dict[str, Any]:
        """Predict using text input"""
        prompt = self._create_analysis_prompt(text)
        try:
            response = self.client.models.generate_content(
                model=self.text_model_name,
                contents=prompt
            )
            
            if not response.text:
                raise ValueError("Empty response from model")
            
            result = self._parse_response(response.text)
            
            return {
                "status": "success",
                "prediction": result["prediction"],
                "probability": result["confidence"] / 100,  # Convert to 0-1 scale
                "explanation": result["reasoning"],
                "red_flags": result.get("red_flags", []),
                "source_credibility": result.get("source_credibility", "Unknown"),
                "manipulation_indicators": result.get("manipulation_indicators", []),
                "fact_check_recommendations": result.get("fact_check_recommendations", [])
            }
        except Exception as e:
            return {
                "status": "error",
                "prediction": "ERROR",
                "probability": 0.5,
                "explanation": f"Analysis failed: {str(e)}",
                "red_flags": ["Analysis error occurred"],
                "source_credibility": "Unknown"
            }

    def predict_image(self, image_input: Union[str, bytes]) -> Dict[str, Any]:
        """Predict using image input"""
        try:
            # Handle different image input types
            if isinstance(image_input, str):
                if image_input.startswith(('http://', 'https://')):
                    response_img = requests.get(image_input)
                    img = Image.open(BytesIO(response_img.content))
                else:
                    img = Image.open(image_input)
            else:
                img = Image.open(BytesIO(image_input))

            # Convert PIL Image to bytes
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            prompt = """Analyze this image for potential fake news indicators. Consider:
1. Image manipulation or editing signs
2. Context and authenticity
3. Any misleading elements or doctored content
4. Visual inconsistencies
5. Metadata concerns

Provide your analysis in JSON format with:
{
    "prediction": "REAL" or "FAKE",
    "confidence": (0-100),
    "reasoning": "detailed explanation",
    "image_analysis": "specific image findings",
    "manipulation_detected": true/false,
    "red_flags": ["list of issues"]
}"""
            
            # Create the content with image using correct types
            response = self.client.models.generate_content(
                model=self.vision_model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type='image/png',
                                    data=img_bytes
                                )
                            )
                        ]
                    )
                ]
            )
            
            if not response.text:
                raise ValueError("Empty response from model")
            
            result = self._parse_response(response.text)
            
            return {
                "status": "success",
                "prediction": result["prediction"],
                "probability": result["confidence"] / 100,
                "explanation": result["reasoning"],
                "image_analysis": result.get("image_analysis", "No specific image analysis provided"),
                "manipulation_detected": result.get("manipulation_detected", False),
                "red_flags": result.get("red_flags", [])
            }
        except Exception as e:
            return {
                "status": "error",
                "prediction": "ERROR",
                "probability": 0.5,
                "explanation": f"Image analysis failed: {str(e)}",
                "image_analysis": "Analysis error occurred",
                "manipulation_detected": False,
                "red_flags": ["Image analysis error"]
            }

    def _create_analysis_prompt(self, text: str) -> str:
        """Create a detailed prompt for news analysis"""
        return f"""As an expert fact-checker and fake news detector, analyze this content for authenticity.
Provide your analysis in JSON format with the following structure:
{{
    "prediction": "REAL" or "FAKE",
    "confidence": (number between 0-100),
    "reasoning": "detailed explanation of why",
    "red_flags": ["list", "of", "suspicious", "elements"],
    "source_credibility": "assessment of source credibility",
    "manipulation_indicators": ["list", "of", "potential", "manipulation", "techniques"],
    "fact_check_recommendations": ["specific", "points", "to", "verify"]
}}

Content to analyze:
{text}

Consider these factors:
1. Writing style and tone (sensationalist vs. neutral)
2. Source credibility and reputation
3. Fact verification and evidence quality
4. Emotional manipulation tactics
5. Logical consistency and coherence
6. Expert citations and their validity
7. Technical accuracy of claims
8. Bias indicators and political slant
9. Context and timing of publication
10. Cross-reference potential with known facts

Be thorough and provide specific evidence for your assessment."""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's response into structured format"""
        try:
            # Try to parse as JSON first
            if '{' in response_text and '}' in response_text:
                # Extract JSON from response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
                
                # Clean up potential markdown code blocks
                json_str = json_str.replace('```json', '').replace('```', '').strip()
                
                parsed = json.loads(json_str)
                
                # Ensure all required fields exist
                if 'prediction' not in parsed:
                    parsed['prediction'] = 'UNCERTAIN'
                if 'confidence' not in parsed:
                    parsed['confidence'] = 50
                if 'reasoning' not in parsed:
                    parsed['reasoning'] = response_text
                    
                return parsed
            
            # Fallback parsing if not JSON
            is_fake = any(word in response_text.lower() for word in ['fake', 'false', 'misleading', 'deceptive', 'fabricated', 'misinformation'])
            is_real = any(word in response_text.lower() for word in ['real', 'true', 'authentic', 'genuine', 'legitimate'])
            
            # Determine confidence based on language strength
            confidence = 50  # default
            if any(word in response_text.lower() for word in ['certainly', 'clearly', 'definitely', 'obviously', 'undoubtedly']):
                confidence = 80
            elif any(word in response_text.lower() for word in ['likely', 'probably', 'appears', 'seems']):
                confidence = 65
            elif any(word in response_text.lower() for word in ['possibly', 'might', 'could', 'perhaps']):
                confidence = 55
            
            # Determine prediction
            if is_fake and not is_real:
                prediction = "FAKE"
            elif is_real and not is_fake:
                prediction = "REAL"
            else:
                prediction = "UNCERTAIN"
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reasoning": response_text,
                "red_flags": self._extract_red_flags(response_text),
                "source_credibility": "Needs verification",
                "manipulation_indicators": [],
                "fact_check_recommendations": []
            }
            
        except json.JSONDecodeError as e:
            # JSON parsing failed, use fallback
            return {
                "prediction": "UNCERTAIN",
                "confidence": 50,
                "reasoning": f"Failed to parse analysis as JSON: {response_text[:500]}",
                "red_flags": self._extract_red_flags(response_text),
                "source_credibility": "Unknown"
            }
        except Exception as e:
            return {
                "prediction": "UNCERTAIN",
                "confidence": 50,
                "reasoning": f"Failed to parse analysis: {str(e)}",
                "red_flags": ["Analysis parsing error"],
                "source_credibility": "Unknown"
            }
    
    def _extract_red_flags(self, text: str) -> List[str]:
        """Extract red flags from response text"""
        red_flags = []
        lower_text = text.lower()
        
        # Common red flag indicators with their descriptions
        indicators = {
            'emotional manipulation': 'Uses emotional manipulation',
            'emotionally charged': 'Uses emotional manipulation',
            'no source': 'Lacks credible sources',
            'lacking source': 'Lacks credible sources',
            'unverified': 'Contains unverified claims',
            'conspiracy': 'Contains conspiracy theories',
            'clickbait': 'Uses clickbait tactics',
            'sensational': 'Uses sensationalist language',
            'outdated': 'Contains outdated information',
            'inconsistent': 'Shows inconsistencies',
            'contradiction': 'Contains contradictions',
            'exaggerated': 'Uses exaggerated claims',
            'biased': 'Shows clear bias',
            'propaganda': 'Contains propaganda elements',
            'misleading': 'Contains misleading information',
            'manipulated': 'Shows signs of manipulation',
            'doctored': 'May contain doctored content',
            'out of context': 'Information taken out of context'
        }
        
        for indicator, flag in indicators.items():
            if indicator in lower_text and flag not in red_flags:
                red_flags.append(flag)
        
        return red_flags if red_flags else ["No specific red flags identified"]
    
    def batch_predict_text(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict multiple texts in batch"""
        results = []
        for text in texts:
            result = self.predict_text(text)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the models being used"""
        return {
            "text_model": self.text_model_name,
            "vision_model": self.vision_model_name,
            "api_configured": self.api_key is not None,
            "capabilities": {
                "text_analysis": True,
                "image_analysis": True,
                "batch_processing": True
            }
        }