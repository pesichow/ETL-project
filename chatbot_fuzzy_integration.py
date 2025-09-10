# chatbot_fuzzy_integration.py
from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional
from fuzzy_utils import fuzzy_matcher
from data_storage import data_store
import logging
from pydantic import BaseModel
from fuzzywuzzy import process, fuzz

router = APIRouter(
    prefix="/api/v1/chatbot",
    tags=["Chatbot with Fuzzy Matching"],
    responses={404: {"description": "Not found"}}
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    correct_text: bool = True
    threshold: int = 75
    preserve_case: bool = True
    aggressive_mode: bool = False

class VocabularyUpdate(BaseModel):
    terms: List[str]

@router.post("/process")
async def process_chatbot_message(request: ChatbotRequest = Body(...)):
    """Process chatbot message with advanced fuzzy matching"""
    try:
        # Prepare context
        targets = []
        if request.session_id and request.session_id in data_store:
            df = data_store[request.session_id]["df"]
            targets.extend(df.values.flatten().tolist())
            fuzzy_matcher.add_special_terms(df.columns.tolist())
        
        # Combine all possible targets
        all_targets = list(fuzzy_matcher.common_words.union(fuzzy_matcher.special_terms))
        all_targets.extend(targets)
        all_targets = list(set(str(t) for t in all_targets if t and str(t).strip()))
        
        # Process message
        words = request.message.split()
        corrections = []
        corrected_words = []
        
        for word in words:
            # Skip URLs, emails, etc.
            if any(c in word for c in ['@', ':', '/', '.com']):
                corrected_words.append(word)
                continue
                
            # Check if word is known
            if fuzzy_matcher.is_known_word(word):
                corrected_words.append(word)
                continue
                
            # Find best match
            match = fuzzy_matcher.find_best_match(word, all_targets, request.threshold)
            
            # Apply correction if good match found
            if match:
                corrected_word = match[0]
                if request.preserve_case:
                    if word.istitle():
                        corrected_word = corrected_word.title()
                    elif word.isupper():
                        corrected_word = corrected_word.upper()
                
                corrections.append({
                    "original": word,
                    "corrected": corrected_word,
                    "confidence": match[1],
                    "source": "dataset" if request.session_id else "dictionary"
                })
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        # Aggressive mode for remaining words
        if request.aggressive_mode:
            for i, word in enumerate(corrected_words):
                if not fuzzy_matcher.is_known_word(word):
                    match = fuzzy_matcher.find_best_match(
                        word,
                        list(fuzzy_matcher.common_words),
                        request.threshold - 10
                    )
                    if match:
                        corrections.append({
                            "original": word,
                            "corrected": match[0],
                            "confidence": match[1],
                            "source": "aggressive_correction"
                        })
                        corrected_words[i] = match[0]
        
        return {
            "original_message": request.message,
            "processed_message": " ".join(corrected_words),
            "corrections": corrections,
            "correction_stats": {
                "total_words": len(words),
                "corrected_words": len(corrections),
                "correction_rate": f"{(len(corrections)/len(words))*100:.1f}%"
            }
        }

    except Exception as e:
        logger.error(f"Chatbot processing error: {str(e)}")
        raise HTTPException(500, detail=str(e))

@router.post("/add-vocabulary")
async def add_vocabulary(request: VocabularyUpdate = Body(...)):
    """Add domain-specific terms to the chatbot's dictionary"""
    try:
        fuzzy_matcher.add_special_terms(request.terms)
        return {
            "status": "success",
            "terms_added": len(request.terms),
            "total_terms": len(fuzzy_matcher.special_terms)
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@router.get("/dictionary-info")
async def get_dictionary_info():
    """Get information about the chatbot's dictionary"""
    return {
        "common_words_count": len(fuzzy_matcher.common_words),
        "special_terms_count": len(fuzzy_matcher.special_terms),
        "sample_terms": list(fuzzy_matcher.special_terms)[:10] if fuzzy_matcher.special_terms else []
    }