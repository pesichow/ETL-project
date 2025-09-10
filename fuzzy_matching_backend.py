from fastapi import APIRouter
from spellchecker import SpellChecker
from typing import Optional

router = APIRouter()

# Language-specific spell checkers
SPELL_CHECKERS = {
    'en': SpellChecker(),  # English
    'es': SpellChecker(language='es'),  # Spanish
    'fr': SpellChecker(language='fr'),  # French
    # Add more languages as needed
}

DEFAULT_LANGUAGE = 'en'

@router.post("/correct-spelling")
async def correct_spelling(
    text: str,
    language: Optional[str] = DEFAULT_LANGUAGE,
    only_first_suggestion: Optional[bool] = False
):
    """
    Correct spelling for any word in the input text
    
    Parameters:
    - text: Input text to correct
    - language: Supported languages: 'en', 'es', 'fr'
    - only_first_suggestion: Return only top suggestion
    
    Returns:
    - Original and corrected text
    - Correction details per word
    """
    try:
        # Get the appropriate spell checker
        spell = SPELL_CHECKERS.get(language, SPELL_CHECKERS[DEFAULT_LANGUAGE])
        
        corrections = []
        words = text.split()
        
        for word in words:
            # Get spelling info
            is_correct = word in spell
            top_correction = spell.correction(word) if not is_correct else word
            candidates = list(spell.candidates(word)) if not is_correct else []
            
            correction_data = {
                "original": word,
                "corrected": top_correction,
                "is_correct": is_correct,
                "suggestions": candidates[:1] if only_first_suggestion else candidates
            }
            corrections.append(correction_data)
        
        # Generate corrected text
        corrected_text = " ".join([c["corrected"] for c in corrections])
        
        return {
            "original": text,
            "corrected_text": corrected_text,
            "language": language,
            "corrections": corrections
        }
        
    except Exception as e:
        return {"error": str(e)}