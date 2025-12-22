import httpx
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

async def get_dictionary_definition(word: str) -> dict:
    """Fetch dictionary definition from Free Dictionary API."""
    try:
        word_lower = word.lower()
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word_lower}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            
            if response.status_code == 404:
                return {"definitions": None, "phonetic": None, "error": "Word not found"}
            
            if not response.is_success:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Dictionary API error: {response.status_code}"
                )
            
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                first_entry = data[0]
                definitions_list = []
                phonetic = None
                
                if "phonetic" in first_entry and first_entry["phonetic"]:
                    phonetic = first_entry["phonetic"]
                elif "phonetics" in first_entry and len(first_entry["phonetics"]) > 0:
                    for ph in first_entry["phonetics"]:
                        if ph.get("text"):
                            phonetic = ph.get("text")
                            break
                
                if "meanings" in first_entry:
                    for meaning in first_entry["meanings"]:
                        part_of_speech = meaning.get("partOfSpeech", "")
                        if "definitions" in meaning:
                            for def_item in meaning["definitions"]:
                                definition_text = def_item.get("definition")
                                example = def_item.get("example")
                                
                                if definition_text:
                                    def_obj = {
                                        "definition": definition_text,
                                        "partOfSpeech": part_of_speech
                                    }
                                    if example:
                                        def_obj["example"] = example
                                    definitions_list.append(def_obj)
                
                if definitions_list:
                    result = {
                        "definitions": definitions_list[:10],
                        "phonetic": phonetic
                    }
                    return result
            
            return {"definitions": None, "phonetic": None, "error": "No definition found"}
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timeout")
    except httpx.RequestError as e:
        logger.error(f"Error fetching dictionary definition: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch definition")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
