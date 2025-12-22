from services.dictionary_service import get_dictionary_definition

async def get_word_definition(word: str):
    """Get dictionary definition for a word."""
    return await get_dictionary_definition(word)
