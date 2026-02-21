export interface Definition {
  partOfSpeech: string;
  definition: string;
  example?: string;
}

export interface DictionaryResult {
  word: string;
  phonetic?: string;
  definitions: Definition[];
}

export async function fetchDefinition(word: string): Promise<DictionaryResult | null> {
  try {
    const response = await fetch(
      `https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(word)}`
    );

    if (!response.ok) return null;

    const data = await response.json() as Array<{
      word: string;
      phonetic?: string;
      meanings: Array<{
        partOfSpeech: string;
        definitions: Array<{ definition: string; example?: string }>;
      }>;
    }>;

    if (!Array.isArray(data) || data.length === 0) return null;

    const entry = data[0];
    const definitions: Definition[] = [];

    for (const meaning of entry.meanings ?? []) {
      for (const def of meaning.definitions ?? []) {
        definitions.push({
          partOfSpeech: meaning.partOfSpeech,
          definition: def.definition,
          example: def.example,
        });
      }
    }

    return {
      word: entry.word,
      phonetic: entry.phonetic,
      definitions,
    };
  } catch {
    return null;
  }
}
