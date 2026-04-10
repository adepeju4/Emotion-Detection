export interface EmotionDefinition {
  name: string;
  icon: string;
  autismContext: string;
  relatedEmotions?: string[];
  physicalCues?: string[];
}

export const getEmotionDefinition = (): EmotionDefinition | null => {
  return null;
};



