export interface EmotionDefinition {
  name: string;
  icon: string;
  autismContext: string;
  relatedEmotions?: string[];
  physicalCues?: string[];
}

export const getEmotionDefinition = (emotionName: string): EmotionDefinition | null => {
  return null;
};



