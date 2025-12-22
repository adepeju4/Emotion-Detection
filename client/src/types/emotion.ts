export interface EmotionResult {
  label: string;
  confidence: number;
  model: string;
  processing_time: number;
  all_predictions: Record<string, number>;
}

export interface FrameAnalysis {
  timestamp: number;
  result: EmotionResult;
}

export interface ModelInfo {
  available: boolean;
  labels: string[];
  input_shape: [number, number];
} 