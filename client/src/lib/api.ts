import type { ModelInfo } from "@/types/emotion";

const BASE_URL = import.meta.env.VITE_API_URL ?? "";

export interface FaceResult {
  face_id: number;
  label: string;
  confidence: number;
  all_predictions: Record<string, number>;
  cropped_face?: string;
}

export interface AnalysisResponse {
  model: string;
  processing_time: number;
  faces_detected: number;
  faces: FaceResult[];
}

async function analyzeImage(image: Blob | File, model: string): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", image);
  formData.append("model", model);

  const response = await fetch(`${BASE_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<AnalysisResponse>;
}

async function getModels(): Promise<Record<string, ModelInfo>> {
  const response = await fetch(`${BASE_URL}/api/models`);

  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.status}`);
  }

  return response.json() as Promise<Record<string, ModelInfo>>;
}

export const api = { analyzeImage, getModels };
