import { useEffect, useState } from "react";
import faceapi from "@vladmandic/face-api";

export type DetectedFace = {
  box: faceapi.Box;
  image: HTMLCanvasElement;
};

 
export function useFaceDetector() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        await faceapi.nets.tinyFaceDetector.loadFromUri("/models");
        setReady(true);
      } catch (err) {
        console.error("Failed to load face-api models", err);
      }
    })();
  }, []);

  const detect = async (video: HTMLVideoElement): Promise<DetectedFace[]> => {
    if (!ready) return [];

    const detections = await faceapi.detectAllFaces(
      video,
      new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.5 }),
    );

    const faces: DetectedFace[] = detections.map(
      (det: {
        box: { x: number; y: number; width: number; height: number };
      }) => {
        const { x, y, width, height } = det.box;
        const w = Math.round(width);
        const h = Math.round(height);
        const sx = Math.round(x);
        const sy = Math.round(y);
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = w;
        faceCanvas.height = h;
        const ctx = faceCanvas.getContext("2d")!;
        ctx.drawImage(video, sx, sy, w, h, 0, 0, w, h);
        return {
          box: { x: sx, y: sy, width: w, height: h },
          image: faceCanvas,
        };
      },
    );

    return faces;
  };

  return { ready, detect };
} 