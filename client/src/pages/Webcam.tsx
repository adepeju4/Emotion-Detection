import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Camera, Video } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import type { EmotionResult, ModelInfo } from "@/types/emotion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useFaceDetector } from "@/hooks/useFaceDetector";

const EmotionBarChart = ({ predictions }: { predictions: Record<string, number> }) => {
  const data = Object.entries(predictions)
    .map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value: value * 100
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="w-full h-[300px] mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{
            top: 20,
            right: 20,
            left: 20,
            bottom: 60
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="name" 
            angle={-45}
            textAnchor="end"
            height={60}
            interval={0}
          />
          <YAxis 
            label={{ 
              value: 'Confidence (%)', 
              angle: -90, 
              position: 'insideLeft',
              style: { textAnchor: 'middle' }
            }}
            domain={[0, 100]}
          />
          <Tooltip 
            formatter={(value: number) => [`${value.toFixed(1)}%`, 'Confidence']}
            animationDuration={300}
          />
          <Bar 
            dataKey="value" 
            fill="#4F46E5"
            radius={[4, 4, 0, 0]}
            animationDuration={500}
            animationBegin={0}
            isAnimationActive={true}
          >
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`}
                fill={`hsl(${240 + (index * 5)}, 84%, 59%)`}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default function Webcam() {
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<EmotionResult | null>(null);
  const [selectedModel, setSelectedModel] = useState("fer2013");
  const [error, setError] = useState<string>("");
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyzeIntervalRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const { ready: faceReady, detect } = useFaceDetector();

  useEffect(() => {
    api.getModels()
      .then(setModels)
      .catch(error => {
        console.error('Failed to fetch models:', error);
        setError('Failed to connect to the API. Please try again later.');
      });

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (analyzeIntervalRef.current) {
        clearInterval(analyzeIntervalRef.current);
      }
    };
  }, []);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          if (overlayRef.current && videoRef.current) {
            overlayRef.current.width = videoRef.current!.videoWidth;
            overlayRef.current.height = videoRef.current!.videoHeight;
          }
          setIsWebcamActive(true);
          setError("");
          startAnalysis();
        };
      }
    } catch (err) {
      setError("Failed to access webcam. Please ensure you have granted camera permissions.");
      console.error("Error accessing webcam:", err);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (analyzeIntervalRef.current) {
      clearInterval(analyzeIntervalRef.current);
      analyzeIntervalRef.current = null;
    }
    setIsWebcamActive(false);
    setAnalysisResults(null);
  };

  const analyzeFrame = async () => {
    if (!isWebcamActive || isAnalyzing || !faceReady) return;
    
    if (!videoRef.current || !overlayRef.current) return;

    setIsAnalyzing(true);

    try {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      const ctx = overlay.getContext('2d');
      if (!ctx) throw new Error('Canvas context unavailable');

      ctx.clearRect(0, 0, overlay.width, overlay.height);

      const faces = await detect(video, overlay);

      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2;
      faces.forEach(({ box }) => {
        ctx.strokeRect(box.x, box.y, box.width, box.height);
      });

      const predictions = await Promise.all(
        faces.map(async ({ image }) => {
          const blob = await new Promise<Blob>((resolve) =>
            image.toBlob((b) => resolve(b as Blob), 'image/jpeg', 0.9)
          );
          const result = await api.analyzeImage(blob, selectedModel);
          return result.faces && result.faces.length > 0 ? result.faces[0] : null;
        })
      );

      ctx.fillStyle = '#22c55e';
      ctx.font = '16px sans-serif';
      predictions.forEach((pred, idx) => {
        if (pred) {
          const box = faces[idx].box;
          ctx.fillText(`${pred.label} ${(pred.confidence * 100).toFixed(0)}%`, box.x, box.y - 6);
        }
      });

      const validPredictions = predictions.filter(p => p !== null);
      if (validPredictions.length > 0) {
        setAnalysisResults(validPredictions[0]);
      }
      setError('');
    } catch (error) {
      console.error('Error analyzing frame:', error);
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError('Failed to analyze frame. Please try again.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const startAnalysis = () => {
    if (analyzeIntervalRef.current) {
      clearInterval(analyzeIntervalRef.current);
    }
    
    analyzeIntervalRef.current = window.setInterval(analyzeFrame, 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-24">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center gap-4 mb-8">
          <Link to="/" className="text-gray-500 hover:text-gray-700">
            <ArrowLeft className="w-6 h-6" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Real-time Emotion Analysis</h1>
            <p className="text-gray-600">Use your webcam for live emotion detection</p>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          <Card className="p-8">
            <h2 className="text-xl font-semibold mb-6">Webcam Controls</h2>
            
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select AI Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isWebcamActive}
              >
                {Object.entries(models).map(([id, info]) => (
                  <option 
                    key={id} 
                    value={id}
                    disabled={!info.available}
                  >
                    {id.toUpperCase()} ({info.available ? 'Available' : 'Not Available'})
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-4">
              <Button
                onClick={isWebcamActive ? stopWebcam : startWebcam}
                className="w-full bg-gradient-to-r from-blue to-purple-600 text-white hover:from-blue-600 hover:to-purple-700"
              >
                {isWebcamActive ? (
                  <>
                    <Video className="w-4 h-4 mr-2" />
                    Stop Webcam
                  </>
                ) : (
                  <>
                    <Camera className="w-4 h-4 mr-2" />
                    Start Webcam
                  </>
                )}
              </Button>

              {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-800">{error}</p>
                </div>
              )}
            </div>
          </Card>

          <Card className="p-8">
            <h2 className="text-xl font-semibold mb-6">Webcam Preview</h2>
            
            <div className="space-y-4">
              <div className="relative bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-[480px] object-contain"
                />
                <canvas
                  ref={overlayRef}
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                />
                
                {analysisResults && (
                  <div className="absolute top-2 right-2 bg-black/75 text-white p-4 rounded-lg">
                    <p className="font-semibold">{analysisResults.label}</p>
                    <p className="text-sm opacity-75">
                      Confidence: {(analysisResults.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                )}

                {isAnalyzing && (
                  <div className="absolute bottom-2 right-2">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                  </div>
                )}
              </div>

              <canvas ref={canvasRef} className="hidden" />

              {analysisResults && (
                <div className="mt-8 p-4 bg-white border rounded-lg">
                  <h3 className="font-semibold text-gray-800 mb-4">Emotion Analysis</h3>
                  <EmotionBarChart predictions={analysisResults.all_predictions} />
                  <p className="text-xs text-gray-500 mt-4 text-center">
                    Processing time: {analysisResults.processing_time.toFixed(3)}s
                  </p>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
} 