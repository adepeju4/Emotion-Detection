import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Camera, CameraOff, RotateCcw } from "lucide-react";
import { useState, useRef, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import type { ModelInfo } from "@/types/emotion";
import type { AnalysisResponse, FaceResult } from "@/lib/api";

export default function WebcamAnalysis() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("fer2013");
  const [error, setError] = useState<string>("");
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [, setCapturedImage] = useState<string | null>(null);
  const [, setShowCapturedImage] = useState(false);
  const [autoAnalyze, setAutoAnalyze] = useState(false);
  const [analysisInterval, setAnalysisInterval] = useState<number | null>(null);
  const analysisIntervalRef = useRef<number | null>(null);
  const [croppedFaces, setCroppedFaces] = useState<Record<number, string>>({});
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    api.getModels()
      .then(setModels)
      .catch(error => {
        console.error('Failed to fetch models:', error);
        setError('Failed to connect to the API. Please try again later.');
      });
  }, []);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });
      
      if (videoRef.current) {
        const videoEl = videoRef.current;
        videoEl.srcObject = stream as MediaStream;
        streamRef.current = stream;

        
        videoEl.onloadedmetadata = () => {
          
          videoEl.oncanplay = () => {
            setIsStreaming(true);
            setError("");
          };
          
          videoEl.play?.().catch(() => {});
        };
      }
    } catch (error) {
      console.error('Error accessing webcam:', error);
      setError("Unable to access webcam. Please check permissions and try again.");
    }
  }, []);

  const stopRealTimeAnalysis = useCallback(() => {
    const id = analysisIntervalRef.current ?? analysisInterval;
    if (id !== null) {
      window.clearInterval(id);
      analysisIntervalRef.current = null;
      setAnalysisInterval(null);
    }
    setAutoAnalyze(false);
  }, [analysisInterval]);

  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setAnalysisResults(null);
    setCapturedImage(null);
    setShowCapturedImage(false);
    stopRealTimeAnalysis();
  }, [stopRealTimeAnalysis]);

  const captureFrame = useCallback((): Promise<Blob | null> => {
    if (!videoRef.current || !canvasRef.current) {
      console.log('Capture failed: videoRef or canvasRef not available');
      return Promise.resolve(null);
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.log('Capture failed: canvas context not available');
      return Promise.resolve(null);
    }

    
    if (video.readyState < 2) {
      console.log('Capture failed: video not ready, readyState:', video.readyState);
      return Promise.resolve(null);
    }

    
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log('Capture failed: video has no dimensions');
      return Promise.resolve(null);
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    return new Promise(resolve => {
      canvas.toBlob(blob => {
        console.log('Frame captured successfully, blob size:', blob?.size);
        resolve(blob);
      }, 'image/jpeg', 0.95);
    });
  }, []);

  const analyzeFrame = useCallback(async () => {
    console.log('analyzeFrame called, isStreaming:', isStreaming, 'videoRef:', !!videoRef.current);
    
    if (!isStreaming) {
      setError("Webcam is not active. Please start the webcam first.");
      return;
    }
    
    if (!videoRef.current) {
      setError("Video element not found. Please refresh the page.");
      return;
    }
    
    setIsAnalyzing(true);
    setError("");
    
    try {
      
      await new Promise(resolve => setTimeout(resolve, 200));
      
      const frame = await captureFrame();
      if (!frame) {
        setError("Failed to capture frame from webcam - video may not be ready yet");
        return;
      }

      const result = await api.analyzeImage(frame, selectedModel);
      setAnalysisResults(result);

      if (result.faces && result.faces.length > 0 && canvasRef.current) {
        const canvas = canvasRef.current;
        const crops: Record<number, string> = {};
        for (const face of result.faces) {
          const { x, y, width, height } = face.bbox;
          const cropCanvas = document.createElement('canvas');
          cropCanvas.width = width;
          cropCanvas.height = height;
          const ctx = cropCanvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(canvas, x, y, width, height, 0, 0, width, height);
            crops[face.face_id] = cropCanvas.toDataURL('image/jpeg', 0.85);
          }
        }
        setCroppedFaces(crops);
      }


    } catch (error) {
      console.error('Error analyzing frame:', error);
      
      if (!autoAnalyze) {
        if (error instanceof Error) {
          setError(error.message);
        } else {
          setError("Failed to analyze frame. Please try again.");
        }
      }
    } finally {
      setIsAnalyzing(false);
    }
  }, [isStreaming, selectedModel, captureFrame, autoAnalyze]);

  const startRealTimeAnalysis = useCallback(() => {
    if (!isStreaming) {
      setError("Please start the webcam first before starting real-time analysis.");
      return;
    }
    
    if (analysisInterval !== null) {
      window.clearInterval(analysisInterval);
      setAnalysisInterval(null);
    }
    
    
    setTimeout(() => {
      const interval = window.setInterval(() => {
        if (!isAnalyzing && isStreaming && videoRef.current && 
            videoRef.current.readyState >= 3 && 
            videoRef.current.videoWidth > 0 && 
            videoRef.current.videoHeight > 0) {
          analyzeFrame();
        }
      }, 3000);
      
      setAnalysisInterval(interval);
      analysisIntervalRef.current = interval;
      setAutoAnalyze(true);
    }, 2000);
  }, [isAnalyzing, isStreaming, analyzeFrame, analysisInterval]);

  const resetAnalysis = useCallback(() => {
    setAnalysisResults(null);
    setCapturedImage(null);
    setShowCapturedImage(false);
    setCroppedFaces({});
  }, []);

  
  useEffect(() => {
    return () => {
      if (analysisIntervalRef.current !== null) {
        window.clearInterval(analysisIntervalRef.current);
      }
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-white py-16 sm:py-20 md:py-24">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <article className="prose prose-lg prose-gray max-w-none">
          <header className="mb-8 sm:mb-12">
            <div className="flex items-center gap-4 mb-4 sm:mb-6">
              <Link to="/" className="text-gray-500 hover:text-gray-700">
                <ArrowLeft className="w-5 h-5 sm:w-6 sm:h-6" />
              </Link>
            </div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-gray-900 mb-3 sm:mb-4">Real-Time Webcam Analysis</h1>
            <p className="text-base sm:text-lg md:text-xl text-gray-600 mb-4 sm:mb-6">Analyze emotions in real-time using your webcam</p>
            <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-0 text-xs sm:text-sm text-gray-500 mb-6 sm:mb-8">
              <span>By Adepeju Peace Orefejo</span>
              <span className="hidden sm:inline mx-2">•</span>
              <span>MSc. Software Engineering 2025</span>
            </div>
          </header>

          <section className="mb-8 sm:mb-12">
            <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-4 sm:mb-8">Live Emotion Detection</h2>
            <p className="text-sm sm:text-base text-gray-700 mb-6 sm:mb-8">
              Use your webcam to capture and analyze emotions in real-time. The system will detect faces and provide detailed emotion analysis with confidence scores.
            </p>
            
            
            <div className="bg-amber-50 border-l-4 border-amber-400 p-4 sm:p-6 mb-6 sm:mb-8">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-amber-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-amber-800">Important Notice</h3>
                  <div className="mt-2 text-sm text-amber-700">
                    <p>
                      This emotion detection system is for research and educational purposes. Model accuracy varies by dataset and conditions. 
                      For detailed performance metrics, limitations, and methodology, please refer to our{' '}
                      <Link to="/research" className="font-medium underline hover:text-amber-900">
                        Research Findings
                      </Link>.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <div className="grid lg:grid-cols-2 gap-8">
            
            <Card className="p-8 border border-gray-200 shadow-sm">
              <h3 className="text-xl font-semibold mb-6 text-gray-900">Webcam Controls</h3>
              
              
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select AI Model
                </label>
                <select
                  title="Select AI Model"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={isAnalyzing}
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
                {!isStreaming ? (
                  <Button
                    onClick={startWebcam}
                    className="w-full bg-green-600 text-white hover:bg-green-700 flex items-center justify-center gap-2"
                  >
                    <Camera className="w-5 h-5" />
                    Start Webcam
                  </Button>
                ) : (
                  <Button
                    onClick={stopWebcam}
                    className="w-full bg-red-600 text-white hover:bg-red-700 flex items-center justify-center gap-2"
                  >
                    <CameraOff className="w-5 h-5" />
                    Stop Webcam
                  </Button>
                )}

                {isStreaming && (
                  <>
                    {!autoAnalyze ? (
                      <Button
                        onClick={startRealTimeAnalysis}
                        className="w-full bg-gradient-to-r from-green to-blue-600 text-white hover:from-green-600 hover:to-blue-700 flex items-center justify-center gap-2"
                      >
                        <Camera className="w-5 h-5" />
                        Start Real-Time Analysis
                      </Button>
                    ) : (
                      <Button
                        onClick={stopRealTimeAnalysis}
                        className="w-full bg-gradient-to-r from-red to-orange-600 text-white hover:from-red-600 hover:to-orange-700 flex items-center justify-center gap-2"
                      >
                        <CameraOff className="w-5 h-5" />
                        Stop Real-Time Analysis
                      </Button>
                    )}

                    <Button
                      onClick={analyzeFrame}
                      disabled={isAnalyzing}
                      variant="outline"
                      className="w-full"
                    >
                      {isAnalyzing ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
                          Analyzing...
                        </>
                      ) : (
                        'Analyze Current Frame'
                      )}
                    </Button>

                    <Button
                      onClick={resetAnalysis}
                      variant="outline"
                      className="w-full flex items-center justify-center gap-2"
                    >
                      <RotateCcw className="w-4 h-4" />
                      Reset Analysis
                    </Button>
                  </>
                )}
              </div>

              {error && (
                <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-800">{error}</p>
                </div>
              )}
            </Card>

            
            <Card className="p-8 border border-gray-200 shadow-sm">
              <h3 className="text-xl font-semibold mb-6 text-gray-900">Live Preview & Analysis</h3>
              
              <div className="space-y-4">
                
                <div className="relative bg-black rounded-lg overflow-hidden">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-64 object-cover"
                  />
                  
                  
                  {analysisResults && (
                    <div className="absolute top-2 right-2 bg-black/75 text-white p-4 rounded-lg">
                      <p className="font-semibold">Faces Detected: {analysisResults.faces_detected}</p>
                      <p className="text-sm opacity-75">
                        Processing time: {analysisResults.processing_time.toFixed(3)}s
                      </p>
                      {autoAnalyze && (
                        <p className="text-xs opacity-75 mt-1">
                          🔴 Real-time analysis active
                        </p>
                      )}
                    </div>
                  )}

                  
                  {isAnalyzing && (
                    <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                      <div className="text-center text-white">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
                        <p>{autoAnalyze ? 'Analyzing Frame...' : 'Analyzing Frame...'}</p>
                      </div>
                    </div>
                  )}
                </div>

                
                <canvas ref={canvasRef} className="hidden" />

                


                
                {analysisResults && (
                  <div className="mt-8 space-y-6">
                    <h4 className="text-lg font-semibold text-gray-900 mb-4">Analysis Results</h4>
                    {analysisResults.faces && analysisResults.faces.map((face: FaceResult) => (
                      <div key={face.face_id} className="p-6 bg-gray-50 border border-gray-200 rounded-lg">
                        <div className="flex items-start gap-6 mb-6">
                          <div className="flex-shrink-0">
                            <h5 className="font-semibold mb-3 text-gray-900">Face {face.face_id}: {face.label}</h5>
                            {croppedFaces[face.face_id] && (
                              <div className="w-28 h-28 border-2 border-gray-300 rounded-lg overflow-hidden shadow-sm">
                                <img
                                  src={croppedFaces[face.face_id]}
                                  alt={`Cropped face ${face.face_id}`}
                                  className="w-full h-full object-cover"
                                />
                              </div>
                            )}
                          </div>
                          <div className="flex-1">
                            <div className="space-y-4">
                              {Object.entries(face.all_predictions)
                                .sort(([, a], [, b]) => (b as number) - (a as number))
                                .map(([emotion, confidence]) => (
                                  <div key={emotion} className="flex items-center justify-between">
                                    <span className="capitalize font-medium text-gray-700">{emotion}</span>
                                    <div className="flex items-center gap-4">
                                      <div className="w-40 bg-gray-200 rounded-full h-2">
                                        <div
                                          className={`bg-${emotion.toLowerCase()} h-2 rounded-full transition-all duration-500 w-${(confidence as number) * 100}%`}
                                        ></div>
                                      </div>
                                      <span className="text-sm font-medium w-12 text-right text-gray-600">
                                        {((confidence as number) * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                  </div>
                                ))}
                            </div>
                            <p className="text-sm text-gray-500 mt-4">
                              <strong>Overall Confidence:</strong> {(face.confidence * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                    <div className="p-6 bg-blue-50 border border-blue-200 rounded-lg">
                      <h5 className="font-semibold text-blue-900 mb-2">Analysis Summary</h5>
                      <p className="text-sm text-blue-800 mb-1">
                        <strong>Total faces detected:</strong> {analysisResults.faces_detected}
                      </p>
                      <p className="text-sm text-blue-800">
                        <strong>Processing time:</strong> {analysisResults.processing_time.toFixed(3)}s
                      </p>
                    </div>
                  </div>
                )}

                {!isStreaming && (
                  <div className="flex items-center justify-center h-64 bg-gray-100 rounded-lg">
                    <div className="text-center">
                      <Camera className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-500">Start webcam to begin analysis</p>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </div>
        </article>
      </div>
    </div>
  );
}
