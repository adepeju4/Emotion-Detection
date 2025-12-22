import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Upload, Play, Pause, RotateCcw, Download, Volume2, VolumeX } from "lucide-react";

import { Link } from "react-router-dom";
import { useState, useRef, useCallback, useEffect } from "react";
import { api } from "@/lib/api";
import type { EmotionResult, ModelInfo, FrameAnalysis } from "@/types/emotion";

export default function UploadVideo() {
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | undefined>(undefined);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<FrameAnalysis[]>([]);
  const [currentResult, setCurrentResult] = useState<EmotionResult | null>(null);
  const [selectedModel, setSelectedModel] = useState("fer2013");
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [isMuted, setIsMuted] = useState(true);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    api.getModels()
      .then(setModels)
      .catch(error => {
        console.error('Failed to fetch models:', error);
        setError('Failed to connect to the API. Please try again later.');
      });
  }, []);

  useEffect(() => {
    if (!videoRef.current || analysisResults.length === 0) return;

    const updateCurrentResult = () => {
      const currentTime = videoRef.current?.currentTime || 0;
      const result = analysisResults.find((r, i) => {
        const nextTime = i < analysisResults.length - 1 ? analysisResults[i + 1].timestamp : Infinity;
        return currentTime >= r.timestamp && currentTime < nextTime;
      });
      setCurrentResult(result?.result || null);
    };

    const video = videoRef.current;
    video.addEventListener('timeupdate', updateCurrentResult);
    return () => video.removeEventListener('timeupdate', updateCurrentResult);
  }, [analysisResults]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        handleVideoUpload(file);
      } else {
        setError("Please upload a valid video file");
      }
    }
  }, []);

  const handleVideoUpload = (file: File) => {
    console.log('Selected video file:', file);
    console.log('File name:', file.name);
    console.log('File size:', file.size);
    console.log('File type:', file.type);
    
    if (file.size > 50 * 1024 * 1024) {
      setError("Video file is too large. Maximum size is 50MB.");
      return;
    }
    setUploadedVideo(file);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setAnalysisResults([]);
    setCurrentResult(null);
    setError(null);
    setIsPlaying(false);
    setIsMuted(true);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleVideoUpload(e.target.files[0]);
    }
  };

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const resetVideo = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  };

  const captureFrame = async (timestamp: number): Promise<Blob | null> => {
    if (!videoRef.current || !canvasRef.current) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    return new Promise(resolve => {
      canvas.toBlob(blob => {
        resolve(blob);
      }, 'image/jpeg', 0.95);
    });
  };

  const analyzeFrame = async (timestamp: number): Promise<EmotionResult | null> => {
    const frame = await captureFrame(timestamp);
    if (!frame) return null;

    try {
      const response = await api.analyzeImage(frame, selectedModel);
      
      
      if (response.faces && response.faces.length > 0) {
        const firstFace = response.faces[0];
        return {
          label: firstFace.label,
          confidence: firstFace.confidence,
          model: selectedModel,
          processing_time: response.processing_time,
          all_predictions: firstFace.all_predictions
        };
      }
      
      return null;
    } catch (error) {
      console.error('Error analyzing frame:', error);
      return null;
    }
  };

  const analyzeVideo = async () => {
    if (!uploadedVideo || !videoRef.current) return;
    
    setIsAnalyzing(true);
    setAnalysisResults([]);
    setError("");
    
    const video = videoRef.current;
    const duration = video.duration;
    const frameInterval = 1;
    const totalFrames = Math.floor(duration / frameInterval);
    let analyzedFrames = 0;
    
    try {
      const results: FrameAnalysis[] = [];
      
      for (let time = 0; time < duration; time += frameInterval) {
        video.currentTime = time;
        await new Promise(resolve => {
          const handleSeeked = () => {
            video.removeEventListener('seeked', handleSeeked);
            resolve(null);
          };
          video.addEventListener('seeked', handleSeeked);
        });
        
        const result = await analyzeFrame(time);
        if (result) {
          results.push({ timestamp: time, result });
        }
        
        analyzedFrames++;
        setProgress((analyzedFrames / totalFrames) * 100);
      }
      
      setAnalysisResults(results);
      if (results.length === 0) {
        setError("No emotions were detected in the video");
      }
    } catch (error) {
      console.error('Error analyzing video:', error);
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError("Failed to analyze video. Please try again.");
      }
    } finally {
      setIsAnalyzing(false);
      setProgress(0);
      video.currentTime = 0;
    }
  };

  const downloadResults = () => {
    if (analysisResults.length === 0) return;

    const data = {
      video_name: uploadedVideo?.name,
      model: selectedModel,
      analysis_timestamp: new Date().toISOString(),
      frames: analysisResults
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `emotion_analysis_${uploadedVideo?.name || 'video'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !videoRef.current.muted;
      setIsMuted(!isMuted);
    }
  };


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
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-gray-900 mb-3 sm:mb-4">Video Emotion Analysis</h1>
            <p className="text-base sm:text-lg md:text-xl text-gray-600 mb-4 sm:mb-6">Upload a video to detect emotions frame by frame using AI analysis</p>
            <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-0 text-xs sm:text-sm text-gray-500 mb-6 sm:mb-8">
              <span>By Adepeju Peace Orefejo</span>
              <span className="hidden sm:inline mx-2">•</span>
              <span>MSc. Software Engineering 2025</span>
            </div>
          </header>

          <section className="mb-8 sm:mb-12">
            <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-4 sm:mb-8">Upload & Analyze Video</h2>
            <p className="text-sm sm:text-base text-gray-700 mb-6 sm:mb-8">
              Upload a video file to analyze emotions across multiple frames. The system will process the video frame by frame, detecting emotions and providing comprehensive analysis results with timeline visualization.
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
            <h3 className="text-xl font-semibold mb-6 text-gray-900">Upload Video</h3>
            
              
            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                dragActive 
                  ? 'border-blue bg-blue-50 scale-105' 
                  : 'border-gray-300 hover:border-blue hover:bg-blue-50'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-lg font-medium text-gray-700 mb-2">
                Drop your video here, or click to browse
              </p>
              <p className="text-sm text-gray-500 mb-4">
                Supports MP4, AVI, MOV files up to 50MB
              </p>
              <Button 
                onClick={() => fileInputRef.current?.click()}
                className="bg-blue text-white hover:bg-blue/90"
              >
                Choose Video File
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>

              
            <div className="mt-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select AI Model
              </label>
              <select
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

              
            {uploadedVideo && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-sm text-green-800">
                  <strong>File:</strong> {uploadedVideo.name}
                </p>
                <p className="text-sm text-green-800">
                  <strong>Size:</strong> {(uploadedVideo.size / (1024 * 1024)).toFixed(2)} MB
                </p>
                <p className="text-sm text-green-800">
                  <strong>Type:</strong> {uploadedVideo.type}
                </p>
              </div>
            )}

            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}

              
            {analysisResults.length > 0 && (
              <div className="mt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Analysis Results</h3>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={downloadResults}
                    className="flex items-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download JSON
                  </Button>
                </div>
                <div className="p-4 bg-white border rounded-lg">
                  <p className="text-sm text-gray-600 mb-2">
                    <strong>Frames Analyzed:</strong> {analysisResults.length}
                  </p>
                  <p className="text-sm text-gray-600">
                    <strong>Duration:</strong> {videoRef.current?.duration.toFixed(1)}s
                  </p>
                </div>
              </div>
            )}
          </Card>

          
          <Card className="p-8">
            <h2 className="text-xl font-semibold mb-6">Video Preview</h2>
            
            <div className="space-y-4">
              <div className="relative bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  src={videoUrl}
                  className="w-full h-64 object-contain"
                  muted={isMuted}
                  onEnded={() => setIsPlaying(false)}
                />
                
                  
                {currentResult && (
                  <div className="absolute top-2 right-2 bg-black/75 text-white p-4 rounded-lg">
                    <p className="font-semibold">{currentResult.label}</p>
                    <p className="text-sm opacity-75">
                      Confidence: {(currentResult.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                )}

                  
                {isAnalyzing && (
                  <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                    <div className="text-center text-white">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
                      <p>Analyzing Video: {progress.toFixed(1)}%</p>
                    </div>
                  </div>
                )}
              </div>
              
              
              <canvas ref={canvasRef} className="hidden" />
              
              
              <div className="flex items-center justify-center gap-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={togglePlayPause}
                  disabled={!videoUrl || isAnalyzing}
                  className="flex items-center gap-2"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isPlaying ? 'Pause' : 'Play'}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={resetVideo}
                  disabled={!videoUrl || isAnalyzing}
                  className="flex items-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </Button>
              </div>

                
              <Button
                onClick={analyzeVideo}
                disabled={!videoUrl || isAnalyzing}
                className="w-full bg-gradient-to-r from-blue to-purple-600 text-white hover:from-blue-600 hover:to-purple-700 disabled:opacity-50"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Analyzing Video...
                  </>
                ) : (
                  'Analyze Emotions'
                )}
              </Button>

                
              {currentResult && (
                <div className="mt-8 p-4 bg-white border rounded-lg">
                  <h3 className="font-semibold text-gray-800 mb-4">Current Frame Analysis</h3>
                  <div className="space-y-3">
                    {Object.entries(currentResult.all_predictions)
                      .sort(([, a], [, b]) => b - a)
                      .map(([emotion, confidence]) => (
                        <div key={emotion} className="flex items-center justify-between">
                          <span className="capitalize font-medium">{emotion}</span>
                          <div className="flex items-center gap-3">
                            <div className="w-32 bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-blue h-2 rounded-full transition-all duration-500"
                                style={{ width: `${confidence * 100}%` }}
                              ></div>
                            </div>
                            <span className="text-sm font-medium w-12 text-right">
                              {(confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-4 text-center">
                    Processing time: {currentResult.processing_time.toFixed(3)}s
                  </p>
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