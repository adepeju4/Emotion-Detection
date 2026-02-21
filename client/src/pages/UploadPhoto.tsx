import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, Upload } from "lucide-react";
import { useState, useRef, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import type { ModelInfo } from "@/types/emotion";
import type { AnalysisResponse, FaceResult } from "@/lib/api";

export default function UploadPhoto() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResponse | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("affectnet");
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string>("");
  const [models, setModels] = useState<Record<string, ModelInfo>>({});
  const [, setAnalysisHistory] = useState<AnalysisResponse[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getModels()
      .then(setModels)
      .catch(error => {
        console.error('Failed to fetch models:', error);
        setError('Failed to connect to the API. Please try again later.');
      });
  }, []);

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  const handleImageUpload = useCallback((file: File) => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    const isValidType = allowedTypes.includes(file.type) || 
                        fileExtension === 'jpg' || 
                        fileExtension === 'jpeg' || 
                        fileExtension === 'png';
    
    if (!isValidType) {
      setError("Only JPG and PNG image files are supported. Please upload a JPG or PNG file.");
      setSelectedImage(null);
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
      setImageUrl("");
      return;
    }
    
    if (file.size > 15 * 1024 * 1024) {
      setError("Image file is too large. Maximum size is 15MB.");
      setSelectedImage(null);
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
      setImageUrl("");
      return;
    }
    
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    
    setSelectedImage(file);
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setAnalysisResults(null);
    setError("");
  }, [imageUrl]);

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
      handleImageUpload(file);
    }
  }, [handleImageUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleImageUpload(e.target.files[0]);
    }
  }, [handleImageUpload]);


  const analyzeImage = useCallback(async () => {
    if (!selectedImage) return;
    
    setIsAnalyzing(true);
    setError("");
    
    try {
      const result = await api.analyzeImage(selectedImage, selectedModel);
      setAnalysisResults(result);
      setAnalysisHistory(prev => [...prev, result]);
    } catch (error) {
      console.error('Error analyzing image:', error);
      if (error instanceof Error) {
        setError(error.message);
      } else {
        setError("Failed to analyze image. Please try again.");
      }
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedImage, selectedModel]);


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
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-gray-900 mb-3 sm:mb-4">Photo Emotion Analysis</h1>
            <p className="text-base sm:text-lg md:text-xl text-gray-600 mb-4 sm:mb-6">Upload a photo to detect emotions using AI-powered facial analysis</p>
            <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-0 text-xs sm:text-sm text-gray-500 mb-6 sm:mb-8">
              <span>By Adepeju Peace Orefejo</span>
              <span className="hidden sm:inline mx-2">•</span>
              <span>MSc. Software Engineering 2025</span>
            </div>
          </header>

          <section className="mb-8 sm:mb-12">
            <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-4 sm:mb-8">Upload & Analyze Photo</h2>
            <p className="text-sm sm:text-base text-gray-700 mb-6 sm:mb-8">
              Upload a photo to analyze emotions using state-of-the-art deep learning models. The system will automatically detect faces and provide detailed emotion analysis with confidence scores.
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

          <div className="grid lg:grid-cols-2 gap-6 sm:gap-8">
            <Card className="p-4 sm:p-6 md:p-8 border border-gray-200 shadow-sm">
              <h3 className="text-lg sm:text-xl font-semibold mb-4 sm:mb-6 text-gray-900">Upload Photo</h3>
            

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300" />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-white text-gray-500">Or</span>
              </div>
            </div>

            <div className="mt-6 sm:mt-8">
              <div
                className={`border-2 border-dashed rounded-xl p-4 sm:p-6 md:p-8 text-center transition-all duration-300 ${
                  dragActive 
                    ? 'border-blue bg-blue-50 scale-105' 
                    : 'border-gray-300 hover:border-blue hover:bg-blue-50'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <Upload className="w-8 h-8 sm:w-10 sm:h-10 md:w-12 md:h-12 text-gray-400 mx-auto mb-3 sm:mb-4" />
                <p className="text-sm sm:text-base md:text-lg font-medium text-gray-700 mb-2">
                  Drop your photo here, or click to browse
                </p>
                <p className="text-xs sm:text-sm text-gray-500 mb-3 sm:mb-4">
                  Supports JPG, PNG files up to 15MB
                </p>
                <Button 
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-blue text-white hover:bg-blue/90"
                >
                  Choose Photo
                </Button>
                  <input
                    title="Choose Photo"
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/jpg,image/png,.jpg,.jpeg,.png"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
            </div>

            <div className="mt-6">
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

            {selectedImage && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-sm text-green-800">
                  <strong>File:</strong> {selectedImage.name}
                </p>
                <p className="text-sm text-green-800">
                  <strong>Size:</strong> {(selectedImage.size / (1024 * 1024)).toFixed(2)} MB
                </p>
                <p className="text-sm text-green-800">
                  <strong>Type:</strong> {selectedImage.type}
                </p>
              </div>
            )}

            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}
          </Card>

            <Card className="p-4 sm:p-6 md:p-8 border border-gray-200 shadow-sm">
              <h3 className="text-lg sm:text-xl font-semibold mb-4 sm:mb-6 text-gray-900">Image Preview & Analysis</h3>
            
            {imageUrl ? (
              <div className="space-y-3 sm:space-y-4">
                <div className="relative bg-black rounded-lg overflow-hidden">
                  <img
                    src={imageUrl}
                    alt="Preview"
                    className="w-full h-48 sm:h-56 md:h-64 object-contain"
                      onError={() => {
                      setError("Failed to load image preview. The file may be corrupted or in an unsupported format.");
                      setImageUrl("");
                      setSelectedImage(null);
                    }}
                  />
                  
                  {/* {analysisResults && analysisResults.faces && analysisResults.faces.map((face: any) => (
                    <div
                      key={face.face_id}
                      className="absolute border-2 border-green-500 bg-green-500/20"
                      style={{
                        left: `${(face.bbox.x / 640) * 100}%`,
                        top: `${(face.bbox.y / 480) * 100}%`,
                        width: `${(face.bbox.width / 640) * 100}%`,
                        height: `${(face.bbox.height / 480) * 100}%`,
                      }}
                    >
                      <div className="absolute -top-6 left-0 bg-green-500 text-white px-2 py-1 text-xs rounded">
                        Face {face.face_id}: {face.label} ({(face.confidence * 100).toFixed(1)}%)
                      </div>
                    </div>
                  ))} */}
                  {analysisResults && (
                    <div className="absolute top-2 right-2 bg-black/75 text-white p-4 rounded-lg">
                      <p className="font-semibold">Faces Detected: {analysisResults.faces_detected}</p>
                      <p className="text-sm opacity-75">
                        Processing time: {analysisResults.processing_time.toFixed(3)}s
                      </p>
                    </div>
                  )}
                </div>
                <Button
                  onClick={analyzeImage}
                  disabled={isAnalyzing}
                  className="w-full bg-gradient-to-r from-blue to-purple-600 text-white hover:from-blue-600 hover:to-purple-700 disabled:opacity-50"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Analyzing Image...
                    </>
                  ) : (
                    'Analyze Emotions'
                  )}
                </Button>
                {analysisResults && (
                  <div className="mt-8 space-y-6">
                    <h4 className="text-lg font-semibold text-gray-900 mb-4">Analysis Results</h4>
                      {analysisResults.faces && analysisResults.faces.map((face: FaceResult) => (
                      <div key={face.face_id} className="p-6 bg-gray-50 border border-gray-200 rounded-lg">
                        <div className="flex items-start gap-6 mb-6">
                          <div className="flex-shrink-0">
                            <h5 className="font-semibold mb-3 text-gray-900">Face {face.face_id}: {face.label}</h5>
                            {face.cropped_face && (
                              <div className="w-28 h-28 border-2 border-gray-300 rounded-lg overflow-hidden shadow-sm">
                                <img
                                  src={`data:image/png;base64,${face.cropped_face}`}
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
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 bg-gray-100 rounded-lg">
                <div className="text-center">
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No image uploaded yet</p>
                </div>
              </div>
            )}
          </Card>
          </div>
        </article>
      </div>
    </div>
  );
} 