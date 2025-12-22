import { Link } from "react-router-dom";

export default function Research() {
  return (
    <div className="min-h-screen bg-white py-24">
      <div className="max-w-6xl mx-auto">
        <article className="prose prose-lg prose-gray max-w-none">
          <header className="mb-12">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">Research Methodology & Findings</h1>
            <p className="text-xl text-gray-600 mb-6">Empirical comparison of convolutional neural networks for facial emotion recognition</p>
            <div className="flex items-center text-sm text-gray-500 mb-8">
              <span>By Adepeju Peace Orefejo</span>
              <span className="mx-2">•</span>
              <span>MSc. Software Engineering 2025</span>
        </div>
          </header>

          <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">Abstract</h2>
            <p className="text-gray-700 mb-6">
              This research presents an empirical comparison of convolutional neural network architectures for facial emotion recognition across three benchmark datasets: FER2013, CK+, and AffectNet. The study evaluates model performance, architectural differences, and practical implications for autism support applications.
            </p>
            <p className="text-gray-700 mb-6">
              Our findings demonstrate significant performance variations across datasets, with CK+ achieving perfect accuracy (100.00%) in controlled conditions, AffectNet reaching 62.51% in real-world scenarios, and FER2013 achieving 54.67% test accuracy. The research provides insights into the challenges of emotion recognition across different data conditions and the potential for AI-assisted support tools.
            </p>
          </section>
        
        <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">Datasets</h2>
            <p className="text-gray-700 mb-6">
              Three publicly available datasets were selected to evaluate model performance across different data characteristics and emotion categories.
            </p>
            
            <div className="space-y-8">
              <div className="border-l-4 border-blue-500 pl-6 py-4 bg-blue-50">
                <h3 className="text-xl font-semibold text-gray-900 mb-3">FER2013 Dataset</h3>
                <div className="grid md:grid-cols-2 gap-6">
              <div>
                    <p className="text-gray-700 mb-4">
                      The Facial Expression Recognition 2013 dataset contains 35,887 grayscale images of faces displaying seven basic emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.
                    </p>
                    <ul className="space-y-2 text-sm text-gray-600">
                      <li>• <strong>Training set:</strong> 28,709 images</li>
                      <li>• <strong>Test set:</strong> 7,178 images</li>
                      <li>• <strong>Image size:</strong> 48×48 pixels</li>
                      <li>• <strong>Emotions:</strong> 7 classes</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-4 border">
                    <h4 className="font-semibold text-gray-800 mb-2">Performance Results</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Test Accuracy:</span>
                        <span className="font-semibold text-blue-600">54.67%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Test Loss:</span>
                        <span className="font-semibold text-gray-800">1.2739</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-l-4 border-purple-500 pl-6 py-4 bg-purple-50">
                <h3 className="text-xl font-semibold text-gray-900 mb-3">AffectNet Dataset</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-gray-700 mb-4">
                      AffectNet is a large-scale dataset containing over 1 million facial images from the internet, annotated with eight emotion categories including contempt.
                    </p>
                    <ul className="space-y-2 text-sm text-gray-600">
                      <li>• <strong>Training set:</strong> 16,108 images</li>
                      <li>• <strong>Test set:</strong> 14,518 images</li>
                      <li>• <strong>Image size:</strong> 48×48 pixels</li>
                      <li>• <strong>Emotions:</strong> 8 classes</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-4 border">
                    <h4 className="font-semibold text-gray-800 mb-2">Performance Results</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Test Accuracy:</span>
                        <span className="font-semibold text-purple-600">62.51%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Test Loss:</span>
                        <span className="font-semibold text-gray-800">2.6024</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-l-4 border-emerald-500 pl-6 py-4 bg-emerald-50">
                <h3 className="text-xl font-semibold text-gray-900 mb-3">CK+ Dataset</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-gray-700 mb-4">
                      The Extended Cohn-Kanade dataset contains 593 image sequences from 123 subjects, displaying facial expressions across six basic emotions plus contempt. This dataset features high-quality, controlled laboratory conditions.
                    </p>
                    <ul className="space-y-2 text-sm text-gray-600">
                      <li>• <strong>Training set:</strong> 593 sequences</li>
                      <li>• <strong>Test set:</strong> 123 subjects</li>
                      <li>• <strong>Image size:</strong> 48×48 pixels</li>
                      <li>• <strong>Emotions:</strong> 7 classes</li>
                    </ul>
                  </div>
                  <div className="bg-white rounded-lg p-4 border">
                    <h4 className="font-semibold text-gray-800 mb-2">Performance Results</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Test Accuracy:</span>
                        <span className="font-semibold text-emerald-600">100.00%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Test Loss:</span>
                        <span className="font-semibold text-gray-800">0.1121</span>
                      </div>
                    </div>
                </div>
              </div>
            </div>
          </div>
        </section>

          <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">Methodology</h2>
            
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Model Architecture</h3>
            <p className="text-gray-700 mb-8">
              Convolutional Neural Networks (CNNs) were implemented using TensorFlow/Keras for each dataset. The architectures consisted of multiple convolutional layers with ReLU activation functions, followed by max pooling layers for dimensionality reduction. Batch normalization was applied to improve training stability, and dropout layers were included to prevent overfitting. The final layers included a global average pooling layer and dense layers with softmax activation for emotion classification.
            </p>

            <h3 className="text-xl font-semibold text-gray-900 mb-6">Training Process</h3>
            <div className="space-y-6">
              <div className="flex items-start">
                <span className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-semibold mr-4 mt-1">1</span>
              <div>
                  <h4 className="font-semibold text-gray-800 mb-2">Data Preprocessing</h4>
                  <p className="text-gray-700 text-sm">Images were resized to 48×48 pixels, normalized, and augmented with rotation, zoom, and horizontal flip to increase dataset diversity.</p>
                  </div>
                  </div>
              
              <div className="flex items-start">
                <span className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-semibold mr-4 mt-1">2</span>
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2">Model Training</h4>
                  <p className="text-gray-700 text-sm">Models were trained using categorical cross-entropy loss, Adam optimizer, and early stopping to prevent overfitting.</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <span className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-semibold mr-4 mt-1">3</span>
                <div>
                  <h4 className="font-semibold text-gray-800 mb-2">Evaluation</h4>
                  <p className="text-gray-700 text-sm">Performance was measured using test accuracy, loss, and confusion matrices to assess classification quality.</p>
              </div>
            </div>
          </div>
        </section>
        
        <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">Results & Analysis</h2>
            
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Performance Comparison</h3>
            <div className="overflow-x-auto mb-12">
              <table className="w-full border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold text-gray-800">Dataset</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold text-gray-800">Test Accuracy</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold text-gray-800">Test Loss</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold text-gray-800">Training Images</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold text-gray-800">Test Images</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">FER2013</td>
                    <td className="border border-gray-300 px-4 py-3 text-blue-600 font-semibold">54.67%</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">1.2739</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">28,709</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">7,178</td>
                  </tr>
                  <tr className="bg-gray-50">
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">AffectNet</td>
                    <td className="border border-gray-300 px-4 py-3 text-purple-600 font-semibold">62.51%</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">2.6024</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">16,108</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">14,518</td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">CK+</td>
                    <td className="border border-gray-300 px-4 py-3 text-emerald-600 font-semibold">100.00%</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">0.1121</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">593</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">123</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h3 className="text-xl font-semibold text-gray-900 mb-6">Key Findings</h3>
            <div className="space-y-8">
              <div className="border-l-4 border-green-500 pl-6 py-4 bg-green-50">
                <h4 className="font-semibold text-gray-800 mb-2">Dataset Characteristics Impact Performance</h4>
                <p className="text-gray-700 text-sm">
                  CK+ achieved perfect accuracy (100.00%) due to its controlled laboratory conditions and high-quality images. AffectNet (62.51%) outperformed FER2013 (54.67%) likely due to its larger training set and additional emotion category (contempt).
                </p>
              </div>
              
              <div className="border-l-4 border-orange-500 pl-6 py-4 bg-orange-50">
                <h4 className="font-semibold text-gray-800 mb-2">Controlled vs. Real-World Conditions</h4>
                <p className="text-gray-700 text-sm">
                  The significant performance difference between CK+ (100%) and real-world datasets (54-62%) highlights the challenge of emotion recognition in uncontrolled conditions. Factors such as lighting, pose, and individual expression variations significantly impact accuracy.
                </p>
              </div>
              
              <div className="border-l-4 border-blue-500 pl-6 py-4 bg-blue-50">
                <h4 className="font-semibold text-gray-800 mb-2">Dataset Size and Quality Trade-offs</h4>
                <p className="text-gray-700 text-sm">
                  CK+ demonstrates that high-quality, controlled data can achieve excellent performance despite smaller dataset size. However, real-world applications require models trained on diverse, uncontrolled data like FER2013 and AffectNet.
                </p>
              </div>
            </div>

            <h3 className="text-xl font-semibold text-gray-900 mb-6">Visual Analysis</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <img
                  src="/src/assets/results/fer2013/confusion_matrix.png"
                  alt="FER2013 Confusion Matrix"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">FER2013 Confusion Matrix</p>
                <p className="text-xs text-gray-500 mt-1">54.67% accuracy</p>
              </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/affectnet/confusion_matrix.png"
                  alt="AffectNet Confusion Matrix"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">AffectNet Confusion Matrix</p>
                <p className="text-xs text-gray-500 mt-1">62.51% accuracy</p>
              </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/ckplus/confusion_matrix.png"
                  alt="CK+ Confusion Matrix"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">CK+ Confusion Matrix</p>
                <p className="text-xs text-gray-500 mt-1">100.00% accuracy</p>
              </div>
                </div>
          </section>

          <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">Training Analysis</h2>
            
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Training History</h3>
            <p className="text-gray-700 mb-6">
              The training process for each model was monitored to understand convergence patterns and identify potential overfitting. The following visualizations show the training progression across different datasets.
            </p>
            
            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <div className="text-center">
                <img
                  src="/src/assets/results/fer2013/training_history.png"
                  alt="FER2013 Training History"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">FER2013 Training History</p>
                <p className="text-xs text-gray-500 mt-1">Loss and accuracy over epochs</p>
              </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/affectnet/training_history.png"
                  alt="AffectNet Training History"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">AffectNet Training History</p>
                <p className="text-xs text-gray-500 mt-1">Loss and accuracy over epochs</p>
                </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/ckplus/training_history.png"
                  alt="CK+ Training History"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">CK+ Training History</p>
                <p className="text-xs text-gray-500 mt-1">Loss and accuracy over epochs</p>
              </div>
            </div>

            <h3 className="text-xl font-semibold text-gray-900 mb-6">Per-Class Performance</h3>
            <p className="text-gray-700 mb-6">
              Understanding which emotions are easier or harder to recognize provides insights into the model's strengths and weaknesses. The following charts show per-class accuracy for each dataset.
            </p>
            
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <img
                  src="/src/assets/results/fer2013/per_class_accuracy.png"
                  alt="FER2013 Per-Class Accuracy"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">FER2013 Per-Class Accuracy</p>
              </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/affectnet/per_class_accuracy.png"
                  alt="AffectNet Per-Class Accuracy"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">AffectNet Per-Class Accuracy</p>
              </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/ckplus/per_class_accuracy.png"
                  alt="CK+ Per-Class Accuracy"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">CK+ Per-Class Accuracy</p>
            </div>
          </div>
        </section>

          <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">Feature Analysis</h2>
            
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Convolutional Feature Maps</h3>
            <p className="text-gray-700 mb-6">
              Feature maps from convolutional layers reveal what patterns the models learn to recognize. These visualizations show how the neural networks process facial features to identify emotions.
            </p>
            
            <div className="space-y-12">
              <div>
                <h4 className="text-lg font-semibold text-gray-800 mb-4">FER2013 Feature Maps</h4>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="text-center">
                    <img
                      src="/src/assets/results/fer2013/feature_maps_conv2d.png"
                      alt="FER2013 Feature Maps Layer 1"
                      className="w-full rounded-lg border shadow-sm mb-3"
                    />
                    <p className="text-sm text-gray-600">First Convolutional Layer</p>
                  </div>
                  <div className="text-center">
                    <img
                      src="/src/assets/results/fer2013/feature_maps_conv2d_1.png"
                      alt="FER2013 Feature Maps Layer 2"
                      className="w-full rounded-lg border shadow-sm mb-3"
                    />
                    <p className="text-sm text-gray-600">Second Convolutional Layer</p>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="text-lg font-semibold text-gray-800 mb-4">AffectNet Feature Maps</h4>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="text-center">
                    <img
                      src="/src/assets/results/affectnet/feature_maps_conv2d.png"
                      alt="AffectNet Feature Maps Layer 1"
                      className="w-full rounded-lg border shadow-sm mb-3"
                    />
                    <p className="text-sm text-gray-600">First Convolutional Layer</p>
                  </div>
                  <div className="text-center">
                    <img
                      src="/src/assets/results/affectnet/feature_maps_conv2d_1.png"
                      alt="AffectNet Feature Maps Layer 2"
                      className="w-full rounded-lg border shadow-sm mb-3"
                    />
                    <p className="text-sm text-gray-600">Second Convolutional Layer</p>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="text-lg font-semibold text-gray-800 mb-4">CK+ Feature Maps</h4>
                <div className="grid md:grid-cols-2 gap-6">
                <div className="text-center">
                    <img
                      src="/src/assets/results/ckplus/feature_maps_conv2d.png"
                      alt="CK+ Feature Maps Layer 1"
                      className="w-full rounded-lg border shadow-sm mb-3"
                    />
                    <p className="text-sm text-gray-600">First Convolutional Layer</p>
                  </div>
                  <div className="text-center">
                    <img
                      src="/src/assets/results/ckplus/feature_maps_conv2d_1.png"
                      alt="CK+ Feature Maps Layer 2"
                      className="w-full rounded-lg border shadow-sm mb-3"
                    />
                    <p className="text-sm text-gray-600">Second Convolutional Layer</p>
                </div>
              </div>
            </div>
          </div>
        </section>

          <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">Sample Predictions</h2>
            
            <p className="text-gray-700 mb-6">
              Sample predictions demonstrate how the models perform on individual test cases. These examples show both successful classifications and common misclassification patterns.
            </p>
            
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <img
                  src="/src/assets/results/fer2013/sample_predictions.png"
                  alt="FER2013 Sample Predictions"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">FER2013 Sample Predictions</p>
                <p className="text-xs text-gray-500 mt-1">Predicted vs. actual emotions</p>
              </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/affectnet/sample_predictions.png"
                  alt="AffectNet Sample Predictions"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">AffectNet Sample Predictions</p>
                <p className="text-xs text-gray-500 mt-1">Predicted vs. actual emotions</p>
              </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/ckplus/sample_predictions.png"
                  alt="CK+ Sample Predictions"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">CK+ Sample Predictions</p>
                <p className="text-xs text-gray-500 mt-1">Predicted vs. actual emotions</p>
            </div>
          </div>
        </section>

          <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">Technical Implementation</h2>
            
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Preprocessing Pipeline</h3>
            <p className="text-gray-700 mb-6">
              Consistent preprocessing was applied across all datasets to ensure fair comparison. The pipeline included normalization, augmentation, and standardization steps.
            </p>
            
            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <div className="text-center">
                <img
                  src="/src/assets/results/preprocessing_examples/fer2013_happiness_preprocessing.png"
                  alt="FER2013 Preprocessing Example"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">FER2013 Preprocessing</p>
                <p className="text-xs text-gray-500 mt-1">Happiness emotion example</p>
                </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/preprocessing_examples/affectnet_happiness_preprocessing.png"
                  alt="AffectNet Preprocessing Example"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">AffectNet Preprocessing</p>
                <p className="text-xs text-gray-500 mt-1">Happiness emotion example</p>
              </div>
              <div className="text-center">
                <img
                  src="/src/assets/results/preprocessing_examples/ck+_happiness_preprocessing.png"
                  alt="CK+ Preprocessing Example"
                  className="w-full rounded-lg border shadow-sm mb-3"
                />
                <p className="text-sm text-gray-600">CK+ Preprocessing</p>
                <p className="text-xs text-gray-500 mt-1">Happiness emotion example</p>
              </div>
            </div>

            <h3 className="text-xl font-semibold text-gray-900 mb-6">Hardware and Software</h3>
            <div className="bg-gray-50 rounded-lg p-6 mb-8">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-gray-800 mb-3">Hardware Configuration</h4>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li>• <strong>Processor:</strong> Apple M1 Chip</li>
                    <li>• <strong>Memory:</strong> 16GB Unified Memory</li>
                    <li>• <strong>GPU:</strong> Apple M1 GPU with Metal Performance Shaders</li>
                    <li>• <strong>Storage:</strong> SSD for fast data access</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-gray-800 mb-3">Software Stack</h4>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li>• <strong>Framework:</strong> TensorFlow 2.16.2</li>
                    <li>• <strong>Acceleration:</strong> TensorFlow Metal for GPU</li>
                    <li>• <strong>Language:</strong> Python 3.11</li>
                    <li>• <strong>Libraries:</strong> Keras, NumPy, OpenCV</li>
                  </ul>
                </div>
              </div>
            </div>

            <h3 className="text-xl font-semibold text-gray-900 mb-6">Training Parameters</h3>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold text-gray-800">Parameter</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold text-gray-800">FER2013</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold text-gray-800">AffectNet</th>
                    <th className="border border-gray-300 px-4 py-3 text-left font-semibold text-gray-800">CK+</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">Batch Size</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">32</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">32</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">16</td>
                  </tr>
                  <tr className="bg-gray-50">
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">Epochs</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">50</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">50</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">30</td>
                  </tr>
                  <tr>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">Learning Rate</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">0.001</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">0.001</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">0.001</td>
                  </tr>
                  <tr className="bg-gray-50">
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">Optimizer</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">Adam</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">Adam</td>
                    <td className="border border-gray-300 px-4 py-3 text-gray-700">Adam</td>
                  </tr>
                </tbody>
              </table>
          </div>
        </section>

          <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">Limitations & Future Work</h2>
            
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Current Limitations</h3>
            <ul className="space-y-4 mb-8">
              <li className="flex items-start">
                <span className="text-gray-400 mr-3 mt-1">•</span>
                <span className="text-gray-700">Real-world datasets (FER2013: 54.67%, AffectNet: 62.51%) show moderate accuracy levels, indicating room for improvement in uncontrolled conditions</span>
              </li>
                  <li className="flex items-start">
                <span className="text-gray-400 mr-3 mt-1">•</span>
                <span className="text-gray-700">CK+ perfect accuracy (100%) achieved in controlled laboratory conditions may not generalize to real-world applications</span>
                  </li>
                  <li className="flex items-start">
                <span className="text-gray-400 mr-3 mt-1">•</span>
                <span className="text-gray-700">Models trained on general populations may not account for neurodivergent expression patterns</span>
                  </li>
                  <li className="flex items-start">
                <span className="text-gray-400 mr-3 mt-1">•</span>
                <span className="text-gray-700">No evaluation of cultural or demographic bias in emotion recognition</span>
                  </li>
                </ul>

            <h3 className="text-xl font-semibold text-gray-900 mb-6">Future Research Directions</h3>
                <ul className="space-y-4">
                  <li className="flex items-start">
                <span className="text-gray-400 mr-3 mt-1">•</span>
                <span className="text-gray-700">Investigation of transfer learning and pre-trained models for improved accuracy</span>
              </li>
              <li className="flex items-start">
                <span className="text-gray-400 mr-3 mt-1">•</span>
                <span className="text-gray-700">Development of autism-specific training datasets and evaluation metrics</span>
                  </li>
                  <li className="flex items-start">
                <span className="text-gray-400 mr-3 mt-1">•</span>
                <span className="text-gray-700">Integration of temporal information for dynamic emotion recognition</span>
                  </li>
                  <li className="flex items-start">
                <span className="text-gray-400 mr-3 mt-1">•</span>
                <span className="text-gray-700">User studies with autistic individuals to validate practical utility</span>
                  </li>
                </ul>
          </section>

          <section className="mb-20">
            <h2 className="text-2xl font-bold text-gray-900 mb-8">References</h2>
            <div className="space-y-6 text-sm text-gray-700">
              <p>
                <strong>Goodfellow, I. J., et al.</strong> (2013). Challenges in representation learning: A report on three machine learning contests. <em>Neural Networks</em>, 64, 59-63.
              </p>
              <p>
                <strong>Lucey, P., et al.</strong> (2010). The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression. <em>2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition-Workshops</em>.
              </p>
              <p>
                <strong>Mollahosseini, A., et al.</strong> (2017). AffectNet: A database for facial expression, valence, and arousal computing in the wild. <em>IEEE Transactions on Affective Computing</em>, 10(1), 18-31.
              </p>
              <p>
                <strong>Ekman, P., & Friesen, W. V.</strong> (1971). Constants across cultures in the face and emotion. <em>Journal of Personality and Social Psychology</em>, 17(2), 124-129.
              </p>
            </div>
          </section>

          <div className="border-t pt-12 mt-16">
            <div className="flex flex-col sm:flex-row gap-4 justify-between items-center">
              <div className="text-sm text-gray-500">
                <p>Research conducted as part of MSc. Software Engineering program</p>
                <p>Big Academy UAE in partnership with Euclea Business School</p>
              </div>
              <div className="flex gap-4">
                <Link to="/learn" className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                  Learn about Autism Support →
                </Link>
                <Link to="/upload-photo" className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                  Try Emotion Detection →
                </Link>
              </div>
            </div>
          </div>
        </article>
      </div>
    </div>
  );
} 