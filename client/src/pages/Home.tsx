import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

export default function Home() {
  const EMOTION_CLASS: Record<string, string> = {
    anger: 'emotion-anger',
    happiness: 'emotion-happiness',
    sadness: 'emotion-sadness',
    fear: 'emotion-afraid',
    neutral: 'emotion-neutral',
    surprise: 'emotion-surprise',
    disgust: 'emotion-disgust',
    contempt: 'emotion-contempt'
  };
  return (
    <div className="min-h-screen bg-white text-gray-900">
      <header className="max-w-6xl mx-auto py-16 md:py-24">
        <h1 className="text-3xl md:text-5xl font-semibold leading-tight tracking-tight">
          EMOTION DETECTION FOR AUTISM SUPPORT
        </h1>
        <p className="mt-3 text-sm md:text-base text-gray-700">Adepeju Peace Orefejo</p>
        <p className="mt-1 text-sm text-gray-600">Big Academy UAE in partnership with Euclea Business School</p>
        <p className="mt-1 text-sm text-gray-600">MSc. Software Engineering 2025</p>
      </header>

      <nav className="sticky top-0 z-10 border-y bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80">
        <div className="max-w-6xl mx-auto">
          <ul className="flex gap-4 md:gap-6 py-2 md:py-3 text-sm overflow-x-auto whitespace-nowrap scrollbar-none">
            <li><a href="#abstract" className="hover:underline">Abstract</a></li>
            <li><a href="#datasets" className="hover:underline">Datasets</a></li>
            <li><a href="#methods" className="hover:underline">Methodology</a></li>
            <li><a href="#results" className="hover:underline">Results</a></li>
            <li><a href="#code" className="hover:underline">Code</a></li>
            <li><a href="#contact" className="hover:underline">Contact</a></li>
          </ul>
        </div>
      </nav>

      <section className="max-w-6xl mx-auto py-10 md:py-12">
        <div className="grid md:grid-cols-2 gap-10 items-start">
          <div>
            <p className="text-base md:text-lg leading-relaxed">
              This project presents an empirical comparison of convolutional baselines for facial emotion
              recognition across FER2013, CK+, and AffectNet. The site summarizes the problem space, datasets,
              baselines, and observed results without making performance claims or deployment promises.
            </p>
            <div className="mt-6 flex flex-col sm:flex-row gap-3">
              <a href="#code">
                <Button variant="outline" className="border-gray-300 text-gray-900 hover:bg-gray-100">View Code</Button>
              </a>
            </div>
          </div>
          <div className="border rounded-lg p-4">
            <div className="text-sm text-gray-600 mb-3">Emotion classes</div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs text-center">
              {[
                'Anger',
                'Happiness',
                'Sadness',
                'Fear',
                'Neutral',
                'Surprise',
                'Disgust',
                'Contempt',
                'Happiness'
              ].map((lbl, i) => {
                const key = String(lbl).toLowerCase();
                const cls = EMOTION_CLASS[key] ?? 'emotion-neutral';
                return (
                  <div key={i} className={`aspect-square border rounded flex items-center justify-center ${cls}`}>
                    {lbl}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      <section id="abstract" className="max-w-6xl mx-auto py-10 md:py-12 border-t">
        <h2 className="text-2xl font-semibold mb-4">ABSTRACT</h2>
        <p className="leading-relaxed text-gray-800">
          This work presents an empirical study of convolutional baselines for facial emotion recognition in
          natural interactions. Models are trained on FER2013, CK+, and AffectNet. The study prioritises external
          validity over benchmark optimisation and documents the impact of dataset noise, posed vs. spontaneous
          expression bias, and class imbalance on generalisation.
        </p>
      </section>

      <section id="datasets" className="max-w-6xl mx-auto py-10 md:py-12 border-t">
        <h2 className="text-2xl font-semibold mb-4">DATASETS</h2>
        <ul className="list-disc pl-6 space-y-2 text-gray-800">
          <li>FER2013 — 48×48 grayscale, in-the-wild; noisy labels and class imbalance.</li>
          <li>CK+ — posed apex expressions; high accuracy but limited external validity.</li>
          <li>AffectNet — large-scale web images; diverse but label noise present.</li>
        </ul>
      </section>

      <section id="methods" className="max-w-6xl mx-auto py-10 md:py-12 border-t">
        <h2 className="text-2xl font-semibold mb-6">METHODOLOGY</h2>
        <div className="space-y-6 text-gray-800">
          <p className="text-lg leading-relaxed">
            Baselines follow a controlled pipeline to enable like-for-like comparisons across FER2013, CK+, and
            AffectNet. Choices emphasise reproducibility over peak benchmark scores.
          </p>
          <ul className="list-disc pl-6 space-y-4">
            <li className="leading-relaxed">
              <span className="font-medium">Preprocessing:</span> face-cropped inputs resized to 48×48; grayscale
              conversion; pixel normalisation to [0,1]. No alignment beyond dataset-provided crops.
            </li>
            <li className="leading-relaxed">
              <span className="font-medium">Architectures:</span> Two CNN variants implemented:
              <ol className="list-decimal pl-6 mt-2 space-y-1">
                <li>Basic CNN with 4 conv blocks (32→64→128→256 filters), L2 regularization, and dropout</li>
                <li>Improved CNN with BatchNormalization and GlobalAveragePooling2D for better generalization</li>
              </ol>
            </li>
            <li className="leading-relaxed">
              <span className="font-medium">Training:</span> categorical cross-entropy loss; Adam optimiser (learning rate 0.001); ReduceLROnPlateau (factor 0.5, patience 5, min_lr 1e-6); early stopping (patience 5) for improved CNN only; 50 epochs with batch size 32.
            </li>
            <li className="leading-relaxed">
              <span className="font-medium">Evaluation:</span> accuracy, weighted F1-score, per-class precision/recall, and confusion matrices. Results reported on held-out splits consistent with each dataset's protocol.
            </li>
            <li className="leading-relaxed">
              <span className="font-medium">Visualization:</span> training history plots, confusion matrices, feature map visualizations, and sample predictions generated for model interpretability.
            </li>
            <li className="leading-relaxed">
              <span className="font-medium">Implementation:</span> modular architecture with separate preprocessing, training, evaluation, and visualization modules for each dataset.
            </li>
          </ul>
        </div>
      </section>

      <section id="results" className="max-w-6xl mx-auto py-10 md:py-12 border-t">
        <h2 className="text-2xl font-semibold mb-4">RESULTS (SUMMARY)</h2>
        <ul className="list-disc pl-6 space-y-2 text-gray-800">
          <li>CK+: near-ceiling on posed expressions; limited transfer to spontaneous affect.</li>
          <li>FER2013/AffectNet: modest overall accuracy; minority classes (e.g., disgust, contempt) underperform.</li>
          <li>Robustness sensitive to occlusion, pose, and illumination; higher-res RGB recommended.</li>
        </ul>
        <div className="mt-6">
          <Link to="/research">
            <Button variant="outline" className="border-gray-300 text-gray-900 hover:bg-gray-100">Open Full Findings</Button>
          </Link>
        </div>
      </section>

      <section id="code" className="max-w-6xl mx-auto py-10 md:py-12 border-t">
        <h2 className="text-2xl font-semibold mb-4">CODE</h2>
        <p className="text-gray-800 mb-4">Source code for training, evaluation, and the demo UI is organised under <code className="px-1 py-0.5 bg-gray-100 border rounded">src/</code> and <code className="px-1 py-0.5 bg-gray-100 border rounded">client/</code>. See Research for experimental details.</p>
        <div className="mt-4">
          <a 
            href="https://github.com/adepeju4/Emotion-Detection" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
            </svg>
            View on GitHub
          </a>
        </div>
      </section>

      <section id="contact" className="max-w-6xl mx-auto py-10 md:py-12 border-t">
        <h2 className="text-2xl font-semibold mb-4">CONTACT</h2>
        <p className="text-gray-800">For correspondence, please use the contact link in the site navigation.</p>
      </section>
    </div>
  );
}
