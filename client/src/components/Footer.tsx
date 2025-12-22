import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-50 border-t border-gray-200 mt-12 sm:mt-16">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 sm:gap-8">
          <div>
            <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-3 sm:mb-4">About This Research</h3>
            <p className="text-gray-600 text-xs sm:text-sm leading-relaxed">
              This emotion detection research project explores how Convolutional Neural Networks can support autistic individuals in recognizing and understanding facial expressions. Our goal is to create assistive technology that respects neurodiversity.
            </p>
          </div>
          
          <div>
            <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-3 sm:mb-4">Research Focus</h3>
            <ul className="space-y-1.5 sm:space-y-2 text-xs sm:text-sm text-gray-600">
              <li>• Autism advocacy and understanding</li>
              <li>• Emotion recognition challenges</li>
              <li>• Supportive technology development</li>
              <li>• Neurodiversity acceptance</li>
            </ul>
          </div>
          
          <div className="sm:col-span-2 md:col-span-1">
            <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-3 sm:mb-4">Contact</h3>
            <div className="space-y-1.5 sm:space-y-2 text-xs sm:text-sm text-gray-600">
              <p><strong>Researcher:</strong> Adepeju Peace Orefejo</p>
              <p><strong>Institution:</strong> Big Academy UAE</p>
              <p><strong>Partnership:</strong> Euclea Business School</p>
              <p><strong>Project Type:</strong> Academic Research</p>
            </div>
          </div>
        </div>
        
        <div className="border-t border-gray-200 mt-6 sm:mt-8 pt-6 sm:pt-8 text-center">
          <p className="text-xs sm:text-sm text-gray-500 px-4">
            © 2025 Emotion Detection Research Project. This research is conducted with respect for neurodiversity and aims to support autistic individuals through understanding, not "fixing."
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
