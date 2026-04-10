import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Sunburst from "@/components/Sunburst";
import { fetchDefinition, type DictionaryResult } from "@/lib/dictionary";
import { Skeleton } from "@/components/ui/skeleton";

export default function Emotions() {
  const [selectedEmotion, setSelectedEmotion] = useState<string | null>(null);

  const handleEmotionSelect = (emotion: string) => {
    if (emotion && emotion !== "emotions") {
      setSelectedEmotion(emotion);
    }
  };

  const { data: dictionaryResult, isLoading: isLoadingOxford } = useQuery<DictionaryResult | null>({
    queryKey: ["dictionary", selectedEmotion],
    queryFn: () => fetchDefinition(selectedEmotion!),
    enabled: !!selectedEmotion,
  });

  return (
    <div className="min-h-screen bg-white pt-2 pb-6 sm:pb-8">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8 items-start lg:items-center min-h-[calc(100vh-3rem)]">
          <div className="bg-white rounded-lg p-4 sm:p-6">
            <div className="flex justify-center items-center overflow-x-auto">
              <Sunburst onSelect={handleEmotionSelect} />
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 sm:p-6">
            {selectedEmotion ? (
              <div className="space-y-6">
                <div className="border-b pb-4">
                  <h2 className="text-xl sm:text-2xl md:text-3xl font-bold text-gray-900 break-words">{selectedEmotion}</h2>
                </div>

                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">Dictionary Definition</h3>
                    {isLoadingOxford ? (
                      <div className="space-y-4">
                        <Skeleton className="h-5 w-24" />
                        <div className="space-y-3">
                          {[1, 2, 3].map((idx) => (
                            <div key={idx} className="border-l-2 border-gray-200 pl-4 space-y-2">
                              <Skeleton className="h-3 w-16" />
                              <Skeleton className="h-4 w-full" />
                              <Skeleton className="h-4 w-3/4" />
                              <Skeleton className="h-3 w-2/3" />
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : dictionaryResult && dictionaryResult.definitions ? (
                      <>
                        {dictionaryResult.phonetic && (
                          <p className="text-gray-600 italic mb-4 text-base">/{dictionaryResult.phonetic}/</p>
                        )}
                        <div className="space-y-6">
                          {Object.entries(
                            dictionaryResult.definitions.reduce((acc, def) => {
                              const pos = def.partOfSpeech || "other";
                              if (!acc[pos]) acc[pos] = [];
                              acc[pos].push(def);
                              return acc;
                            }, {} as Record<string, typeof dictionaryResult.definitions>)
                          ).map(([partOfSpeech, definitions]) => (
                            <div key={partOfSpeech} className="space-y-3">
                              <h4 className="text-sm font-bold text-gray-700 uppercase tracking-wider">
                                {partOfSpeech}
                              </h4>
                              <div className="space-y-4 pl-4 border-l-2 border-gray-200">
                                {definitions.map((def, idx) => (
                                  <div key={idx} className="space-y-2">
                                    <p className="text-gray-800 leading-relaxed">{def.definition}</p>
                                    {def.example && (
                                      <p className="text-gray-600 italic text-sm pl-4 border-l-2 border-gray-100">
                                        <span className="font-medium text-gray-700">Example:</span> "{def.example}"
                                      </p>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                        <cite className="text-xs text-gray-400 mt-4 block">Free Dictionary API</cite>
                      </>
                    ) : (
                      <p className="text-gray-500 italic">Definition not available.</p>
                    )}
                  </div>

                </div>
              </div>
            ) : (
              <div className="text-center py-8 sm:py-12">
                <div className="text-4xl sm:text-6xl mb-4">🎯</div>
                <h3 className="text-lg sm:text-xl font-semibold text-gray-900 mb-2">Select an Emotion</h3>
                <p className="text-sm sm:text-base text-gray-600 mb-4 sm:mb-6 px-4">
                  Click on any segment of the emotion wheel to see its definition and details.
                </p>
                <div className="bg-gray-50 rounded-lg p-4 sm:p-6 text-left">
                  <h4 className="text-sm sm:text-base font-semibold text-gray-900 mb-3">How to Use This Tool</h4>
                  <ul className="list-disc pl-5 sm:pl-6 space-y-2 text-xs sm:text-sm text-gray-700">
                    <li>Click on any segment of the wheel to zoom into that emotion category</li>
                    <li>Explore sub-emotions and their relationships</li>
                    <li>Click the center circle to zoom back out</li>
                    <li>Use this as a reference when trying to identify or communicate your feelings</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}





