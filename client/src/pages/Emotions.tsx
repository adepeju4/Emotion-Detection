import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Sunburst from "@/components/Sunburst";
import { getEmotionDefinition } from "@/data/emotionDefinitions";
import { fetchDefinition, type DictionaryResult } from "@/lib/dictionary";
import { Skeleton } from "@/components/ui/skeleton";

export default function Emotions() {
  const [selectedEmotion, setSelectedEmotion] = useState<string | null>(null);

  const handleEmotionSelect = (emotion: string) => {
    if (emotion && emotion !== "emotions") {
      setSelectedEmotion(emotion);
    }
  };

  const emotionDef = selectedEmotion ? getEmotionDefinition(selectedEmotion) : null;

  const { data: dictionaryResult, isLoading: isLoadingOxford } = useQuery<DictionaryResult | null>({
    queryKey: ["dictionary", selectedEmotion],
    queryFn: () => fetchDefinition(selectedEmotion!),
    enabled: !!selectedEmotion,
  });

  return (
    <div className="min-h-screen bg-white py-6">
      <div className=" px-4">


        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center min-h-[calc(100vh-3rem)]">
          <div className="bg-white rounded-lg p-6">
            <div className="flex justify-center items-center">
              <Sunburst onSelect={handleEmotionSelect} />
            </div>
          </div>

          <div className="bg-white rounded-lg p-6">
            {selectedEmotion ? (
              <div className="space-y-6">
                <div className="border-b pb-4">
                  <div className="flex items-center mb-3">
                    {emotionDef?.icon && <span className="text-4xl mr-4">{emotionDef.icon}</span>}
                    <h2 className="text-3xl font-bold text-gray-900">{selectedEmotion}</h2>
                  </div>
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

                  {emotionDef?.autismContext && (
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">Autism Context</h3>
                      <p className="text-gray-700 leading-relaxed italic">{emotionDef.autismContext}</p>
                    </div>
                  )}

                  {emotionDef?.relatedEmotions && emotionDef.relatedEmotions.length > 0 && (
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">Related Emotions</h3>
                      <div className="flex flex-wrap gap-2">
                        {emotionDef.relatedEmotions.map((emotion, idx) => (
                          <span
                            key={idx}
                            className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm font-medium cursor-pointer hover:bg-blue-100 transition-colors"
                            onClick={() => setSelectedEmotion(emotion)}
                          >
                            {emotion}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {emotionDef?.physicalCues && emotionDef.physicalCues.length > 0 && (
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">Physical Cues</h3>
                      <ul className="list-disc pl-6 space-y-1">
                        {emotionDef.physicalCues.map((cue, idx) => (
                          <li key={idx} className="text-gray-700">{cue}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">🎯</div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Select an Emotion</h3>
                <p className="text-gray-600 mb-6">
                  Click on any segment of the emotion wheel to see its definition and details.
                </p>
                <div className="bg-gray-50 rounded-lg p-6 text-left">
                  <h4 className="font-semibold text-gray-900 mb-3">How to Use This Tool</h4>
                  <ul className="list-disc pl-6 space-y-2 text-sm text-gray-700">
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





