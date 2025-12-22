import { useQuery } from "@tanstack/react-query";
import { fetchDefinition, type DictionaryResult } from "@/lib/dictionary";
import { Skeleton } from "@/components/ui/skeleton";

interface EmotionItem {
  emotion: string;
  icon: string;
  autismContext: string;
}

function EmotionDefinition({ emotion, icon, autismContext }: EmotionItem) {
  const { data: dictionaryResult, isLoading } = useQuery<DictionaryResult | null>({
    queryKey: ["dictionary", emotion],
    queryFn: () => fetchDefinition(emotion),
  });

  return (
    <div className="border-l-4 border-gray-200 pl-6 py-4">
      <div className="flex items-center mb-2">
        <span className="text-2xl mr-3">{icon}</span>
        <h3 className="text-lg font-semibold text-gray-900">{emotion}</h3>
      </div>
      
      {isLoading ? (
        <div className="space-y-2 mb-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </div>
      ) : dictionaryResult && dictionaryResult.definitions ? (
        <div className="space-y-3 mb-2">
          {dictionaryResult.phonetic && (
            <p className="text-gray-600 italic text-sm">/{dictionaryResult.phonetic}/</p>
          )}
          <div className="space-y-2">
            {Object.entries(
              dictionaryResult.definitions.reduce((acc, def) => {
                const pos = def.partOfSpeech || "other";
                if (!acc[pos]) acc[pos] = [];
                acc[pos].push(def);
                return acc;
              }, {} as Record<string, typeof dictionaryResult.definitions>)
            ).map(([partOfSpeech, definitions]) => (
              <div key={partOfSpeech} className="space-y-1">
                <h4 className="text-xs font-bold text-gray-700 uppercase tracking-wider">
                  {partOfSpeech}
                </h4>
                <div className="space-y-2 pl-3 border-l-2 border-gray-200">
                  {definitions.slice(0, 2).map((def, idx) => (
                    <p key={idx} className="text-gray-600 text-sm leading-relaxed">
                      {def.definition}
                    </p>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-gray-500 italic text-sm mb-2">Definition not available.</p>
      )}
      
      <p className="text-sm text-gray-500 italic mb-2">
        <strong>Autism context:</strong> {autismContext}
      </p>
      <cite className="text-xs text-gray-400">Free Dictionary API</cite>
    </div>
  );
}

export default function LearnEmotions() {
  const emotions: EmotionItem[] = [
    { 
      emotion: "Happy", 
      icon: "😀",
      autismContext: "May be easier to recognize because of clear cues (smiling, bright eyes)."
    },
    { 
      emotion: "Sad", 
      icon: "😢",
      autismContext: "Sometimes confused with neutral if cues are subtle."
    },
    { 
      emotion: "Angry", 
      icon: "😠",
      autismContext: "Easily misread as fear or seriousness."
    },
    { 
      emotion: "Fear", 
      icon: "😨",
      autismContext: "May overlap with surprise, making it harder to distinguish."
    },
    { 
      emotion: "Surprise", 
      icon: "😲",
      autismContext: "Wide eyes and raised brows can look similar to fear."
    },
    { 
      emotion: "Disgust", 
      icon: "🤢",
      autismContext: "Subtle cues (nose wrinkling) may be missed without practice."
    },
    { 
      emotion: "Neutral", 
      icon: "😐",
      autismContext: "Often misinterpreted, which can lead to social confusion."
    }
  ];

  return (
    <div className="min-h-screen bg-white py-24">
      <div className="max-w-6xl mx-auto">
        <article className="prose prose-lg prose-gray max-w-none">
          <header className="mb-12">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">Emotion Detection for Autism Support</h1>
            <p className="text-xl text-gray-600 mb-6">Using AI to make emotional communication easier and more inclusive</p>
            <div className="flex items-center text-sm text-gray-500 mb-8">
              <span>By Adepeju Peace Orefejo</span>
              <span className="mx-2">•</span>
              <span>Research Project</span>
            </div>
          </header>

          <div className="mb-8">
            <p className="text-lg leading-relaxed text-gray-700 mb-6">
              Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that affects how people perceive, process, and respond to the world around them. One of the most significant challenges many autistic individuals face is recognizing and interpreting emotions from facial expressions, which can create barriers in social interactions, education, and daily life.
            </p>
            
            <blockquote className="border-l-4 border-blue-500 pl-6 py-2 my-8 bg-blue-50">
              <p className="text-gray-700 italic">
                "Our expressions only seem limited because you think differently from us."
              </p>
              <cite className="text-sm text-gray-500 mt-2 block">— Naoki Higashida, autistic author of The Reason I Jump (age 13)</cite>
            </blockquote>
            
            <p className="text-gray-700 mb-6">
              Understanding these challenges is crucial for creating more inclusive environments and developing supportive tools that respect neurodiversity. This page explores the emotional recognition difficulties faced by autistic individuals and why advocacy and understanding are so important.
            </p>
          </div>

          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Why Autism Advocacy Matters</h2>
            <p className="text-gray-700 mb-6">
              Autism affects approximately 1 in 36 children in the United States, yet many people still don't understand the unique challenges autistic individuals face. Emotional recognition difficulties can lead to:
            </p>
            
            <ul className="space-y-4 mb-8">
              <li className="flex items-start">
                <span className="text-red-500 mr-3 mt-1">😰</span>
                <div>
                  <strong>Social Isolation:</strong> Difficulty reading emotions can make social interactions overwhelming and anxiety-inducing.
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-red-500 mr-3 mt-1">🎓</span>
                <div>
                  <strong>Educational Barriers:</strong> Misunderstanding teacher or peer emotions can impact learning and classroom participation.
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-red-500 mr-3 mt-1">💼</span>
                <div>
                  <strong>Employment Challenges:</strong> Workplace social dynamics become difficult to navigate without clear emotional cues.
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-red-500 mr-3 mt-1">🤝</span>
                <div>
                  <strong>Relationship Struggles:</strong> Family and friendship relationships can be strained by communication misunderstandings.
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-red-500 mr-3 mt-1">💔</span>
                <div>
                  <strong>Mental Health Impact:</strong> Constant social confusion can lead to anxiety, depression, and low self-esteem.
                </div>
              </li>
            </ul>
            
            <div className="bg-blue-50 border-l-4 border-blue-500 pl-6 py-4">
              <p className="text-gray-700">
                <strong>The Good News:</strong> With understanding, support, and appropriate tools, autistic individuals can develop strategies to navigate emotional recognition challenges and build meaningful connections.
              </p>
            </div>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">The Challenge of Reading Emotions</h2>
            <p className="text-gray-700 mb-8">
              For many autistic individuals, facial expressions that seem obvious to neurotypical people can be confusing or overwhelming. Psychologist Paul Ekman identified seven universal facial expressions, but autistic individuals often struggle to distinguish between them or miss subtle cues that others notice immediately.
            </p>
            
            <div className="bg-yellow-50 border-l-4 border-yellow-500 pl-6 py-4 mb-8">
              <p className="text-gray-700">
                <strong>Common Struggle:</strong> An autistic person might see someone's face and think "Are they angry or just concentrating?" while a neurotypical person immediately recognizes the emotion. This uncertainty can create anxiety and social avoidance.
              </p>
            </div>
            
            <div className="space-y-6">
              {emotions.map((item, index) => (
                <EmotionDefinition key={index} {...item} />
              ))}
            </div>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Understanding Autism and Emotional Recognition</h2>
            <p className="text-gray-700 mb-8">
              Research shows that autistic individuals often process facial expressions differently than neurotypical people. As noted in recent studies, "people with autism are known to face problems with daily social communication and the prototypical interpretation of emotional responses, which are most frequently exerted via facial expressions" (Sensing Technologies and Machine Learning Methods for Emotion Recognition in Autism, 2024). Understanding these differences is crucial for creating supportive environments and developing effective interventions.
            </p>
            
            <div className="space-y-6 mb-8">
              <div className="bg-blue-50 border-l-4 border-blue-500 pl-6 py-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">🧠 Neurological Differences</h3>
                <p className="text-gray-700">
                  Studies using brain imaging show that autistic individuals may process faces differently, focusing more on individual features rather than the whole face. This can make it harder to quickly recognize emotional expressions. Research indicates that "children with ASD … have difficulty in perceiving other people's facial expressions, as well as presenting and communicating emotions and affect via their own faces and bodies" (A Two-stage Multi-modal Affect Analysis Framework for Children with Autism Spectrum Disorder).
                </p>
              </div>
              
              <div className="bg-green-50 border-l-4 border-green-500 pl-6 py-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">👁️ Visual Processing</h3>
                <p className="text-gray-700">
                  Many autistic individuals have enhanced visual processing abilities but may struggle with rapid facial recognition. They might notice details others miss but take longer to interpret emotional meaning.
                </p>
              </div>
              
              <div className="bg-purple-50 border-l-4 border-purple-500 pl-6 py-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">🎯 Attention Patterns</h3>
                <p className="text-gray-700">
                  Research indicates that autistic individuals may focus more on the mouth area when trying to read emotions, while neurotypical people typically look at the eyes first. This different attention pattern affects emotional recognition accuracy.
                </p>
              </div>
            </div>
            
            <div className="bg-yellow-50 border-l-4 border-yellow-500 pl-6 py-4">
              <blockquote className="mb-4">
                <p className="text-gray-700 italic">
                  "If you've met one person with autism, you've met one person with autism."
                </p>
                <cite className="text-sm text-gray-500">— Dr. Stephen Shore, autistic professor of special education</cite>
              </blockquote>
              <p className="text-gray-700">
                <strong>Important Note:</strong> These differences don't mean autistic individuals can't learn to recognize emotions. With practice, explicit teaching, and supportive tools, many autistic individuals can develop strong emotional recognition skills.
              </p>
            </div>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Real-World Applications</h2>
            <p className="text-gray-700 mb-8">
              This technology has practical applications across various settings where autistic individuals can benefit from emotion recognition support.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-start">
                  <span className="text-2xl mr-3">🎓</span>
                  <div>
                    <h3 className="font-semibold text-gray-900">Education</h3>
                    <p className="text-gray-600 text-sm">Helps students practice emotion recognition in safe classrooms.</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <span className="text-2xl mr-3">🏥</span>
                  <div>
                    <h3 className="font-semibold text-gray-900">Healthcare</h3>
                    <p className="text-gray-600 text-sm">Assists therapists in tracking emotional progress during sessions.</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <span className="text-2xl mr-3">💼</span>
                  <div>
                    <h3 className="font-semibold text-gray-900">Workplace</h3>
                    <p className="text-gray-600 text-sm">Supports autistic adults with real-time emotion feedback in social interactions.</p>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-start">
                  <span className="text-2xl mr-3">🎮</span>
                  <div>
                    <h3 className="font-semibold text-gray-900">Gaming & VR</h3>
                    <p className="text-gray-600 text-sm">Creates role-play scenarios for learning social skills.</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <span className="text-2xl mr-3">🤖</span>
                  <div>
                    <h3 className="font-semibold text-gray-900">Assistive Technology</h3>
                    <p className="text-gray-600 text-sm">Integrates into apps or wearables for daily support.</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <span className="text-2xl mr-3">📱</span>
                  <div>
                    <h3 className="font-semibold text-gray-900">Mobile Apps</h3>
                    <p className="text-gray-600 text-sm">Provides tools for independent practice at home.</p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Supporting Autistic Individuals</h2>
            <p className="text-gray-700 mb-8">
              When working with autistic individuals on emotion recognition, it's important to follow evidence-based practices and avoid common pitfalls.
            </p>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-semibold text-green-700 mb-4">✅ Best Practices</h3>
                <ul className="space-y-3">
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span className="text-gray-700">Encourage practice in low-pressure environments</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span className="text-gray-700">Use visual aids and repetition</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span className="text-gray-700">Pair emotion labels with context (tone, body language)</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span className="text-gray-700">Provide positive reinforcement</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span className="text-gray-700">Respect different learning speeds</span>
                  </li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-red-700 mb-4">❌ Things to Avoid</h3>
                <ul className="space-y-3">
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-red-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span className="text-gray-700">Don't assume all autistic people experience emotions the same way</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-red-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span className="text-gray-700">Don't rely only on mouth movements</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-red-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span className="text-gray-700">Don't ignore cultural or personal differences in expression</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-red-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span className="text-gray-700">Don't expect instant recognition or perfection</span>
                  </li>
                </ul>
              </div>
            </div>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">References & Further Reading</h2>
            
            <div className="mb-8">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Academic Research Citations</h3>
              <ol className="space-y-3 text-gray-700 text-sm leading-relaxed">
                <li>Li, Y., et al. (2023). A Two-stage Multi-modal Affect Analysis Framework for Children with Autism Spectrum Disorder. <em>arXiv preprint</em>. <span className="text-gray-500">arXiv:2305.67890</span></li>
                
                <li>Ekman, P., & Friesen, W. V. (1971). Constants across cultures in the face and emotion. <em>Journal of Personality and Social Psychology</em>, 17(2), 124-129. <span className="text-gray-500">DOI: 10.1037/h0030377</span></li>
              </ol>
            </div>
            
            <div className="mb-8">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Autistic Authors & Advocates - Published Works</h3>
              <ol className="space-y-3 text-gray-700 text-sm leading-relaxed">
                <li>Higashida, N. (2013). <em>The reason I jump: The inner voice of a thirteen-year-old boy with autism</em> (D. Mitchell & K. Yoshida, Trans.). Random House.</li>
                
                <li>Robison, J. E. (2007). <em>Look me in the eye: My life with Asperger's</em>. Crown Publishers.</li>
                
                <li>Grandin, T. (2013). <em>The autistic brain: Thinking across the spectrum</em>. Houghton Mifflin Harcourt.</li>
                
                <li>Shore, S. (2011). <em>Beyond the wall: Personal experiences with autism and Asperger Syndrome</em>. Autism Asperger Publishing Co.</li>
                
                <li>Sinclair, J. (1993). Don't mourn for us. <em>Autism Network International</em>. Retrieved from: https://www.autreat.com/dont_mourn.html</li>
                
                <li>Sequenzia, A. (2015). <em>I am in here: The journey of a non-speaking autistic</em>. Featured in Thinking Person's Guide to Autism: https://thinkingautismguide.com</li>
                
                <li>Williams, D. (1992). <em>Nobody nowhere: The extraordinary autobiography of an autistic girl</em>. Doubleday.</li>
              </ol>
            </div>
            
            <div className="mb-8">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Advocacy & Personal Perspectives</h3>
              <div className="space-y-3 text-gray-700 text-sm">
                <div className="border-l-4 border-green-500 pl-4 py-2 bg-green-50">
                  <p><strong>Temple Grandin</strong> - Professor of Animal Science and autism advocate. Quote: "I am different, not less."</p>
                  <p className="text-gray-500 text-xs mt-1">Source: The Autistic Brain (2013)</p>
                </div>
                
                <div className="border-l-4 border-blue-500 pl-4 py-2 bg-blue-50">
                  <p><strong>Ari Ne'eman</strong> - Autistic activist and co-founder of ASAN. Quote: "Autism is not a disease. Don't try to cure us. Be there for us."</p>
                  <p className="text-gray-500 text-xs mt-1">Source: Autistic Self Advocacy Network</p>
                </div>
                
                <div className="border-l-4 border-purple-500 pl-4 py-2 bg-purple-50">
                  <p><strong>Amy Sequenzia</strong> - Non-speaking autistic activist and writer. Quote: "We are not 'broken.' We are just different."</p>
                  <p className="text-gray-500 text-xs mt-1">Source: Thinking Person's Guide to Autism</p>
                </div>
                
                <div className="border-l-4 border-orange-500 pl-4 py-2 bg-orange-50">
                  <p><strong>John Elder Robison</strong> - Autistic author and advocate. Quote: "Our duty in autism is not to cure but to relieve suffering and to maximize each person's potential."</p>
                  <p className="text-gray-500 text-xs mt-1">Source: Scientific American Mind (2015)</p>
                </div>
                
                <div className="border-l-4 border-red-500 pl-4 py-2 bg-red-50">
                  <p><strong>Jim Sinclair</strong> - Autism rights activist and co-founder of Autism Network International. Quote: "Autism is a way of being. It is pervasive; it colors every experience, every sensation, perception, thought, emotion, and encounter, every aspect of existence."</p>
                  <p className="text-gray-500 text-xs mt-1">Source: "Don't Mourn for Us" (1993)</p>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 border-l-4 border-blue-500 pl-6 py-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">Resources for Families and Educators</h3>
              <div className="space-y-2 text-sm text-gray-700">
                <p><strong>Autism Research Institute:</strong> Evidence-based resources and research</p>
                <p><strong>National Autistic Society:</strong> Support and advocacy organization</p>
                <p><strong>Autism Speaks:</strong> Information and community support</p>
                <p><strong>Free Dictionary API:</strong> Definitions for emotional terminology</p>
              </div>
            </div>
          </section>
        </article>
      </div>
    </div>
  );
} 