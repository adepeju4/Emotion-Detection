import { Routes, Route } from "react-router-dom";
import { Toaster } from "sonner";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import ScrollToTop from "@/components/ScrollToTop";
import Home from "./pages/Home";
import About from "./pages/About";
import UploadPhoto from "./pages/UploadPhoto";
import UploadVideo from "./pages/UploadVideo";
import WebcamAnalysis from "./pages/WebcamAnalysis";
import LearnEmotions from "./pages/LearnEmotions";
import Research from "./pages/Research";
import Emotions from "./pages/Emotions";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 60, // 1 hour
      gcTime: 1000 * 60 * 60 * 24, // 24 hours (formerly cacheTime)
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ScrollToTop />
      <Navbar />
      <div className="pt-16 lg:pt-20">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/upload-photo" element={<UploadPhoto />} />
        <Route path="/upload-video" element={<UploadVideo />} />
        <Route path="/webcam" element={<WebcamAnalysis />} />
        <Route path="/learn" element={<LearnEmotions />} />
        <Route path="/research" element={<Research />} />
        <Route path="/emotions" element={<Emotions />} />
      </Routes>
      </div>
      <Footer />
      <Toaster position="top-right" />
    </QueryClientProvider>
  );
}

export default App;
