import { Link, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const linkBase = "transition-colors";
  const linkScrolled = "text-gray-700 hover:text-blue-600";
  const linkTop = "text-textLarge hover:text-blue-600";
  const linkActive = "text-blue-600";

  const getLinkClassName = (path: string) => {
    const isActive = location.pathname === path;
    if (isActive) {
      return cn(linkBase, linkActive);
    }
    return cn(linkBase, scrolled ? linkScrolled : linkTop);
  };

  return (
    <nav
      className={cn(
        "fixed top-0 left-0 w-full z-50 backdrop-blur-md transition-colors", 
        scrolled ? "bg-white/80 shadow-sm" : "bg-transparent"
      )}
    >
      <div className="max-w-6xl mx-auto h-14 flex items-center justify-between">
        <Link to="/" className={cn("font-bold text-lg", "text-textLarge")}>Using AI to Support Autism through Emotion Recognition</Link>
        <div className="flex gap-6 text-sm">
          <Link to="/learn" className={getLinkClassName("/learn")}>Learn</Link>
          <Link to="/research" className={getLinkClassName("/research")}>Research</Link>
          <Link to="/emotions" className={getLinkClassName("/emotions")}>Emotions</Link>
          <Link to="/upload-photo" className={getLinkClassName("/upload-photo")}>Upload Photo</Link>
          <Link to="/upload-video" className={getLinkClassName("/upload-video")}>Upload Video</Link>
          <Link to="/webcam" className={getLinkClassName("/webcam")}>Webcam</Link>
        </div>
      </div>
    </nav>
  );
} 