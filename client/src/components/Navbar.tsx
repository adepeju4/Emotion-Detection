import { Link, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";
import { Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

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

  const navLinks = [
    { path: "/learn", label: "Learn" },
    { path: "/research", label: "Research" },
    { path: "/emotions", label: "Emotions" },
    { path: "/upload-photo", label: "Upload Photo" },
    { path: "/upload-video", label: "Upload Video" },
    { path: "/webcam", label: "Webcam" },
  ];

  return (
    <nav
      className={cn(
        "fixed top-0 left-0 w-full z-50 backdrop-blur-md transition-colors", 
        scrolled ? "bg-white/80 shadow-sm" : "bg-transparent"
      )}
    >
      <div className="max-w-6xl mx-auto h-20 px-4 flex items-center justify-between">
        <Link 
          to="/" 
          className={cn(
            "font-bold text-sm sm:text-base md:text-lg truncate pr-2",
            "text-textLarge"
          )}
        >
          <span className="hidden sm:inline">Using AI to Support Autism through Emotion Recognition</span>
          <span className="sm:hidden">Emotion Detection</span>
        </Link>
        
        {/* Desktop Navigation */}
        <div className="hidden lg:flex gap-2 text-sm">
          {navLinks.map(({ path, label }) => (
            <Link key={path} to={path} className={cn(getLinkClassName(path), "px-3 py-2 rounded-md")}>
              {label}
            </Link>
          ))}
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="lg:hidden p-2 text-gray-700 hover:text-blue-600 transition-colors"
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-white/95 backdrop-blur-md border-t border-gray-200 shadow-lg">
          <div className="max-w-6xl mx-auto px-4 py-4 space-y-3">
            {navLinks.map(({ path, label }) => (
              <Link
                key={path}
                to={path}
                className={cn(
                  "block py-2 text-sm transition-colors",
                  getLinkClassName(path)
                )}
                onClick={() => setMobileMenuOpen(false)}
              >
                {label}
              </Link>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
} 