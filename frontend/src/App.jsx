import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { LayoutDashboard, PlusCircle, Activity, Info } from "lucide-react";
import AnalysisPage  from "./components/AnalysisPage";
import DashboardPage from "./components/DashboardPage";
import ResultPage    from "./components/ResultPage";
import ModelInfoPage from "./components/ModelInfoPage";
import AboutPage     from "./components/AboutPage";

function Navbar() {
  const base     = "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors";
  const active   = `${base} bg-blue-50 text-blue-700`;
  const inactive = `${base} text-gray-600 hover:text-gray-900 hover:bg-gray-100`;
  const cls      = ({ isActive }) => isActive ? active : inactive;

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center h-16 gap-8">
          <NavLink to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-gray-900 text-lg tracking-tight">SkinSentiment</span>
          </NavLink>
          <div className="hidden md:flex items-center gap-1">
            <NavLink to="/" end className={cls}><LayoutDashboard className="w-4 h-4" /> Dashboard</NavLink>
            <NavLink to="/analyze"    className={cls}><PlusCircle className="w-4 h-4" /> Analisis Baru</NavLink>
            <NavLink to="/model-info" className={cls}><Activity className="w-4 h-4" /> System Metrics</NavLink>
            <NavLink to="/about"      className={cls}><Info className="w-4 h-4" /> Tentang Sistem</NavLink>
          </div>
        </div>
      </div>
    </nav>
  );
}

function AppContent() {
  const { pathname } = useLocation();
  // Halaman-halaman yang punya background sendiri (tidak perlu bg-gray-50)
  const noBg = ["/", "/analyze", "/about", "/model-info"].includes(pathname)
    || pathname.startsWith("/results/");

  return (
    <div className={`min-h-screen text-gray-900 font-sans ${noBg ? "" : "bg-gray-50"}`}>
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/"            element={<DashboardPage />} />
          <Route path="/analyze"     element={<AnalysisPage />} />
          <Route path="/results/:id" element={<ResultPage />} />
          <Route path="/model-info"  element={<ModelInfoPage />} />
          <Route path="/about"       element={<AboutPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return <BrowserRouter><AppContent /></BrowserRouter>;
}
