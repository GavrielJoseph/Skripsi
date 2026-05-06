import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { LayoutDashboard, PlusCircle, Scale, Activity } from "lucide-react";
import AnalysisPage from "./components/AnalysisPage";
import DashboardPage from "./components/DashboardPage";
import ResultPage from "./components/ResultPage";
import ModelInfoPage from "./components/ModelInfoPage";
import ComparePage from "./components/ComparePage";

function Navbar() {
  const baseClass = "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors";
  const activeClass = `${baseClass} bg-blue-50 text-blue-700`;
  const inactiveClass = `${baseClass} text-gray-600 hover:text-gray-900 hover:bg-gray-100`;
  
  const getNavClass = ({ isActive }) => isActive ? activeClass : inactiveClass;

  return (
    <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-8">
            <NavLink to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-gray-900 text-lg tracking-tight">SkinSentiment</span>
            </NavLink>
            
            <div className="hidden md:flex items-center gap-1">
              <NavLink to="/" end className={getNavClass}>
                <LayoutDashboard className="w-4 h-4" />
                Dashboard
              </NavLink>
              <NavLink to="/analyze" className={getNavClass}>
                <PlusCircle className="w-4 h-4" />
                Analisis Baru
              </NavLink>
              <NavLink to="/compare" className={getNavClass}>
                <Scale className="w-4 h-4" />
                Bandingkan
              </NavLink>
              <NavLink to="/model-info" className={getNavClass}>
                <Activity className="w-4 h-4" />
                System Metrics
              </NavLink>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      {/* Background diubah ke warna terang (gray-50) */}
      <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
        <Navbar />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/analyze" element={<AnalysisPage />} />
            <Route path="/results/:id" element={<ResultPage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/model-info" element={<ModelInfoPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}