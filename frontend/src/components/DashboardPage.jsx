import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Tag, MessageSquare, CheckCircle2, BarChart3, FolderOpen, Trash2, AlertCircle, X } from "lucide-react";

const API = "http://127.0.0.1:8000/api";

function StatCard({ label, value, colorClass, icon: Icon }) {
  return (
    <div className="bg-white border border-gray-200 shadow-sm rounded-xl p-5">
      <div className="flex items-center justify-between mb-2">
        <p className="text-gray-500 text-sm font-medium">{label}</p>
        <span className={colorClass}><Icon className="w-5 h-5" /></span>
      </div>
      <p className="text-3xl font-bold text-gray-900">{value}</p>
    </div>
  );
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const [brands, setBrands] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedBrand, setSelectedBrand] = useState(null);

  const fetchBrands = () => {
    setLoading(true);
    fetch(`${API}/sessions/by-brand`)
      .then(r => r.json())
      .then(d => { if (d.status === "success") setBrands(d.brands); setLoading(false); })
      .catch(() => setLoading(false));
  };

  useEffect(() => {
    fetchBrands();
  }, []);

  // Fungsi Hapus Satu Brand (Semua sesi di dalamnya)
  const handleDeleteBrand = async (brandName) => {
    if (!window.confirm(`PERINGATAN: Yakin ingin menghapus SELURUH data analisis untuk brand "${brandName}"?`)) return;

    try {
      const res = await fetch(`${API}/brands/${encodeURIComponent(brandName)}`, { method: "DELETE" });
      if (res.ok) {
        if (selectedBrand === brandName) setSelectedBrand(null);
        fetchBrands(); 
      } else {
        alert("Gagal menghapus brand. Pastikan endpoint backend tersedia.");
      }
    } catch (error) {
      alert("Terjadi kesalahan jaringan.");
    }
  };

  // Fungsi Hapus Satu Sesi Saja (Dalam Modal)
  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation(); // Mencegah klik baris yang mengarah ke Result Page
    if (!window.confirm("Yakin ingin menghapus sesi pengujian ini?")) return;

    try {
      // Pastikan backend Laravel punya Route::delete('/api/sessions/{id}')
      const res = await fetch(`${API}/sessions/${sessionId}`, { method: "DELETE" });
      if (res.ok) {
        fetchBrands(); // Refresh data. Modal akan otomatis ter-update karena brands state berubah.
      } else {
        alert("Gagal menghapus sesi.");
      }
    } catch (error) {
      alert("Terjadi kesalahan jaringan.");
    }
  };

  const totalReviews  = brands.reduce((a, b) => a + b.total_reviews, 0);
  const totalPositive = brands.reduce((a, b) => a + b.positive_count, 0);
  const avgPositive   = totalReviews > 0 ? ((totalPositive / totalReviews) * 100).toFixed(1) : "0.0";

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-500 text-sm font-medium">Memuat data dashboard...</span>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto relative">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 mb-1">Dashboard Analisis</h1>
          <p className="text-gray-500 text-sm">Ringkasan performa sentimen ulasan produk</p>
        </div>
        <button onClick={() => navigate("/analyze")} className="bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium px-4 py-2 rounded-md transition-colors shadow-sm">
          + Analisis Baru
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Brand" value={brands.length} colorClass="text-blue-500" icon={Tag} />
        <StatCard label="Total Ulasan" value={totalReviews.toLocaleString()} colorClass="text-indigo-500" icon={MessageSquare} />
        <StatCard label="Ulasan Positif" value={`${avgPositive}%`} colorClass="text-green-500" icon={CheckCircle2} />
        <StatCard label="Total Sesi" value={brands.reduce((a,b)=>a+b.total_sessions,0)} colorClass="text-purple-500" icon={BarChart3} />
      </div>

      {brands.length === 0 ? (
        <div className="bg-white border border-gray-200 rounded-xl p-16 text-center shadow-sm">
          <FolderOpen className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-gray-900 font-semibold text-lg mb-2">Belum ada data analisis</h3>
          <p className="text-gray-500 text-sm mb-6">Unggah file CSV ulasan produk untuk memulai analisis sentimen.</p>
          <button onClick={() => navigate("/analyze")} className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-md font-medium text-sm transition-colors shadow-sm">
            Mulai Analisis Pertama
          </button>
        </div>
      ) : (
        <>
          <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">Daftar Analisis per Brand</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {brands.map((brand, i) => {
              const pct = brand.total_reviews > 0 ? ((brand.positive_count / brand.total_reviews) * 100).toFixed(1) : 0;
              
              return (
                <div key={i} className="bg-white rounded-xl p-5 border border-gray-200 shadow-sm hover:border-blue-400 hover:shadow-md transition-all duration-200 cursor-pointer group" onClick={() => setSelectedBrand(brand.brand_name)}>
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <h3 className="font-bold text-gray-900 text-lg group-hover:text-blue-600 transition-colors">{brand.brand_name}</h3>
                      <p className="text-gray-500 text-xs mt-1">{brand.total_sessions} sesi pengujian · {brand.total_reviews.toLocaleString()} ulasan total</p>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <span className={`text-xs font-bold px-3 py-1 rounded-full ${pct >= 70 ? "bg-green-100 text-green-700 border border-green-200" : "bg-red-100 text-red-700 border border-red-200"}`}>
                        {pct}% positif
                      </span>
                      <button onClick={(e) => { e.stopPropagation(); handleDeleteBrand(brand.brand_name); }} className="text-gray-400 hover:text-red-600 p-1 transition-colors bg-white rounded-md hover:bg-red-50" title="Hapus data brand ini">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  <div className="flex h-2.5 rounded-full overflow-hidden mb-3 bg-gray-100">
                    <div className="bg-green-500" style={{ width: `${pct}%` }} />
                    <div className="bg-red-500" style={{ width: `${100 - pct}%` }} />
                  </div>
                  <div className="flex gap-6 text-xs font-medium">
                    <span className="text-green-600 flex items-center gap-1"><CheckCircle2 className="w-3.5 h-3.5" /> {brand.positive_count.toLocaleString()} positif</span>
                    <span className="text-red-600 flex items-center gap-1"><AlertCircle className="w-3.5 h-3.5" /> {brand.negative_count.toLocaleString()} negatif</span>
                  </div>
                </div>
              );
            })}
          </div>

          {selectedBrand && (() => {
            const brand = brands.find(b => b.brand_name === selectedBrand);
            if(!brand) return null;
            return (
              <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/40 backdrop-blur-sm transition-opacity" onClick={() => setSelectedBrand(null)}>
                <div className="bg-white rounded-xl shadow-2xl w-full max-w-5xl max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
                  
                  <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between bg-gray-50/50">
                    <div>
                      <h3 className="font-bold text-gray-900 flex items-center gap-2 text-lg">
                        <FolderOpen className="w-5 h-5 text-blue-600" /> Riwayat Sesi Pengujian
                      </h3>
                      <p className="text-sm text-gray-500 mt-1">Brand: <span className="font-semibold text-gray-700">{selectedBrand}</span></p>
                    </div>
                    <button onClick={() => setSelectedBrand(null)} className="p-2 text-gray-400 hover:text-gray-900 hover:bg-gray-200 rounded-lg transition-colors">
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  
                  <div className="overflow-y-auto p-6 bg-white">
                    <div className="overflow-x-auto rounded-lg border border-gray-200">
                      <table className="w-full text-sm text-left">
                        <thead className="bg-gray-50 text-gray-700 text-xs uppercase">
                          <tr>
                            <th className="px-4 py-3 font-semibold">Nama File Dataset</th>
                            <th className="px-4 py-3 font-semibold text-right">Total Ulasan</th>
                            <th className="px-4 py-3 font-semibold text-right">Positif</th>
                            <th className="px-4 py-3 font-semibold text-right">Negatif</th>
                            <th className="px-4 py-3 font-semibold text-right">Cluster</th>
                            <th className="px-4 py-3 font-semibold text-right">Tanggal Analisis</th>
                            <th className="px-4 py-3 font-semibold text-center w-16">Delete</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {brand.sessions.map((s, i) => {
                            const p = s.total_reviews > 0 ? ((s.positive_count/s.total_reviews)*100).toFixed(1) : 0;
                            return (
                              <tr key={i} onClick={() => navigate(`/results/${s.id}`)} className="hover:bg-blue-50/50 cursor-pointer transition-colors group">
                                <td className="px-4 py-3 font-medium text-gray-900 max-w-xs truncate group-hover:text-blue-600 underline-offset-2 group-hover:underline">{s.filename}</td>
                                <td className="px-4 py-3 text-right text-gray-600">{s.total_reviews.toLocaleString()}</td>
                                <td className="px-4 py-3 text-right text-green-600 font-medium">{s.positive_count.toLocaleString()} ({p}%)</td>
                                <td className="px-4 py-3 text-right text-red-600">{s.negative_count.toLocaleString()}</td>
                                <td className="px-4 py-3 text-right text-blue-600">{s.n_clusters}</td>
                                <td className="px-4 py-3 text-right text-gray-500 text-xs">
                                  {new Date(s.created_at).toLocaleDateString("id-ID", { day:"numeric", month:"short", year:"numeric" })}
                                </td>
                                <td className="px-4 py-3 text-center">
                                  {/* Tombol Hapus Sesi Individual */}
                                  <button onClick={(e) => handleDeleteSession(e, s.id)} className="text-gray-400 hover:text-red-600 p-1.5 rounded-md hover:bg-red-50 transition-colors opacity-0 group-hover:opacity-100" title="Hapus sesi ini">
                                    <Trash2 className="w-4 h-4" />
                                  </button>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
        </>
      )}
    </div>
  );
}