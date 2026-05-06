import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, FileText, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";

const API = "http://127.0.0.1:8000/api";

const STEPS = [
  { label: "Cleaning", desc: "Hapus URL, emoji, tanda baca" },
  { label: "Case Folding", desc: "Ubah ke huruf kecil" },
  { label: "Normalisasi Kata", desc: "Kamus bahasa gaul/alay" },
  { label: "Tokenizing", desc: "Pemisahan token kata" },
  { label: "Stopword Removal", desc: "Hapus kata tidak bermakna" },
  { label: "Stemming", desc: "PySastrawi stemming" },
  { label: "Word2Vec", desc: "Vektorisasi 100 dimensi" },
  { label: "Klasifikasi", desc: "Logistic Regression" },
  { label: "Clustering", desc: "AHC Clustering" },
];

export default function AnalysisPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [existingBrands, setExistingBrands] = useState([]);
  const [selectedBrandMode, setSelectedBrandMode] = useState("existing"); 
  const [selectedExistingBrand, setSelectedExistingBrand] = useState("");
  const [newBrandName, setNewBrandName] = useState("");
  
  const [preview, setPreview] = useState([]);
  const [previewHeaders, setPreviewHeaders] = useState([]);
  
  // State untuk Data Mapping
  const [textColumn, setTextColumn] = useState(""); 
  const [productColumn, setProductColumn] = useState("");

  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/sessions/by-brand`)
      .then(r => r.json())
      .then(d => {
        if (d.status === "success" && d.brands.length > 0) {
          setExistingBrands(d.brands.map(b => b.brand_name));
          setSelectedExistingBrand(d.brands[0].brand_name);
        } else {
          setSelectedBrandMode("new");
        }
      })
      .catch(() => setSelectedBrandMode("new"));
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setError("");
    setPreview([]);
    setPreviewHeaders([]);
    setTextColumn("");
    setProductColumn("");

    const reader = new FileReader();
    reader.onload = (ev) => {
      const lines = ev.target.result.split("\n").filter(Boolean).slice(0, 6);
      if (lines.length === 0) return;
      
      const sep = lines[0].includes(";") ? ";" : ",";
      const headers = lines[0].split(sep).map(h => h.trim().replace(/^\uFEFF/, "").replace(/['"]/g, ''));
      const rows = lines.slice(1).map(l => l.split(sep).map(c => c.trim().replace(/['"]/g, '')));
      
      setPreviewHeaders(headers);
      setPreview(rows);

      // Auto-detect kolom Teks Ulasan
      const autoText = headers.find(h => ["comment", "ulasan", "review", "text", "konten", "isi"].includes(h.toLowerCase()));
      if (autoText) setTextColumn(autoText);
      else if (headers.length > 0) setTextColumn(headers[headers.length - 1]);

      // Auto-detect kolom Nama Produk
      const autoProduct = headers.find(h => ["productname", "product", "nama produk", "nama_produk", "item"].includes(h.toLowerCase()));
      if (autoProduct) setProductColumn(autoProduct);
    };
    reader.readAsText(f);
  };

  const handleAnalyze = async () => {
    const finalBrandName = selectedBrandMode === "existing" ? selectedExistingBrand : newBrandName;

    if (!file) { setError("Pilih file dataset (CSV) terlebih dahulu."); return; }
    if (!finalBrandName.trim()) { setError("Nama brand tidak boleh kosong."); return; }
    if (!textColumn) { setError("Kolom Teks Ulasan wajib dipilih agar ML bisa bekerja."); return; }

    setLoading(true);
    setError("");
    setCurrentStep(0);

    let s = 0;
    const interval = setInterval(() => {
      s = Math.min(s + 1, STEPS.length - 1);
      setCurrentStep(s);
    }, 900);

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("brand_name", finalBrandName.trim());
      
      // Kirim pemetaan kolom ke backend
      fd.append("text_column", textColumn); 
      if (productColumn) fd.append("product_column", productColumn); 

      const res = await fetch(`${API}/analyze`, { method: "POST", body: fd });
      clearInterval(interval);
      setCurrentStep(STEPS.length);

      const data = await res.json();
      if (data.status === "success") {
        setTimeout(() => navigate(`/results/${data.session_id}`), 600);
      } else {
        setError(data.message || "Analisis gagal. Pastikan format CSV benar dan delimiter menggunakan koma atau titik koma.");
        setLoading(false);
        setCurrentStep(-1);
      }
    } catch {
      clearInterval(interval);
      setError("Koneksi ke backend gagal. Pastikan service Laravel berjalan.");
      setLoading(false);
      setCurrentStep(-1);
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Analisis Sentimen Baru</h1>
        <p className="text-gray-500 text-sm mt-1">Unggah dataset CSV ulasan produk untuk diproses oleh pipeline machine learning.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          
          {/* Card 1: Brand */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h2 className="text-base font-semibold text-gray-900 mb-4 border-b border-gray-100 pb-2">1. Identitas Brand</h2>
            
            {existingBrands.length > 0 && (
              <div className="flex gap-4 mb-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="radio" checked={selectedBrandMode === "existing"} onChange={() => setSelectedBrandMode("existing")} className="text-blue-600 focus:ring-blue-500" />
                  <span className="text-sm font-medium text-gray-700">Pilih Brand Tersedia</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="radio" checked={selectedBrandMode === "new"} onChange={() => setSelectedBrandMode("new")} className="text-blue-600 focus:ring-blue-500" />
                  <span className="text-sm font-medium text-gray-700">Tambah Brand Baru</span>
                </label>
              </div>
            )}

            {selectedBrandMode === "existing" ? (
              <select value={selectedExistingBrand} onChange={(e) => setSelectedExistingBrand(e.target.value)} disabled={loading} className="w-full bg-white border border-gray-300 rounded-md px-3 py-2.5 text-sm text-gray-900 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm">
                {existingBrands.map((b, i) => <option key={i} value={b}>{b}</option>)}
              </select>
            ) : (
              <input type="text" placeholder="Contoh: Skintific, Wardah, dsb." value={newBrandName} onChange={e => { setNewBrandName(e.target.value); setError(""); }} disabled={loading} className="w-full bg-white border border-gray-300 rounded-md px-3 py-2.5 text-sm text-gray-900 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm" />
            )}
          </div>

          {/* Card 2: Upload */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h2 className="text-base font-semibold text-gray-900 mb-4 border-b border-gray-100 pb-2">2. Upload Dataset</h2>
            
            <label className={`mt-2 flex justify-center w-full h-32 px-4 transition bg-white border-2 border-gray-300 border-dashed rounded-lg appearance-none cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 ${loading ? "opacity-50 cursor-not-allowed" : ""} ${file ? "border-blue-500 bg-blue-50/30" : ""}`}>
              <span className="flex items-center space-x-2">
                {file ? (
                  <div className="text-center">
                    <FileText className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                    <span className="font-medium text-blue-700 text-sm">{file.name}</span>
                    <p className="text-gray-500 text-xs mt-1">{(file.size / 1024).toFixed(1)} KB</p>
                  </div>
                ) : (
                  <div className="text-center">
                    <UploadCloud className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <span className="font-medium text-gray-600 text-sm">Klik atau drag file CSV ke sini</span>
                    <p className="text-gray-400 text-xs mt-1">Sistem menerima semua format kolom (Pemetaan dilakukan di bawah)</p>
                  </div>
                )}
              </span>
              <input type="file" accept=".csv" onChange={handleFileChange} disabled={loading} className="hidden" />
            </label>

            {/* Fitur Pemetaan Kolom Data Engineering */}
            {previewHeaders.length > 0 && (
              <div className="mt-6 p-5 bg-gray-50 rounded-lg border border-gray-200">
                <h3 className="text-sm font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-blue-600" /> Pemetaan Kolom (Data Mapping)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide">
                      Kolom Teks Ulasan <span className="text-red-500">*</span>
                    </label>
                    <select value={textColumn} onChange={(e) => setTextColumn(e.target.value)} disabled={loading} className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm">
                      <option value="" disabled>-- Pilih Kolom --</option>
                      {previewHeaders.map((h, i) => <option key={i} value={h}>{h}</option>)}
                    </select>
                    <p className="text-[11px] text-gray-500 mt-1">Wajib. Data ini akan diproses oleh ML.</p>
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide">
                      Kolom Nama Produk (Opsional)
                    </label>
                    <select value={productColumn} onChange={(e) => setProductColumn(e.target.value)} disabled={loading} className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm">
                      <option value="">-- Tidak ada / Abaikan --</option>
                      {previewHeaders.map((h, i) => <option key={i} value={h}>{h}</option>)}
                    </select>
                    <p className="text-[11px] text-gray-500 mt-1">Hanya untuk ditampilkan di tabel hasil.</p>
                  </div>
                </div>
              </div>
            )}

            {previewHeaders.length > 0 && (
              <div className="mt-6">
                <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">Preview Data (5 baris pertama)</p>
                <div className="overflow-x-auto rounded-lg border border-gray-200">
                  <table className="w-full text-xs text-left whitespace-nowrap">
                    <thead className="bg-gray-100 text-gray-700">
                      <tr>
                        {previewHeaders.map((h, i) => (
                          <th key={i} className={`px-4 py-3 border-b border-gray-200 font-semibold ${
                            h === textColumn ? 'bg-blue-100 text-blue-800' : h === productColumn ? 'bg-green-50 text-green-800' : ''
                          }`}>
                            {h}
                            {h === textColumn && <span className="ml-1 text-[10px] bg-blue-200 px-1.5 py-0.5 rounded text-blue-800">ML Target</span>}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 bg-white">
                      {preview.map((row, i) => (
                        <tr key={i} className="hover:bg-gray-50">
                          {row.map((cell, j) => (
                            <td key={j} className={`px-4 py-2.5 text-gray-600 max-w-[200px] truncate ${
                              previewHeaders[j] === textColumn ? 'bg-blue-50/30' : previewHeaders[j] === productColumn ? 'bg-green-50/20' : ''
                            }`}>
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="flex items-start gap-2 p-4 text-sm text-red-700 bg-red-50 rounded-lg border border-red-200">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <p>{error}</p>
            </div>
          )}

          <button onClick={handleAnalyze} disabled={loading} className={`w-full py-3.5 px-4 rounded-lg text-sm font-semibold text-white shadow-sm transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-600 ${loading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"}`}>
            {loading ? <span className="flex items-center justify-center gap-2"><Loader2 className="w-4 h-4 animate-spin" /> Sistem Sedang Memproses Data Pipeline...</span> : "Mulai Analisis Sentimen"}
          </button>
        </div>

        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 sticky top-24">
            <h2 className="text-base font-semibold text-gray-900 mb-4 border-b border-gray-100 pb-2">Status Pipeline ML</h2>
            <div className="space-y-4">
              {STEPS.map((step, i) => {
                const isDone = currentStep > i;
                const isActive = currentStep === i;
                return (
                  <div key={i} className="flex items-start gap-3">
                    <div className="mt-0.5 relative">
                      {isDone ? (
                        <CheckCircle2 className="w-5 h-5 text-green-500" />
                      ) : isActive ? (
                        <span className="relative flex h-5 w-5 items-center justify-center">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-20"></span>
                          <Loader2 className="relative w-5 h-5 text-blue-600 animate-spin" />
                        </span>
                      ) : (
                        <div className="w-5 h-5 rounded-full border-2 border-gray-200" />
                      )}
                      {i !== STEPS.length - 1 && (
                        <div className={`absolute top-6 left-2.5 w-[2px] h-4 ${isDone ? 'bg-green-200' : 'bg-gray-100'}`} />
                      )}
                    </div>
                    <div>
                      <p className={`text-sm font-medium ${isDone ? "text-gray-900" : isActive ? "text-blue-700 font-semibold" : "text-gray-400"}`}>{step.label}</p>
                      <p className="text-xs text-gray-500 mt-0.5">{step.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}