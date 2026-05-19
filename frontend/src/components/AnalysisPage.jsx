import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  UploadCloud, FileText, CheckCircle2, AlertCircle, Loader2,
  Link, Globe, Clock, Info,
} from "lucide-react";

const API = "http://127.0.0.1:8000/api";

const STEPS_UPLOAD = [
  { label: "Cleaning & Emoji",      desc: "Konversi emoji ke kata, hapus URL/mention/tanda baca" },
  { label: "Case Folding",          desc: "Ubah seluruh teks ke huruf kecil" },
  { label: "Normalisasi Kata",      desc: "Perbaiki kata alay/slang via 3 kamus gabungan" },
  { label: "Tokenizing",            desc: "Pisahkan teks menjadi array token" },
  { label: "Stopword Removal",      desc: "Hapus kata tidak bermakna, pertahankan kata negasi" },
  { label: "Negasi Handling",       desc: "Gabung 'tidak'+'cocok' → 'tidak_cocok' (1 token)" },
  { label: "Stemming",              desc: "Kembalikan kata ke bentuk dasar (PySastrawi)" },
  { label: "Bigram & Word2Vec",     desc: "Deteksi frasa + vektorisasi 100 dimensi" },
  { label: "Klasifikasi Sentimen",  desc: "Logistic Regression → positif / negatif + confidence" },
  { label: "AHC Clustering",        desc: "Klasterisasi topik, K optimal via Silhouette Score" },
];

const STEPS_SCRAPE = [
  { label: "Inisialisasi Browser",  desc: "Membuka Chrome headless via Selenium" },
  { label: "Buka Halaman Produk",   desc: "Navigasi ke URL Tokopedia" },
  { label: "Scrape Bintang 5",      desc: "Mengambil ulasan bintang 5" },
  { label: "Scrape Bintang 4",      desc: "Mengambil ulasan bintang 4" },
  { label: "Scrape Bintang 3",      desc: "Mengambil ulasan bintang 3" },
  { label: "Scrape Bintang 2",      desc: "Mengambil ulasan bintang 2" },
  { label: "Scrape Bintang 1",      desc: "Mengambil ulasan bintang 1" },
  { label: "Simpan CSV",            desc: "Simpan hasil scraping ke file CSV" },
  { label: "Cleaning & Normalisasi",desc: "Preprocessing: cleaning, case fold, normalisasi, tokenize, stopword, negasi, stemming" },
  { label: "Bigram & Word2Vec",     desc: "Deteksi frasa + vektorisasi 100 dimensi" },
  { label: "Klasifikasi Sentimen",  desc: "Logistic Regression → positif / negatif + confidence" },
  { label: "AHC Clustering",        desc: "Klasterisasi topik, K optimal via Silhouette Score" },
];

export default function AnalysisPage() {
  const navigate = useNavigate();

  // Mode: "upload" atau "scrape"
  const [mode, setMode] = useState("upload");

  // State shared
  const [existingBrands, setExistingBrands]               = useState([]);
  const [selectedBrandMode, setSelectedBrandMode]         = useState("existing");
  const [selectedExistingBrand, setSelectedExistingBrand] = useState("");
  const [newBrandName, setNewBrandName]                   = useState("");
  const [loading, setLoading]                             = useState(false);
  const [currentStep, setCurrentStep]                     = useState(-1);
  const [error, setError]                                 = useState("");

  // State mode upload
  const [file, setFile]                     = useState(null);
  const [preview, setPreview]               = useState([]);
  const [previewHeaders, setPreviewHeaders] = useState([]);
  const [textColumn, setTextColumn]         = useState("");
  const [productColumn, setProductColumn]   = useState("");

  // State mode scrape
  const [scrapeUrl, setScrapeUrl]                 = useState("");
  const [scrapeProductName, setScrapeProductName] = useState("");
   const [scrapeMaxPages, setScrapeMaxPages]       = useState(50);

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

      const sep     = lines[0].includes(";") ? ";" : ",";
      const headers = lines[0].split(sep).map(h =>
        h.trim().replace(/^\uFEFF/, "").replace(/['"]/g, "")
      );
      const rows = lines.slice(1).map(l =>
        l.split(sep).map(c => c.trim().replace(/['"]/g, ""))
      );

      setPreviewHeaders(headers);
      setPreview(rows);

      const autoText = headers.find(h =>
        ["comment", "ulasan", "review", "text", "konten", "isi"].includes(h.toLowerCase())
      );
      if (autoText) setTextColumn(autoText);
      else if (headers.length > 0) setTextColumn(headers[headers.length - 1]);

      const autoProduct = headers.find(h =>
        ["productname", "product", "nama produk", "nama_produk", "item"].includes(h.toLowerCase())
      );
      if (autoProduct) setProductColumn(autoProduct);
    };
    reader.readAsText(f);
  };

  const handleAnalyze = async () => {
    const finalBrandName = selectedBrandMode === "existing"
      ? selectedExistingBrand
      : newBrandName;

    if (!finalBrandName.trim()) {
      setError("Nama brand tidak boleh kosong.");
      return;
    }

    if (mode === "upload") {
      if (!file) { setError("Pilih file dataset (CSV) terlebih dahulu."); return; }
      if (!textColumn) { setError("Kolom Teks Ulasan wajib dipilih."); return; }

      // Validasi ekstensi file
      const ext = file.name.split('.').pop().toLowerCase();
      if (ext !== 'csv') {
        if (ext === 'xlsx' || ext === 'xls') {
          setError("File Excel tidak didukung. Buka file di Excel, lalu simpan ulang sebagai CSV (File → Save As → CSV).");
        } else {
          setError("Format file tidak didukung. Hanya file .csv yang diterima.");
        }
        return;
      }
    } else {
      if (!scrapeUrl.trim()) { setError("URL Tokopedia tidak boleh kosong."); return; }
      if (!scrapeProductName.trim()) { setError("Nama produk tidak boleh kosong."); return; }

      // Validasi URL harus tokopedia.com
      if (!scrapeUrl.includes("tokopedia.com")) {
        setError("URL harus dari tokopedia.com. URL dari Shopee, Lazada, atau platform lain tidak didukung.");
        return;
      }

      // Validasi format URL yang valid
      try {
        const parsed = new URL(scrapeUrl);
        const pathParts = parsed.pathname.split('/').filter(Boolean);
        if (pathParts.length < 2) {
          setError("URL tidak mengarah ke halaman produk. Salin URL langsung dari halaman produk Tokopedia.");
          return;
        }
      } catch {
        setError("Format URL tidak valid. Pastikan URL dimulai dengan https://");
        return;
      }
    }

    setLoading(true);
    setError("");
    setCurrentStep(0);

    const steps = mode === "upload" ? STEPS_UPLOAD : STEPS_SCRAPE;
    let s = 0;

    const interval = setInterval(() => {
      s = Math.min(s + 1, steps.length - 1);
      setCurrentStep(s);
    }, mode === "upload" ? 800 : 2500);

    try {
      let res, data;

      if (mode === "upload") {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("brand_name", finalBrandName.trim());
        fd.append("text_column", textColumn);
        if (productColumn) fd.append("product_column", productColumn);

        res  = await fetch(`${API}/analyze`, { method: "POST", body: fd });
        data = await res.json();
      } else {
        const fd = new FormData();
        fd.append("url",          scrapeUrl.trim());
        fd.append("brand_name",   finalBrandName.trim());
        fd.append("product_name", scrapeProductName.trim());
        fd.append("max_pages",    String(scrapeMaxPages));

        res  = await fetch(`${API}/scrape-and-analyze`, { method: "POST", body: fd });
        data = await res.json();
      }

      clearInterval(interval);
      setCurrentStep(steps.length);

      if (data.status === "success") {
        setTimeout(() => navigate(`/results/${data.session_id}`), 600);
      } else {
        setError(data.message || "Proses gagal. Silakan coba lagi.");
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

  const steps = mode === "upload" ? STEPS_UPLOAD : STEPS_SCRAPE;

  // ─── PERUBAHAN 1: wrapper div → fragment, tambah background ───────────────
  return (
    <>
      <img
        src="/wardah.jpg"
        alt=""
        aria-hidden="true"
        style={{
          position: "fixed", top: 0, left: 0,
          width: "100vw", height: "100vh",
          objectFit: "cover", objectPosition: "center 35%",
          filter: "blur(8px) brightness(0.88) saturate(1.15)",
          transform: "scale(1.06)",
          zIndex: -2, pointerEvents: "none",
        }}
      />
      <div style={{
        position: "fixed", inset: 0,
        background: "rgba(255,255,255,0.32)",
        zIndex: -1, pointerEvents: "none",
      }} />

    {/* ─── Konten identik dengan kode asli, hanya bg card yang diubah ──────── */}
    <div className="max-w-5xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900">Analisis Sentimen Baru</h1>
        <p className="text-slate-600 text-sm mt-1">
          Upload dataset CSV atau scrape langsung dari URL produk Tokopedia.
        </p>
      </div>

      <div className="flex gap-3 mb-6">
        <button
          onClick={() => { setMode("upload"); setError(""); }}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold border transition-all ${
            mode === "upload"
              ? "bg-blue-600 text-white border-blue-600 shadow-sm"
              : "bg-white text-gray-600 border-gray-300 hover:border-blue-400 hover:text-blue-600"
          }`}
        >
          <UploadCloud className="w-4 h-4" /> Upload CSV
        </button>
        <button
          onClick={() => { setMode("scrape"); setError(""); }}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold border transition-all ${
            mode === "scrape"
              ? "bg-blue-600 text-white border-blue-600 shadow-sm"
              : "bg-white text-gray-600 border-gray-300 hover:border-blue-400 hover:text-blue-600"
          }`}
        >
          <Globe className="w-4 h-4" /> Scrape dari Tokopedia
        </button>
      </div>

      {mode === "scrape" && (
        <div className="flex items-start gap-3 p-4 bg-amber-50 border border-amber-200 rounded-xl mb-6 text-sm text-amber-800">
          <Clock className="w-5 h-5 flex-shrink-0 mt-0.5 text-amber-500" />
          <div>
            <p className="font-semibold mb-1">Estimasi waktu: 15–30 menit</p>
            <p className="text-amber-700 text-xs leading-relaxed">
              <span className="font-bold">Jangan tutup atau reload halaman ini selama proses berjalan.</span>{" "}
              Sistem akan membuka browser, mengklik filter bintang satu per satu, dan mengambil ulasan per halaman secara otomatis.
              Semakin banyak halaman yang di-scrape, semakin lama prosesnya.
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">

          {/* ─── PERUBAHAN 2: bg-white → bg-white/90 backdrop-blur-sm border-sky-100 ─ */}
          <div className="bg-white/90 backdrop-blur-sm rounded-xl border border-sky-100 shadow-sm p-6">
            <h2 className="text-base font-semibold text-gray-900 mb-4 border-b border-gray-100 pb-2">
              1. Identitas Brand
            </h2>

            {existingBrands.length > 0 && (
              <div className="flex gap-4 mb-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={selectedBrandMode === "existing"}
                    onChange={() => setSelectedBrandMode("existing")}
                    className="text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Pilih Brand Tersedia</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={selectedBrandMode === "new"}
                    onChange={() => setSelectedBrandMode("new")}
                    className="text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-700">Tambah Brand Baru</span>
                </label>
              </div>
            )}

            {selectedBrandMode === "existing" ? (
              <select
                value={selectedExistingBrand}
                onChange={e => setSelectedExistingBrand(e.target.value)}
                disabled={loading}
                className="w-full bg-white border border-gray-300 rounded-md px-3 py-2.5 text-sm text-gray-900 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
              >
                {existingBrands.map((b, i) => <option key={i} value={b}>{b}</option>)}
              </select>
            ) : (
              <input
                type="text"
                placeholder="Contoh: Skintific, Wardah, dsb."
                value={newBrandName}
                onChange={e => { setNewBrandName(e.target.value); setError(""); }}
                disabled={loading}
                className="w-full bg-white border border-gray-300 rounded-md px-3 py-2.5 text-sm text-gray-900 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
              />
            )}
          </div>

          {mode === "upload" ? (
            <div className="bg-white/90 backdrop-blur-sm rounded-xl border border-sky-100 shadow-sm p-6">
              <h2 className="text-base font-semibold text-gray-900 mb-4 border-b border-gray-100 pb-2">
                2. Upload Dataset CSV
              </h2>

              <label className={`mt-2 flex justify-center w-full h-32 px-4 transition bg-white border-2 border-dashed rounded-lg appearance-none cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 ${
                loading ? "opacity-50 cursor-not-allowed" : ""
              } ${file ? "border-blue-500 bg-blue-50/30" : "border-gray-300"}`}>
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
                      <p className="text-gray-400 text-xs mt-1">Pemetaan kolom dilakukan di bawah</p>
                    </div>
                  )}
                </span>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  disabled={loading}
                  className="hidden"
                />
              </label>

              {previewHeaders.length > 0 && (
                <div className="mt-6 p-5 bg-gray-50 rounded-lg border border-gray-200">
                  <h3 className="text-sm font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-blue-600" /> Pemetaan Kolom
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide">
                        Kolom Teks Ulasan <span className="text-red-500">*</span>
                      </label>
                      <select
                        value={textColumn}
                        onChange={e => setTextColumn(e.target.value)}
                        disabled={loading}
                        className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                      >
                        <option value="" disabled>-- Pilih Kolom --</option>
                        {previewHeaders.map((h, i) => <option key={i} value={h}>{h}</option>)}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide">
                        Kolom Nama Produk (Opsional)
                      </label>
                      <select
                        value={productColumn}
                        onChange={e => setProductColumn(e.target.value)}
                        disabled={loading}
                        className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                      >
                        <option value="">-- Tidak ada --</option>
                        {previewHeaders.map((h, i) => <option key={i} value={h}>{h}</option>)}
                      </select>
                    </div>
                  </div>
                </div>
              )}

              {previewHeaders.length > 0 && (
                <div className="mt-6">
                  <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">
                    Preview Data (5 baris pertama)
                  </p>
                  <div className="overflow-x-auto rounded-lg border border-gray-200">
                    <table className="w-full text-xs text-left whitespace-nowrap">
                      <thead className="bg-gray-100 text-gray-700">
                        <tr>
                          {previewHeaders.map((h, i) => (
                            <th key={i} className={`px-4 py-3 border-b border-gray-200 font-semibold ${
                              h === textColumn ? "bg-blue-100 text-blue-800" : ""
                            }`}>
                              {h}
                              {h === textColumn && (
                                <span className="ml-1 text-[10px] bg-blue-200 px-1.5 py-0.5 rounded text-blue-800">
                                  ML Target
                                </span>
                              )}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100 bg-white">
                        {preview.map((row, i) => (
                          <tr key={i} className="hover:bg-gray-50">
                            {row.map((cell, j) => (
                              <td key={j} className={`px-4 py-2.5 text-gray-600 max-w-[200px] truncate ${
                                previewHeaders[j] === textColumn ? "bg-blue-50/30" : ""
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
          ) : (
            <div className="bg-white/90 backdrop-blur-sm rounded-xl border border-sky-100 shadow-sm p-6">
              <h2 className="text-base font-semibold text-gray-900 mb-4 border-b border-gray-100 pb-2">
                2. Input URL Produk Tokopedia
              </h2>

              <div className="space-y-4">
                {/* === BAGIAN URL DENGAN LOGIKA AUTO-FILL === */}
                <div>
                  <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide">
                    URL Produk Tokopedia <span className="text-red-500">*</span>
                  </label>
                  <div className="flex items-center gap-2 border border-gray-300 rounded-md px-3 py-2.5 bg-white focus-within:ring-1 focus-within:ring-blue-500 focus-within:border-blue-500 shadow-sm">
                    <Link className="w-4 h-4 text-gray-400 flex-shrink-0" />
                    <input
                      type="url"
                      placeholder="https://www.tokopedia.com/namatoko/nama-produk-..."
                      value={scrapeUrl}
                      onChange={(e) => {
                        const val = e.target.value;
                        setScrapeUrl(val);
                        setError("");

                        // Logika ekstrak nama produk otomatis
                        try {
                          if (val.includes("tokopedia.com")) {
                            const urlObj = new URL(val);
                            const pathParts = urlObj.pathname.split('/').filter(Boolean);
                            
                            if (pathParts.length >= 2) {
                              let slug = pathParts[pathParts.length - 1];
                              // Hapus deretan angka di belakang (ID produk)
                              slug = slug.replace(/-\d+$/, '');
                              
                              // Ganti strip (-) dengan spasi dan jadikan kapital
                              const cleanName = slug
                                .split('-')
                                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                                .join(' ');
                                
                              setScrapeProductName(cleanName);
                            }
                          }
                        } catch (err) {
                          // Abaikan jika URL belum valid (misal: baru diketik "http")
                        }
                      }}
                      disabled={loading}
                      className="flex-1 text-sm text-gray-900 outline-none bg-transparent"
                    />
                  </div>
                  <p className="text-[11px] text-gray-500 mt-1">
                    Salin URL dari halaman produk Tokopedia. Nama produk akan otomatis terisi.
                  </p>
                </div>

                {/* Nama Produk */}
                <div>
                  <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide">
                    Nama Produk <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    placeholder="Contoh: Wardah Acnederm Foaming Cleanser"
                    value={scrapeProductName}
                    onChange={e => { setScrapeProductName(e.target.value); setError(""); }}
                    disabled={loading}
                    className="w-full bg-white border border-gray-300 rounded-md px-3 py-2.5 text-sm text-gray-900 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm"
                  />
                  <p className="text-[11px] text-gray-500 mt-1">
                    Digunakan sebagai nama file CSV dan label di dashboard.
                  </p>
                </div>

                {/* Max Pages */}
                <div>
                  <label className="block text-xs font-semibold text-gray-700 mb-1.5 uppercase tracking-wide">
                    Max Halaman per Bintang
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={scrapeMaxPages}
                      onChange={e => setScrapeMaxPages(parseInt(e.target.value) || 50)}
                      disabled={loading}
                      className="w-24 bg-white border border-gray-300 rounded-md px-3 py-2.5 text-sm text-gray-900 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 shadow-sm text-center"
                    />
                    <div className="flex items-start gap-2 text-xs text-gray-500">
                      <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-blue-400" />
                      <span>
                        Semakin banyak halaman, semakin banyak ulasan yang terkumpul —
                        tapi waktu scraping juga semakin lama. 50 halaman per bintang bisa memakan waktu 30 menit lebih.
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 p-4 text-sm text-red-700 bg-red-50 rounded-lg border border-red-200">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <p>{error}</p>
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={loading}
            className={`w-full py-3.5 px-4 rounded-lg text-sm font-semibold text-white shadow-sm transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-600 ${
              loading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                {mode === "upload"
                  ? "Sistem Sedang Memproses Data Pipeline..."
                  : "Scraping & Analisis Sedang Berjalan... (Harap Tunggu)"}
              </span>
            ) : (
              mode === "upload" ? "Mulai Analisis Sentimen" : "Mulai Scraping & Analisis"
            )}
          </button>

          {loading && mode === "scrape" && (
            <p className="text-center text-xs text-gray-500 -mt-2">
              Jangan tutup atau reload halaman ini. Proses berjalan di background.
            </p>
          )}
        </div>

        <div className="lg:col-span-1">
          <div className="bg-white/90 backdrop-blur-sm rounded-xl border border-sky-100 shadow-sm p-6 sticky top-24">
            <h2 className="text-base font-semibold text-gray-900 mb-4 border-b border-gray-100 pb-2">
              Status Pipeline
            </h2>
            <div className="space-y-4">
              {steps.map((step, i) => {
                const isDone   = currentStep > i;
                const isActive = currentStep === i;
                return (
                  <div key={i} className="flex items-start gap-3">
                    <div className="mt-0.5 relative">
                      {isDone ? (
                        <CheckCircle2 className="w-5 h-5 text-green-500" />
                      ) : isActive ? (
                        <span className="relative flex h-5 w-5 items-center justify-center">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-20" />
                          <Loader2 className="relative w-5 h-5 text-blue-600 animate-spin" />
                        </span>
                      ) : (
                        <div className="w-5 h-5 rounded-full border-2 border-gray-200" />
                      )}
                      {i !== steps.length - 1 && (
                        <div className={`absolute top-6 left-2.5 w-[2px] h-4 ${
                          isDone ? "bg-green-200" : "bg-gray-100"
                        }`} />
                      )}
                    </div>
                    <div>
                      <p className={`text-sm font-medium ${
                        isDone ? "text-gray-900" : isActive ? "text-blue-700 font-semibold" : "text-gray-400"
                      }`}>
                        {step.label}
                      </p>
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
    {/* ─── PERUBAHAN 3: tutup fragment ───────────────────────────────────────── */}
    </>
  );
}
