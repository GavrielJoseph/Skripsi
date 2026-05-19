import { useState, useEffect } from "react";
import { Activity, FileCode2, Database, BrainCircuit, GitMerge, CheckCircle, XCircle } from "lucide-react";

const API = "http://127.0.0.1:8000/api";

function MetricBar({ label, value, colorClass, barColor, desc }) {
  return (
    <div className="mb-5">
      <div className="flex justify-between items-end mb-1.5">
        <div>
          <span className="text-gray-800 text-sm font-bold uppercase tracking-wider">{label}</span>
          {desc && <p className="text-gray-500 text-xs mt-0.5">{desc}</p>}
        </div>
        <span className={`font-bold text-lg ${colorClass}`}>{value}%</span>
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
        <div className={`h-2.5 rounded-full transition-all duration-700 ${barColor}`} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function InfoRow({ label, value, isCode }) {
  return (
    <div className="flex justify-between items-start py-3 border-b border-gray-100 last:border-0">
      <span className="text-gray-500 text-sm font-medium">{label}</span>
      <span className={`text-sm text-right max-w-sm ${isCode ? 'font-mono text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded border border-gray-200' : 'text-gray-900 font-semibold'}`}>
        {value}
      </span>
    </div>
  );
}

function ConfusionMatrix({ cm }) {
  if (!cm) return null;
  const { tp, tn, fp, fn } = cm;
  const total = tp + tn + fp + fn;

  return (
    <div className="mt-4">
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <table className="w-full text-sm text-center">
          <thead>
            <tr>
              <th className="bg-gray-50 p-3 border-b border-r border-gray-200 text-gray-500 font-medium w-1/3">n = {total.toLocaleString()}</th>
              <th className="bg-gray-50 p-3 border-b border-gray-200 text-gray-700 font-bold w-1/3">Predicted: Negatif</th>
              <th className="bg-gray-50 p-3 border-b border-gray-200 text-gray-700 font-bold w-1/3">Predicted: Positif</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="bg-gray-50 p-4 border-r border-b border-gray-200 text-gray-700 font-bold">Actual: Negatif</td>
              <td className="p-4 border-b border-r border-gray-200 bg-green-50">
                <div className="text-2xl font-black text-green-700">{tn.toLocaleString()}</div>
                <div className="text-xs text-green-600 font-medium mt-1">True Negative (TN)</div>
              </td>
              <td className="p-4 border-b border-gray-200 bg-red-50">
                <div className="text-2xl font-black text-red-700">{fp.toLocaleString()}</div>
                <div className="text-xs text-red-600 font-medium mt-1">False Positive (FP)</div>
              </td>
            </tr>
            <tr>
              <td className="bg-gray-50 p-4 border-r border-gray-200 text-gray-700 font-bold">Actual: Positif</td>
              <td className="p-4 border-r border-gray-200 bg-red-50">
                <div className="text-2xl font-black text-red-700">{fn.toLocaleString()}</div>
                <div className="text-xs text-red-600 font-medium mt-1">False Negative (FN)</div>
              </td>
              <td className="p-4 bg-green-50">
                <div className="text-2xl font-black text-green-700">{tp.toLocaleString()}</div>
                <div className="text-xs text-green-600 font-medium mt-1">True Positive (TP)</div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      
      <div className="mt-4 bg-gray-50 rounded-lg p-4 border border-gray-200">
        <p className="text-gray-600 text-sm leading-relaxed">
          <span className="font-bold text-gray-900">Catatan Evaluasi:</span> Model berhasil mengklasifikasikan 
          <span className="font-bold text-green-600"> {(tn + tp).toLocaleString()} ulasan dengan benar</span> (TN + TP). 
          Kesalahan tipe I (FP) sebanyak {fp.toLocaleString()} kasus, dan kesalahan tipe II (FN) sebanyak {fn.toLocaleString()} kasus.
        </p>
      </div>
    </div>
  );
}

export default function ModelInfoPage() {
  const [perf, setPerf]       = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/model-performance`)
      .then(r => r.json())
      .then(d => { if (d.status === "success") setPerf(d.performance); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-500 font-medium">Memuat log sistem...</span>
    </div>
  );

  return (
    <>
      {/* Background Wardah — sama dengan halaman lain */}
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

      <div className="max-w-5xl mx-auto">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-slate-900 mb-1 flex items-center gap-2">
            <FileCode2 className="w-6 h-6 text-blue-600" /> System Metrics & Technical Logs
          </h1>
          <p className="text-slate-600 text-sm">Spesifikasi arsitektur model, parameter hyper-tuning, dan log evaluasi klasifikasi.</p>
        </div>

        <div className="space-y-6">

          {/* Metrik Evaluasi */}
          <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
            <h2 className="text-base font-bold text-gray-900 mb-4 flex items-center gap-2 border-b border-gray-100 pb-2">
              <Activity className="w-5 h-5 text-blue-600" /> Laporan Klasifikasi (Classification Report)
            </h2>
            
            {perf ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <p className="text-gray-500 text-xs mb-6 bg-gray-50 p-2 rounded border border-gray-100">
                    Diuji pada populasi testing sebesar <span className="font-bold text-gray-700">{perf?.testing_size?.toLocaleString() ?? "—"} sampel</span> (20% split).
                  </p>
                  <MetricBar label="Accuracy" value={perf.accuracy} colorClass="text-green-600" barColor="bg-green-500" desc="Rasio total tebakan benar terhadap total data uji." />
                  <MetricBar label="Precision" value={perf.precision} colorClass="text-blue-600" barColor="bg-blue-500" desc="Akurasi dari kelas yang diprediksi positif." />
                  <MetricBar label="Recall" value={perf.recall} colorClass="text-purple-600" barColor="bg-purple-500" desc="Kemampuan model menemukan semua data positif aktual." />
                  <MetricBar label="F1-Score" value={perf.f1_score} colorClass="text-yellow-600" barColor="bg-yellow-500" desc="Harmonic mean dari Precision & Recall (Metrik prioritas)." />
                </div>
                <div>
                  <p className="text-gray-800 text-sm font-bold uppercase tracking-wider mb-2">Confusion Matrix</p>
                  {perf?.confusion_matrix ? (
                    <ConfusionMatrix cm={perf.confusion_matrix} />
                  ) : (
                    <div className="text-center py-10 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-500 text-sm">Matrix belum tersedia dari backend.</p>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-10 bg-gray-50 rounded-lg border border-dashed border-gray-300">
                <p className="text-gray-500 text-sm">Data log model belum digenerate oleh backend (train.py belum dijalankan).</p>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Hyperparameters */}
            <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
              <h2 className="text-base font-bold text-gray-900 mb-4 flex items-center gap-2 border-b border-gray-100 pb-2">
                <BrainCircuit className="w-5 h-5 text-blue-600" /> Konfigurasi Algoritma
              </h2>
              <div className="space-y-4 mt-2">
                {[
                  {
                    badge: "Embedding", name: "Word2Vec (Skip-gram)",
                    desc: "Mengekstrak fitur semantik berdimensi rapat (dense vector). Mendukung deteksi N-gram untuk frasa (contoh: 'tidak_cocok').",
                    // params: "vector_size=100, window=5, sg=1, epochs=10"
                  },
                  {
                    badge: "Classification", name: "Logistic Regression",
                    desc: "Estimasi probabilitas biner (sigmoid) dengan penyesuaian bobot kelas untuk menangani ketidakseimbangan data (imbalanced dataset).",
                    // params: "C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced'"
                  },
                  {
                    badge: "Clustering", name: "Agglomerative Hierarchical",
                    desc: "Pengelompokan bottom-up menggunakan metrik jarak antar vektor topik. Jumlah K optimal divalidasi dengan Silhouette Score.",
                    // params: "linkage='average', affinity='cosine', max_k=6"
                  },
                ].map((a, i) => (
                  <div key={i} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="bg-gray-800 text-white text-[10px] px-2 py-0.5 rounded uppercase font-bold tracking-wider">{a.badge}</span>
                      <h3 className="font-bold text-gray-900 text-sm">{a.name}</h3>
                    </div>
                    <p className="text-gray-600 text-xs leading-relaxed mb-3">{a.desc}</p>
                    <div className="bg-gray-100 border border-gray-200 p-2 rounded text-[11px] font-mono text-gray-800 break-words">
                      {a.params}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Dataset Info & Preprocessing */}
            <div className="space-y-6">
              <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
                <h2 className="text-base font-bold text-gray-900 mb-4 flex items-center gap-2 border-b border-gray-100 pb-2">
                  <Database className="w-5 h-5 text-blue-600" /> Spesifikasi Dataset Induk
                </h2>
                <InfoRow label="Skema Pelabelan Asli" value="Rating 1-2 (Negatif), 4-5 (Positif), 3 (Drop)" />
                <InfoRow label="Bobot Kelas (Class Weight)" value="Balanced Penalty" />
                {perf && (
                  <>
                    <InfoRow label="N Total (Clean)" value={`${perf.total_data?.toLocaleString()}`} isCode />
                    <InfoRow label="N Train (80%)" value={`${perf.training_size?.toLocaleString()}`} isCode />
                    <InfoRow label="N Test (20%)" value={`${perf.testing_size?.toLocaleString()}`} isCode />
                  </>
                )}
              </div>

              <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
                <h2 className="text-base font-bold text-gray-900 mb-4 flex items-center gap-2 border-b border-gray-100 pb-2">
                  <GitMerge className="w-5 h-5 text-blue-600" /> Pipeline Preprocessing Teks
                </h2>
                <div className="space-y-2">
                  {[
                    "Regex Cleansing (Drop URLs, Tags, Angka)",
                    "Case Folding (Lowercasing)",
                    "Normalisasi Kamus Gaul & Slang",
                    "Stopword Removal (Pengecualian kata negasi)",
                    "Stemming (PySastrawi)",
                    "N-Gram Collocation Detetion (Frasa penggabung)"
                  ].map((step, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm text-gray-700">
                      <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </>
  );
}
