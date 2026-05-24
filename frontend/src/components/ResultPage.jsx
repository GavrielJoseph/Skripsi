import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from "recharts";
import {
  ArrowLeft, Tag, Calendar, Lightbulb, PieChart as PieChartIcon,
  MessageSquare, BarChart2, CheckCircle2, AlertTriangle,
  Truck, FlaskConical, Banknote, Package, ShieldCheck, Search,
  ChevronDown, ChevronUp, Quote, Download,
} from "lucide-react";

const API = "http://127.0.0.1:8000/api";
const PAGE_SIZE = 10;

function cleanProductName(filename) {
  if (!filename) return "produk ini";
  let name = filename
    .replace(/\.csv$/i, "")
    .replace(/^\d+_/, "")
    .replace(/^scraped_/i, "")
    .replace(/_/g, " ")
    .trim();
  const words = name.split(" ").filter(Boolean);
  return words.slice(0, 6).join(" ");
}

const CLUSTER_SKIP = new Set([
  "nya","yg","aja","jd","tp","banget","sih","buat","biar","mah",
  "kalo","pas","terus","sama","udah","gini","gitu","ya","deh","dong",
  "kok","kan","pake","sekali","baru","dari","ke","di","ini","itu",
  "yang","dan","aku","saya","dia","mereka","kita","kami","untuk",
  "dalam","pada","juga","sudah","ada","saja","lagi","karena",
  "kalau","pun","bisa","akan","jadi","tapi",
]);

function clusterName(keywords) {
  if (!keywords || keywords.length === 0) return "Topik Umum";
  const filtered = keywords.filter(k =>
    !k.includes("_") && !k.startsWith("tidak") && !CLUSTER_SKIP.has(k) && k.length >= 3
  );
  const top = (filtered.length > 0 ? filtered : keywords).slice(0, 2);
  return top.map(k => k.charAt(0).toUpperCase() + k.slice(1)).join(" & ");
}

const TOPIC_META = {
  "Efek & Kecocokan Produk": { icon: FlaskConical, color: "blue"   },
  "Layanan Pengiriman":       { icon: Truck,        color: "purple" },
  "Harga Produk":             { icon: Banknote,     color: "yellow" },
  "Kondisi Kemasan":          { icon: Package,      color: "orange" },
  "Layanan Toko & Keaslian":  { icon: ShieldCheck,  color: "teal"   },
};

const COLOR_MAP = {
  blue:   { bg: "bg-blue-50",   border: "border-blue-200",   text: "text-blue-700",   icon: "text-blue-500"   },
  purple: { bg: "bg-purple-50", border: "border-purple-200", text: "text-purple-700", icon: "text-purple-500" },
  yellow: { bg: "bg-amber-50",  border: "border-amber-200",  text: "text-amber-700",  icon: "text-amber-500"  },
  orange: { bg: "bg-orange-50", border: "border-orange-200", text: "text-orange-700", icon: "text-orange-500" },
  teal:   { bg: "bg-teal-50",   border: "border-teal-200",   text: "text-teal-700",   icon: "text-teal-500"   },
};

// Menghasilkan ringkasan otomatis berbasis output klasifikasi sentimen
// dan distribusi aspek dari topic_overview
function generateInsight(session, topicOverview) {
  const total  = session.total_reviews;
  const pos    = session.positive_count;
  const posPct = ((pos / total) * 100).toFixed(1);

  let sentiment_text = "";
  if      (posPct >= 80) sentiment_text = `positif (${posPct}% ulasan bernada baik)`;
  else if (posPct >= 60) sentiment_text = `cukup baik (${posPct}% ulasan positif)`;
  else if (posPct >= 40) sentiment_text = `terbagi (campuran positif dan negatif)`;
  else                   sentiment_text = `cenderung negatif (hanya ${posPct}% ulasan positif)`;

  const topics = Object.values(topicOverview || {}).filter(t => t.count > 0);
  topics.sort((a, b) => b.count - a.count);

  const topDiscussed = topics.slice(0, 2).map(t => t.label).join(" dan ");
  const topText = topDiscussed
    ? `Aspek yang paling banyak dibahas konsumen adalah "${topDiscussed}".`
    : "";

  const topComplaint = topics
    .filter(t => t.negative > 0)
    .sort((a, b) => (b.neg_share || 0) - (a.neg_share || 0))[0];

  const complaint_text = topComplaint
    ? `Dari ${session.negative_count} keluhan yang terdeteksi, ${topComplaint.neg_share}% berkaitan dengan "${topComplaint.label}".`
    : session.negative_count > 0
    ? `Terdapat ${session.negative_count} ulasan negatif yang tersebar merata di berbagai aspek.`
    : "Tidak ditemukan keluhan yang berarti.";

  return { sentiment_text, topText, complaint_text, posPct };
}

// Analisis sentimen per aspek produk
// Input: topic_overview dari Python (hasil rule-based keyword matching
// terhadap output klasifikasi sentimen Logistic Regression)
function TopicOverviewCard({ topicOverview }) {
  if (!topicOverview || Object.keys(topicOverview).length === 0) return null;
  const sorted = Object.entries(topicOverview)
    .filter(([, v]) => v.count > 0)
    .sort(([, a], [, b]) => b.count - a.count);

  return (
    <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
      <h3 className="font-bold text-gray-900 flex items-center gap-2 text-sm uppercase tracking-wide mb-1">
        <BarChart2 className="w-4 h-4 text-blue-600" /> Analisis Sentimen per Aspek Produk
      </h3>
      <p className="text-gray-500 text-xs mb-5 leading-relaxed">
        Setiap ulasan dicocokkan dengan aspek produk berdasarkan kata kunci yang muncul,
        kemudian dihitung distribusi sentimen positif dan negatifnya per aspek.
        Kolom "% dari keluhan" menunjukkan proporsi keluhan yang berkaitan dengan aspek tersebut.
      </p>
      <div className="space-y-4">
        {sorted.map(([key, data]) => {
          const negPct  = data.count > 0 ? ((data.negative / data.count) * 100).toFixed(0) : 0;
          const meta    = TOPIC_META[data.label] || {};
          const colors  = COLOR_MAP[meta.color || "blue"];
          const Icon    = meta.icon || BarChart2;
          return (
            <div key={key} className={`rounded-xl border p-4 ${colors.bg} ${colors.border}`}>
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Icon className={`w-4 h-4 flex-shrink-0 ${colors.icon}`} />
                  <span className={`text-sm font-bold ${colors.text}`}>{data.label}</span>
                </div>
                <div className="text-right text-xs ml-4 flex-shrink-0">
                  <div className="text-gray-600 font-medium">{data.count} ulasan</div>
                  {data.neg_share > 0 && (
                    <div className="text-red-500 font-semibold">{data.neg_share}% dari keluhan</div>
                  )}
                </div>
              </div>
              <div className="flex h-2.5 rounded-full overflow-hidden bg-gray-200 mb-2">
                <div className="bg-green-500 transition-all" style={{ width: `${data.pos_pct}%` }} />
                <div className="bg-red-400 transition-all"   style={{ width: `${negPct}%` }} />
              </div>
              <div className="flex justify-between text-xs font-semibold mb-2">
                <span className="text-green-600">{data.positive} positif ({data.pos_pct}%)</span>
                <span className={data.negative >= 3 ? "text-red-500" : "text-gray-400"}>
                  {data.negative} negatif ({negPct}%)
                </span>
              </div>
              {data.negative_keywords && data.negative_keywords.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  <span className="text-[10px] text-gray-500 font-medium mr-1">Kata keluhan:</span>
                  {data.negative_keywords.map((kw, i) => (
                    <span key={i} className="text-[10px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded border border-red-200">
                      {kw}
                    </span>
                  ))}
                </div>
              )}

            </div>
          );
        })}
      </div>
    </div>
  );
}



function SampleReviews({ samples }) {
  const [expanded, setExpanded] = useState(false);
  if (!samples) return null;
  const positives = samples.positive || [];
  const negatives = samples.negative || [];
  if (positives.length === 0 && negatives.length === 0) return null;

  return (
    <div className="mt-4 pt-4 border-t border-gray-100">
      <button onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-xs font-semibold text-gray-500 hover:text-gray-800 uppercase tracking-wide transition-colors">
        <Quote className="w-3.5 h-3.5" />
        Contoh Ulasan Nyata
        {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>
      {expanded && (
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
          {positives.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-green-600 mb-2">✓ Ulasan Positif</p>
              <div className="space-y-2">
                {positives.map((r, i) => (
                  <div key={i} className="bg-green-50 border border-green-100 rounded-lg p-3">
                    <p className="text-xs text-gray-700 leading-relaxed italic">"{r}"</p>
                  </div>
                ))}
              </div>
            </div>
          )}
          {negatives.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-red-500 mb-2">✗ Ulasan Negatif</p>
              <div className="space-y-2">
                {negatives.map((r, i) => (
                  <div key={i} className="bg-red-50 border border-red-100 rounded-lg p-3">
                    <p className="text-xs text-gray-700 leading-relaxed italic">"{r}"</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ResultPage() {
  const { id }   = useParams();
  const navigate = useNavigate();
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab]         = useState("overview");
  const [search, setSearch]   = useState("");
  const [filterSentiment, setFilterSentiment] = useState("all");
  const [filterCluster, setFilterCluster]     = useState("all");
  const [page, setPage] = useState(1);

  useEffect(() => {
    fetch(`${API}/results/${id}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [id]);

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3">
      <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-500 font-medium">Memuat hasil analisis...</span>
    </div>
  );

  if (!data || data.status !== "success") return (
    <div className="max-w-3xl mx-auto p-8 text-center bg-white rounded-lg border border-gray-200 mt-8">
      <AlertTriangle className="w-10 h-10 text-red-500 mx-auto mb-3" />
      <p className="text-gray-800 font-medium mb-4">Data analisis tidak ditemukan.</p>
      <button onClick={() => navigate("/")} className="text-blue-600 font-medium hover:underline text-sm flex items-center justify-center gap-1 mx-auto">
        <ArrowLeft className="w-4 h-4" /> Kembali ke Dashboard
      </button>
    </div>
  );

  const { session, clusters, results } = data;
  const topicOverview = session.topic_overview || {};
  const clusterList   = Object.values(clusters);
  const productName   = cleanProductName(session.filename);

  const positivePct = session.total_reviews > 0
    ? ((session.positive_count / session.total_reviews) * 100).toFixed(1) : 0;

  const pieData = [
    { name: "Positif", value: session.positive_count, color: "#22c55e" },
    { name: "Negatif", value: session.negative_count, color: "#ef4444" },
  ];

  const overviewBarData = Object.values(topicOverview)
    .filter(t => t.count > 0)
    .sort((a, b) => b.count - a.count)
    .map(t => ({ name: t.label, Positif: t.positive, Negatif: t.negative }));

  const filtered = results.filter(r => {
    const matchSearch    = !search || r.review_text.toLowerCase().includes(search.toLowerCase());
    const matchSentiment = filterSentiment === "all" || r.sentiment === filterSentiment;
    const matchCluster   = filterCluster === "all" || String(r.cluster_id) === filterCluster;
    return matchSearch && matchSentiment && matchCluster;
  });

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated  = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const insight    = generateInsight(session, topicOverview);

  const TABS = [
    { id: "overview",  label: "Ringkasan",                                icon: PieChartIcon  },
    { id: "aspect",    label: "Analisis Aspek",                           icon: BarChart2     },
    { id: "clusters",  label: `Eksplorasi Topik (${clusterList.length})`, icon: Search        },
    { id: "reviews",   label: `Daftar Ulasan (${session.total_reviews})`, icon: MessageSquare },
  ];

  return (
    <>
      {/* Background Wardah */}
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

      <div className="max-w-6xl mx-auto pb-10">
        <button onClick={() => navigate("/")} className="text-slate-600 hover:text-slate-900 text-sm mb-6 flex items-center gap-1.5 font-medium transition-colors">
          <ArrowLeft className="w-4 h-4" /> Kembali ke Dashboard
        </button>

        {/* Header */}
        <div className="bg-white/90 backdrop-blur-sm border border-sky-100 rounded-xl p-6 mb-6 shadow-sm">
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div className="min-w-0 flex-1">
              <h1 className="text-2xl font-bold text-gray-900 mb-1">{productName}</h1>
              <p className="text-xs text-gray-400 font-mono truncate max-w-xl" title={session.filename}>
                {session.filename}
              </p>
              <div className="flex items-center gap-3 flex-wrap mt-2">
                {session.brand_name && (
                  <span className="bg-blue-50 text-blue-700 text-xs px-3 py-1.5 rounded-md border border-blue-200 flex items-center gap-1.5 font-medium">
                    <Tag className="w-3.5 h-3.5" /> {session.brand_name}
                  </span>
                )}
                <span className="text-gray-500 text-sm flex items-center gap-1.5">
                  <Calendar className="w-4 h-4" />
                  {new Date(session.created_at).toLocaleDateString("id-ID", {
                    weekday: "long", day: "numeric", month: "long", year: "numeric"
                  })}
                </span>
                <button
                  onClick={() => window.open(`${API}/sessions/${session.id}/download-csv`, "_blank")}
                  className="ml-2 flex items-center gap-1.5 px-3 py-1.5 bg-green-50 text-green-700 border border-green-200 hover:bg-green-100 rounded-md text-xs font-semibold shadow-sm transition-all"
                >
                  <Download className="w-3.5 h-3.5" /> Download CSV Raw
                </button>
              </div>
            </div>
            <div className="text-right bg-gray-50 px-4 py-2 rounded-lg border border-gray-100">
              <div className={`text-3xl font-bold ${positivePct >= 70 ? "text-green-600" : "text-red-600"}`}>
                {positivePct}%
              </div>
              <div className="text-gray-500 text-xs font-medium uppercase tracking-wide mt-1">Skor Positif</div>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-4 mt-6 pt-6 border-t border-gray-100">
            {[
              { label: "Total Ulasan",     value: session.total_reviews,  color: "text-gray-900"  },
              { label: "Sentimen Positif", value: session.positive_count, color: "text-green-600" },
              { label: "Sentimen Negatif", value: session.negative_count, color: "text-red-600"   },
              { label: "Jumlah Cluster",   value: session.n_clusters,     color: "text-blue-600"  },
            ].map((s, i) => (
              <div key={i} className="text-center">
                <div className={`text-xl font-bold ${s.color}`}>{s.value.toLocaleString()}</div>
                <div className="text-gray-500 text-xs mt-1 font-medium">{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b border-gray-200 pb-px overflow-x-auto">
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                tab === t.id ? "border-blue-600 text-blue-600" : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              }`}>
              <t.icon className="w-4 h-4" /> {t.label}
            </button>
          ))}
        </div>

        {/* ── RINGKASAN ── */}
        {tab === "overview" && (
          <div className="space-y-6">
            {/* Ringkasan otomatis berbasis output klasifikasi sentimen */}
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-5 shadow-sm">
              <div className="flex items-start gap-3">
                <Lightbulb className="text-blue-600 w-6 h-6 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-blue-900 font-bold text-sm mb-1">Ringkasan Otomatis</p>
                  <p className="text-blue-800 text-sm leading-relaxed">
                    Berdasarkan analisis pada{" "}
                    <span className="font-bold">{session.total_reviews.toLocaleString()} ulasan</span>,
                    sentimen konsumen terpantau{" "}
                    <span className={positivePct >= 70 ? "text-green-700 font-bold" : "text-red-700 font-bold"}>
                      {insight.sentiment_text}
                    </span>.{" "}
                    {insight.topText} {insight.complaint_text}
                  </p>
                  <button onClick={() => setTab("aspect")} className="mt-3 text-blue-600 font-medium text-xs hover:underline flex items-center gap-1">
                    Lihat analisis per aspek produk <ArrowLeft className="w-3 h-3 rotate-180" />
                  </button>
                </div>
              </div>
            </div>

            {/* Grafik distribusi sentimen dan kinerja per aspek */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
                <h3 className="font-bold text-gray-900 mb-4 text-sm uppercase tracking-wide">Distribusi Sentimen</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" innerRadius={70} outerRadius={100} paddingAngle={2} dataKey="value">
                      {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                    </Pie>
                    <Tooltip contentStyle={{ borderRadius: "8px", border: "1px solid #e5e7eb" }}
                      formatter={(v, n) => [v.toLocaleString() + " ulasan", n]} />
                    <Legend iconType="circle" wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
                <h3 className="font-bold text-gray-900 mb-4 text-sm uppercase tracking-wide">Sentimen per Aspek Produk</h3>
                {overviewBarData.length === 0 ? (
                  <div className="h-[250px] flex items-center justify-center text-gray-400 text-sm">
                    Tidak ada data aspek tersedia
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={overviewBarData} layout="vertical" margin={{ left: 0, right: 16 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e5e7eb" />
                      <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis type="category" dataKey="name" width={145} tick={{ fontSize: 10, fontWeight: 500, fill: "#374151" }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ borderRadius: "8px", border: "1px solid #e5e7eb" }} cursor={{ fill: "#f3f4f6" }} />
                      <Bar dataKey="Positif" fill="#22c55e" stackId="a" radius={[0, 2, 2, 0]} />
                      <Bar dataKey="Negatif" fill="#ef4444" stackId="a" radius={[0, 2, 2, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── ANALISIS ASPEK ── */}
        {tab === "aspect" && (
          <div className="space-y-6">
            {/* Interpretasi sentimen keseluruhan */}
            <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
              <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Lightbulb className="w-5 h-5 text-yellow-500" /> Interpretasi Sentimen Keseluruhan
              </h3>
              <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                <p className="text-gray-700 text-sm leading-relaxed">
                  Dari{" "}
                  <span className="font-bold text-gray-900">{session.total_reviews.toLocaleString()} ulasan</span>{" "}
                  konsumen terhadap produk{" "}
                  <span className="font-bold text-blue-600">{productName}</span>,
                  sebanyak{" "}
                  <span className="font-bold text-green-600">{session.positive_count.toLocaleString()} ulasan ({positivePct}%) bersentimen positif</span>{" "}
                  dan{" "}
                  <span className="font-bold text-red-600">{session.negative_count.toLocaleString()} ulasan bersentimen negatif</span>.
                  {positivePct >= 80
                    ? " Secara keseluruhan konsumen menunjukkan kepuasan yang tinggi terhadap produk ini."
                    : positivePct >= 60
                    ? " Secara keseluruhan ulasan positif mendominasi meskipun ada beberapa keluhan."
                    : positivePct >= 40
                    ? " Ulasan positif dan negatif terbagi cukup berimbang pada dataset ini."
                    : " Ulasan negatif mendominasi pada dataset ini."}
                </p>
                {session.negative_count > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs text-gray-500 font-medium">
                      <span className="font-semibold text-gray-700">Distribusi keluhan:</span>{" "}
                      {insight.complaint_text}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Analisis per aspek — output dari Python */}
            {Object.keys(topicOverview).length > 0
              ? <TopicOverviewCard topicOverview={topicOverview} />
              : (
                <div className="text-center py-10 bg-white/90 backdrop-blur-sm rounded-xl border border-sky-100">
                  <p className="text-gray-500 text-sm">Tidak ada data aspek terdeteksi pada dataset ini.</p>
                </div>
              )
            }
          </div>
        )}

        {/* ── EKSPLORASI TOPIK ── */}
        {tab === "clusters" && (
          <div className="space-y-4">
            {/* Silhouette Score — metrik evaluasi kualitas clustering */}
            {session.silhouette_score && (
              <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-5">
                <div className="flex items-center gap-3 flex-wrap">
                  <div>
                    <span className="text-xs text-gray-500 font-medium">Silhouette Score</span>
                    <span className="ml-2 text-sm font-bold text-gray-900 font-mono">
                      {Number(session.silhouette_score).toFixed(4)}
                    </span>
                  </div>
                  <span className="text-gray-200">|</span>
                  <span className="text-xs text-gray-500">
                    {clusterList.length} cluster terbentuk dari {session.total_reviews.toLocaleString()} ulasan
                  </span>
                </div>
              </div>
            )}
            {clusterList.map((cluster, i) => {
              const pct  = cluster.count > 0 ? ((cluster.positive / cluster.count) * 100).toFixed(1) : 0;
              const name = clusterName(cluster.top_keywords);
              return (
                <div key={i} className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="font-bold text-gray-900 text-lg flex items-center gap-2">
                        Cluster {cluster.cluster_id}
                        <span className="text-gray-400 font-normal">|</span>
                        <span className="text-blue-600">{name}</span>
                      </h3>
                      <p className="text-gray-500 text-sm mt-1">
                        {cluster.count.toLocaleString()} ulasan
                        {cluster.avg_confidence && ` · confidence rata-rata ${(cluster.avg_confidence * 100).toFixed(0)}%`}
                      </p>
                    </div>
                    <span className={`text-sm font-bold px-3 py-1 rounded-full border ${
                      pct >= 50 ? "bg-green-50 text-green-700 border-green-200" : "bg-red-50 text-red-700 border-red-200"
                    }`}>
                      {pct}% Positif
                    </span>
                  </div>
                  <div className="flex h-2 rounded-full overflow-hidden mb-4 bg-gray-100">
                    <div className="bg-green-500" style={{ width: `${pct}%` }} />
                    <div className="bg-red-500"   style={{ width: `${100 - pct}%` }} />
                  </div>
                  <p className="text-gray-500 text-xs font-semibold mb-2 uppercase tracking-wide">Kata Kunci Pembentuk Topik</p>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {(cluster.top_keywords || [])
                      .filter(kw => !kw.includes("_") && !kw.startsWith("tidak") && !CLUSTER_SKIP.has(kw))
                      .map((kw, j) => (
                        <span key={j} className="bg-gray-100 text-gray-700 text-xs px-2.5 py-1 rounded border border-gray-200">{kw}</span>
                      ))}
                  </div>
                  <button
                    onClick={() => { setTab("reviews"); setFilterCluster(String(cluster.cluster_id)); setPage(1); }}
                    className="mt-4 text-sm font-medium text-blue-600 hover:text-blue-800 hover:underline flex items-center gap-1"
                  >
                    Inspeksi semua ulasan cluster ini <ArrowLeft className="w-3 h-3 rotate-180" />
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {/* ── DAFTAR ULASAN ── */}
        {tab === "reviews" && (
          <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
            <div className="flex flex-wrap gap-3 mb-6">
              <input type="text" placeholder="Cari teks ulasan..." value={search}
                onChange={e => { setSearch(e.target.value); setPage(1); }}
                className="flex-1 min-w-[200px] bg-white border border-gray-300 rounded-md px-3 py-2 text-gray-900 text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500" />
              <select value={filterSentiment} onChange={e => { setFilterSentiment(e.target.value); setPage(1); }}
                className="bg-white border border-gray-300 rounded-md px-3 py-2 text-gray-900 text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500">
                <option value="all">Semua Sentimen</option>
                <option value="positive">Prediksi Positif</option>
                <option value="negative">Prediksi Negatif</option>
              </select>
              <select value={filterCluster} onChange={e => { setFilterCluster(e.target.value); setPage(1); }}
                className="bg-white border border-gray-300 rounded-md px-3 py-2 text-gray-900 text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500">
                <option value="all">Semua Cluster</option>
                {clusterList.map((c, i) => (
                  <option key={i} value={String(c.cluster_id)}>
                    Cluster {c.cluster_id}: {clusterName(c.top_keywords)}
                  </option>
                ))}
              </select>
            </div>

            <p className="text-gray-500 text-sm mb-4 font-medium">{filtered.length.toLocaleString()} data ditemukan</p>

            <div className="overflow-x-auto rounded-lg border border-gray-200 mb-4">
              <table className="w-full text-sm text-left">
                <thead className="bg-gray-50 text-gray-700 text-xs uppercase">
                  <tr>
                    <th className="px-4 py-3 font-semibold">Teks Ulasan Asli</th>
                    <th className="px-4 py-3 font-semibold text-center w-32">Sentimen</th>
                    <th className="px-4 py-3 font-semibold text-center w-28">Confidence</th>
                    <th className="px-4 py-3 font-semibold text-center w-36">Cluster</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {paginated.length === 0 ? (
                    <tr><td colSpan={4} className="text-center py-8 text-gray-500">Tidak ada ulasan yang cocok.</td></tr>
                  ) : paginated.map((r, i) => (
                    <tr key={i} className="hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-3"><p className="text-gray-900 leading-relaxed">{r.review_text}</p></td>
                      <td className="px-4 py-3 text-center">
                        <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${
                          r.sentiment === "positive"
                            ? "bg-green-50 text-green-700 border-green-200"
                            : "bg-red-50 text-red-700 border-red-200"
                        }`}>
                          {r.sentiment === "positive" ? "Positif" : "Negatif"}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center text-gray-600 font-medium">
                        {r.confidence ? `${(r.confidence * 100).toFixed(0)}%` : "—"}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className="text-xs text-gray-600 font-medium px-2 py-1 bg-gray-100 rounded border border-gray-200">
                          {clusterName(clusters[r.cluster_id]?.top_keywords)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-1 mt-6">
                <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}
                  className="px-3 py-1.5 rounded text-sm font-medium text-gray-600 border border-gray-300 hover:bg-gray-50 disabled:opacity-50 transition-colors">
                  Prev
                </button>
                {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                  const p = totalPages <= 7 ? i + 1 : page <= 4 ? i + 1 : page >= totalPages - 3 ? totalPages - 6 + i : page - 3 + i;
                  return (
                    <button key={p} onClick={() => setPage(p)}
                      className={`w-8 h-8 rounded text-sm font-medium border transition-colors ${
                        page === p ? "bg-blue-600 text-white border-blue-600" : "text-gray-600 border-gray-300 hover:bg-gray-50"
                      }`}>
                      {p}
                    </button>
                  );
                })}
                <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages}
                  className="px-3 py-1.5 rounded text-sm font-medium text-gray-600 border border-gray-300 hover:bg-gray-50 disabled:opacity-50 transition-colors">
                  Next
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}
