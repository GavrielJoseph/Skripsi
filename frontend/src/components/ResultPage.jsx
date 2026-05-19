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
  ChevronDown, ChevronUp, Quote, Info, TrendingDown, Download,
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
  "Harga & Kualitas":         { icon: Banknote,     color: "yellow" },
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

function generateInsight(session, topicOverview) {
  const total  = session.total_reviews;
  const pos    = session.positive_count;
  const posPct = ((pos / total) * 100).toFixed(1);

  let sentiment_text = "";
  if      (posPct >= 80) sentiment_text = `sangat positif (${posPct}% ulasan bernada baik)`;
  else if (posPct >= 60) sentiment_text = `cukup baik (${posPct}% ulasan positif)`;
  else if (posPct >= 40) sentiment_text = `terbagi (campuran positif dan negatif)`;
  else                   sentiment_text = `cenderung negatif (hanya ${posPct}% ulasan positif)`;

  const topics = Object.values(topicOverview || {}).filter(t => t.count > 0);
  topics.sort((a, b) => b.count - a.count);

  const topDiscussed = topics.slice(0, 2).map(t => t.label).join(" dan ");
  const topText = topDiscussed ? `Aspek yang paling banyak dibahas konsumen adalah "${topDiscussed}".` : "";

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

function generateRecommendations(topicOverview, totalNegative) {
  const topics = Object.values(topicOverview || {}).filter(t => t.count > 0);
  const recs   = [];

  const tindakanMap = {
    "Efek & Kecocokan Produk":
      "Perhatikan tipe kulit kamu sebelum membeli — baca ulasan dari konsumen dengan tipe kulit serupa. " +
      "Beberapa konsumen melaporkan ketidakcocokan, jadi pastikan produk ini sesuai kebutuhan kulitmu. " +
      "Jika ragu, cari varian ukuran kecil atau tester terlebih dahulu.",
    "Layanan Pengiriman":
      "Keluhan pengiriman umumnya bukan dari produknya, melainkan dari jasa kurir. " +
      "Pertimbangkan membeli dari toko yang memberikan opsi pilihan ekspedisi sendiri, " +
      "atau pilih toko dengan fitur asuransi pengiriman untuk keamanan lebih.",
    "Harga & Kualitas":
      "Beberapa konsumen menganggap harga kurang sebanding dengan isi/ukuran produk. " +
      "Bandingkan harga per ml/gram dengan produk sejenis sebelum membeli. " +
      "Tunggu promo atau beli di flash sale untuk harga lebih terjangkau.",
    "Kondisi Kemasan":
      "Ada laporan kemasan yang rusak saat diterima. " +
      "Pastikan memilih toko yang mengemas dengan baik (baca ulasan foto produk yang diterima). " +
      "Aktifkan asuransi pengiriman saat checkout untuk perlindungan jika terjadi kerusakan.",
    "Layanan Toko & Keaslian":
      "Pastikan membeli dari toko resmi atau official store yang terverifikasi di Tokopedia " +
      "untuk menghindari risiko produk tidak sesuai atau diragukan keasliannya. " +
      "Cek badge 'Official Store' dan pastikan ada garansi original.",
  };

  topics.forEach(t => {
    const negPct = (t.negative / t.count) * 100;
    const posPct = (t.positive / t.count) * 100;
    const meta   = TOPIC_META[t.label] || {};

    if (negPct >= 30 && t.negative >= 5) {
      recs.push({
        type:      negPct >= 50 ? "danger" : "warning",
        icon:      meta.icon || AlertTriangle,
        label:     t.label,
        judul:     `Perhatikan Aspek: ${t.label}`,
        masalah:   `${t.negative} dari ${t.count} konsumen (${negPct.toFixed(0)}%) memberikan ulasan negatif pada aspek ini.${t.negative_keywords?.length > 0 ? ` Kata yang sering muncul: "${t.negative_keywords.slice(0,3).join('", "')}"` : ""}`,
        tindakan:  tindakanMap[t.label] || "Baca ulasan negatif secara langsung sebelum memutuskan pembelian.",
        stats:     { total: t.count, posPct: t.pos_pct, negPct: negPct.toFixed(0) },
        samples:   t.negative_samples || [],
        isRelative: false,
        isFallback: false,
      });
    }

    if (posPct >= 80 && t.count >= 10) {
      recs.push({
        type:      "success",
        icon:      meta.icon || CheckCircle2,
        label:     t.label,
        judul:     `Disukai Konsumen: ${t.label}`,
        masalah:   `${t.positive} dari ${t.count} konsumen (${posPct.toFixed(0)}%) memberikan ulasan positif tentang aspek ini — aspek terkuat produk ini berdasarkan data ulasan.`,
        tindakan:  `Aspek "${t.label}" konsisten mendapat pujian dari konsumen. Ini dapat dijadikan pertimbangan utama bahwa produk ini terbukti baik pada aspek tersebut.`,
        stats:     { total: t.count, posPct: t.pos_pct, negPct: (100 - t.pos_pct).toFixed(0) },
        samples:   [],
        isRelative: false,
        isFallback: false,
      });
    }
  });

  const hasSignificantNeg = recs.some(r => r.type === "danger" || r.type === "warning");

  if (totalNegative > 0) {
    const topByShare = topics
      .filter(t => t.negative > 0 && t.neg_share > 0)
      .sort((a, b) => (b.neg_share || 0) - (a.neg_share || 0))
      .slice(0, 3);

    if (topByShare.length > 0 && !hasSignificantNeg) {
      topByShare.forEach(t => {
        const meta   = TOPIC_META[t.label] || {};
        const negPct = ((t.negative / t.count) * 100).toFixed(0);
        recs.push({
          type: "info", icon: meta.icon || TrendingDown, label: t.label,
          judul:   `${t.neg_share}% Keluhan Berkaitan dengan ${t.label}`,
          masalah: `Dari total ${totalNegative} ulasan negatif, ${t.neg_share}% (${t.negative} ulasan) berkaitan dengan ${t.label.toLowerCase()}. Proporsi negatif dalam aspek ini: ${negPct}%.${t.negative_keywords?.length > 0 ? ` Kata yang sering muncul: "${t.negative_keywords.slice(0,3).join('", "')}"` : ""}`,
          tindakan: tindakanMap[t.label] || "Perhatikan aspek ini sebelum memutuskan pembelian.",
          stats: { total: t.count, posPct: t.pos_pct, negPct },
          samples: t.negative_samples || [], isRelative: true, isFallback: false,
        });
      });
    } else if (topByShare.length > 0 && hasSignificantNeg) {
      topByShare.slice(0, 2).forEach(t => {
        const alreadyAdded = recs.some(r => r.label === t.label && !r.isFallback);
        if (!alreadyAdded && t.neg_share >= 10) {
          const meta   = TOPIC_META[t.label] || {};
          const negPct = ((t.negative / t.count) * 100).toFixed(0);
          recs.push({
            type: "info", icon: meta.icon || TrendingDown, label: t.label,
            judul:   `Pantau: ${t.label} (${t.neg_share}% dari total keluhan)`,
            masalah: `${t.negative} ulasan negatif tentang aspek ini menyumbang ${t.neg_share}% dari total keluhan.${t.negative_keywords?.length > 0 ? ` Kata yang sering muncul: "${t.negative_keywords.slice(0,3).join('", "')}"` : ""}`,
            tindakan: tindakanMap[t.label] || "Perhatikan aspek ini sebelum memutuskan pembelian.",
            stats: { total: t.count, posPct: t.pos_pct, negPct },
            samples: t.negative_samples || [], isRelative: true, isFallback: false,
          });
        }
      });
    }
  }

  const order = { danger: 0, warning: 1, info: 2, success: 3 };
  return recs.sort((a, b) => order[a.type] - order[b.type]).slice(0, 7);
}

function RecommendationCard({ rec }) {
  const [expanded, setExpanded]       = useState(false);
  const [showSamples, setShowSamples] = useState(false);

  const styleMap = {
    danger:  { card: "border-red-200 bg-red-50",    badge: "bg-red-100 text-red-700 border-red-200",       badgeText: "Perlu Diperhatikan" },
    warning: { card: "border-amber-200 bg-amber-50", badge: "bg-amber-100 text-amber-700 border-amber-200", badgeText: "Cukup Diperhatikan" },
    success: { card: "border-green-200 bg-green-50", badge: "bg-green-100 text-green-700 border-green-200", badgeText: "Nilai Plus Produk" },
    info:    { card: "border-blue-200 bg-blue-50",   badge: "bg-blue-100 text-blue-700 border-blue-200",   badgeText: rec.isRelative ? "Distribusi Keluhan" : "Perlu Dipantau" },
  };
  const style = styleMap[rec.type] || styleMap.info;
  const Icon  = rec.icon;

  return (
    <div className={`rounded-xl border p-5 ${style.card}`}>
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-full bg-white/70 shadow-sm mt-0.5 flex-shrink-0">
          <Icon className="w-5 h-5 text-gray-700" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-2">
            <h4 className="font-bold text-gray-900 text-sm leading-snug">{rec.judul}</h4>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded border flex-shrink-0 ${style.badge}`}>{style.badgeText}</span>
          </div>
          <div className="flex items-center gap-3 mb-3">
            <div className="flex h-1.5 flex-1 rounded-full overflow-hidden bg-gray-200">
              <div className="bg-green-500" style={{ width: `${rec.stats.posPct}%` }} />
              <div className="bg-red-400"   style={{ width: `${rec.stats.negPct}%` }} />
            </div>
            <span className="text-xs text-gray-500 flex-shrink-0">{rec.stats.total} ulasan</span>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed mb-3">{rec.masalah}</p>
          <div className="flex flex-wrap gap-3">
            <button onClick={() => setExpanded(!expanded)}
              className="flex items-center gap-1.5 text-xs font-semibold text-gray-600 hover:text-gray-900 transition-colors">
              <Lightbulb className="w-3.5 h-3.5 text-yellow-500" />
              {expanded ? "Sembunyikan saran" : "Lihat saran pembelian"}
              {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
            {rec.samples && rec.samples.length > 0 && (
              <button onClick={() => setShowSamples(!showSamples)}
                className="flex items-center gap-1.5 text-xs font-semibold text-red-500 hover:text-red-700 transition-colors">
                <Quote className="w-3.5 h-3.5" />
                {showSamples ? "Sembunyikan ulasan" : `Lihat ${rec.samples.length} ulasan negatif nyata`}
                {showSamples ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
            )}
          </div>
          {expanded && (
            <div className="mt-3 p-3 bg-white/70 rounded-lg border border-white/50 text-sm text-gray-700 leading-relaxed">
              <span className="font-semibold block mb-1">💡 Saran sebelum membeli:</span>
              {rec.tindakan}
            </div>
          )}
          {showSamples && rec.samples.length > 0 && (
            <div className="mt-3 space-y-2">
              <p className="text-xs font-semibold text-red-500">Ulasan negatif nyata dari konsumen:</p>
              {rec.samples.map((s, i) => (
                <div key={i} className="bg-white/70 border border-red-100 rounded-lg p-3">
                  <p className="text-xs text-gray-700 leading-relaxed italic">"{s}"</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function TopicOverviewCard({ topicOverview, totalNegative }) {
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
        Setiap ulasan dianalisis untuk mendeteksi aspek produk yang dibahas konsumen.
        Kolom "% dari keluhan" menunjukkan seberapa besar proporsi keluhan yang berkaitan dengan aspek tersebut.
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
                  {data.neg_share > 0 && <div className="text-red-500 font-semibold">{data.neg_share}% dari keluhan</div>}
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
                  <span className="text-[10px] text-gray-500 font-medium mr-1">Keluhan:</span>
                  {data.negative_keywords.map((kw, i) => (
                    <span key={i} className="text-[10px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded border border-red-200">{kw}</span>
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

  const totalPages      = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated       = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const insight         = generateInsight(session, topicOverview);
  const recommendations = generateRecommendations(topicOverview, session.negative_count);

  const TABS = [
    { id: "overview",  label: "Ringkasan Eksekutif",                      icon: PieChartIcon  },
    { id: "insight",   label: "Insight",                    icon: Lightbulb     },
    { id: "clusters",  label: `Eksplorasi Topik (${clusterList.length})`, icon: Search        },
    { id: "reviews",   label: `Daftar Ulasan (${session.total_reviews})`,   icon: MessageSquare },
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

      {/* ── OVERVIEW ── */}
      {tab === "overview" && (
        <div className="space-y-6">
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-5 shadow-sm">
            <div className="flex items-start gap-3">
              <Lightbulb className="text-blue-600 w-6 h-6 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-blue-900 font-bold text-sm mb-1">Ringkasan Otomatis</p>
                <p className="text-blue-800 text-sm leading-relaxed">
                  Berdasarkan analisis NLP pada{" "}
                  <span className="font-bold">{session.total_reviews.toLocaleString()} ulasan</span>,
                  sentimen konsumen terpantau{" "}
                  <span className={positivePct >= 70 ? "text-green-700 font-bold" : "text-red-700 font-bold"}>
                    {insight.sentiment_text}
                  </span>.{" "}
                  {insight.topText} {insight.complaint_text}
                </p>
                <button onClick={() => setTab("insight")} className="mt-3 text-blue-600 font-medium text-xs hover:underline flex items-center gap-1">
                  Lihat insight & rekomendasi lengkap <ArrowLeft className="w-3 h-3 rotate-180" />
                </button>
              </div>
            </div>
          </div>

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
              <h3 className="font-bold text-gray-900 mb-4 text-sm uppercase tracking-wide">Kinerja per Aspek Produk</h3>
              {overviewBarData.length === 0 ? (
                <div className="h-[250px] flex items-center justify-center text-gray-400 text-sm">
                  Jalankan ulang analisis untuk melihat data aspek
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

          {Object.keys(topicOverview).length > 0 && (
            <TopicOverviewCard topicOverview={topicOverview} totalNegative={session.negative_count} />
          )}
        </div>
      )}

      {/* ── INSIGHT & REKOMENDASI ── */}
      {tab === "insight" && (
        <div className="space-y-6">
          <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
            <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-yellow-500" /> Interpretasi Sentimen Sentral
            </h3>
            <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
              <p className="text-gray-700 text-sm leading-relaxed">
                Dari total{" "}
                <span className="font-bold text-gray-900">{session.total_reviews.toLocaleString()} ulasan</span>{" "}
                konsumen terhadap produk{" "}
                <span className="font-bold text-blue-600">{productName}</span>,
                model Machine Learning mendeteksi{" "}
                <span className="font-bold">{positivePct}% ulasan mengandung sentimen positif</span>{" "}
                dan <span className="font-bold text-red-600">{session.negative_count} ulasan negatif</span>.
                {positivePct >= 80
                  ? " Tingkat kepuasan konsumen sangat tinggi. Keluhan yang ada bersifat minor dan tersebar di berbagai aspek."
                  : positivePct >= 60
                  ? " Kepuasan konsumen cukup baik namun ada beberapa aspek yang perlu diperhatikan sebelum membeli."
                  : positivePct >= 40
                  ? " Sentimen terbagi — ada aspek bermasalah yang perlu dicermati sebelum memutuskan pembelian."
                  : " Mayoritas konsumen tidak puas. Pertimbangkan dengan cermat sebelum membeli."}
              </p>
              {session.negative_count > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <p className="text-xs text-gray-500 font-medium">
                    <span className="font-semibold text-gray-700">Tentang {session.negative_count} ulasan negatif:</span>{" "}
                    {insight.complaint_text}
                  </p>
                </div>
              )}
            </div>
          </div>

          {Object.keys(topicOverview).length > 0 && (
            <TopicOverviewCard topicOverview={topicOverview} totalNegative={session.negative_count} />
          )}

          <div className="bg-white/90 backdrop-blur-sm border border-sky-100 shadow-sm rounded-xl p-6">
            <h3 className="font-bold text-gray-900 mb-2 flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-yellow-500" /> Pertimbangan Sebelum Membeli
            </h3>
            <div className="flex items-start gap-2 p-3 bg-gray-50 border border-gray-200 rounded-lg mb-5">
              <Info className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-gray-600 leading-relaxed">
                <span className="font-semibold text-red-600">Perlu Diperhatikan</span> → aspek dengan ≥30% ulasan negatif dari konsumen.{" "}
                <span className="font-semibold text-blue-600">Distribusi Keluhan</span> → aspek mana yang paling sering dikeluhkan dari total ulasan negatif.{" "}
                <span className="font-semibold text-green-600">Nilai Plus Produk</span> → aspek yang paling banyak dipuji konsumen.
              </p>
            </div>
            {recommendations.length === 0 ? (
              <div className="text-center py-10 bg-gray-50 rounded-lg border border-dashed border-gray-300">
                <CheckCircle2 className="w-8 h-8 text-green-400 mx-auto mb-3" />
                <p className="text-gray-700 text-sm font-semibold">Tidak ditemukan keluhan signifikan</p>
                <p className="text-gray-500 text-xs mt-1">Semua aspek produk menunjukkan sentimen yang sangat baik dari konsumen.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {recommendations.map((rec, i) => <RecommendationCard key={i} rec={rec} />)}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── EKSPLORASI TOPIK ── */}
      {tab === "clusters" && (
        <div className="space-y-4">
          <div className="bg-white/90 backdrop-blur-sm border border-sky-100 rounded-lg p-5 mb-4">
            <p className="text-gray-700 text-sm leading-relaxed">
              {session.silhouette_score && (
                <span className="block mt-1 text-gray-500 text-xs">
                  Skor Silhouette: <span className="font-bold text-gray-700">{Number(session.silhouette_score).toFixed(4)}</span>
                </span>
              )}
            </p>
          </div>

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
                {cluster.sample_reviews && <SampleReviews samples={cluster.sample_reviews} />}
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
