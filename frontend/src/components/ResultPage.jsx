import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";
import { ArrowLeft, Tag, Calendar, Lightbulb, PieChart as PieChartIcon, MessageSquare, BarChart2, CheckCircle2, AlertTriangle, Truck, FlaskConical, Banknote, Package, ShieldCheck, ClipboardList } from "lucide-react";

const API = "http://127.0.0.1:8000/api";
const PAGE_SIZE = 10;

// Filter nama cluster
const ENGLISH_SKIP = new Set([
  "good","and","like","the","this","very","is","it","so","for","but",
  "cause","not","bad","great","love","nice","best","well","just","ok",
  "me","my","wow","yes","no","too","also","get","got","use","used","sok"
]);

function clusterName(keywords) {
  if (!keywords || keywords.length === 0) return "Topik Umum";
  const filtered = keywords.filter(k => !ENGLISH_SKIP.has(k));
  const top = (filtered.length > 0 ? filtered : keywords).slice(0, 2);
  return top.map(k => k.charAt(0).toUpperCase() + k.slice(1)).join(" & ");
}

function generateInsight(session, clusters) {
  const total  = session.total_reviews;
  const pos    = session.positive_count;
  const posPct = ((pos / total) * 100).toFixed(1);
  const clusterList = Object.values(clusters);

  let sentiment_text = "";
  if      (posPct >= 80) sentiment_text = `sangat positif (${posPct}% ulasan positif)`;
  else if (posPct >= 60) sentiment_text = `cukup positif (${posPct}% ulasan positif)`;
  else if (posPct >= 40) sentiment_text = `campuran (${posPct}% positif, ${(100 - posPct).toFixed(1)}% negatif)`;
  else                   sentiment_text = `cenderung negatif (hanya ${posPct}% ulasan positif)`;

  const negClusters = clusterList
    .filter(c => c.count > 5)
    .sort((a, b) => (b.negative / b.count) - (a.negative / a.count))
    .slice(0, 2);

  const posClusters = clusterList
    .filter(c => c.count > 5)
    .sort((a, b) => (b.positive / b.count) - (a.positive / a.count))
    .slice(0, 1);

  const cluster_text = negClusters.length > 0
    ? `Topik yang paling banyak mendapat keluhan adalah "${negClusters.map(c => clusterName(c.top_keywords)).join('" dan "')}".`
    : "";

  const positive_text = posClusters.length > 0
    ? `Aspek yang paling dipuji konsumen adalah topik "${clusterName(posClusters[0].top_keywords)}".`
    : "";

  return { sentiment_text, cluster_text, positive_text, posPct };
}

const TOPIC_DETECT = {
  pengiriman: ["kirim","sampai","kurir","paket","ekspedisi","datang","pengiriman","lambat","lama","ongkir","ongkos","resi","jnt","jne","sicepat","gosend"],
  efek_produk: ["jerawat","bruntusan","iritasi","gatal","perih","alergi","kulit","wajah","lembap","kering","cerah","cocok","formula","bahan"],
  harga: ["harga","mahal","murah","worth","sepadan","promo","diskon","hemat","budget","sale"],
  kemasan: ["kemasan","packaging","packing","botol","tube","pot","bocor","pecah","rusak","segel"],
  pembelian: ["palsu","kw","ori","original","asli","seller","toko","reseller","fast","respon"],
};

function detectTopic(keywords) {
  if (!keywords) return "umum";
  for (const [topic, words] of Object.entries(TOPIC_DETECT)) {
    if (keywords.some(k => words.includes(k))) return topic;
  }
  return "umum";
}

function generateRecommendations(clusters) {
  const clusterList = Object.values(clusters);
  const recs = [];

  clusterList.forEach(c => {
    const negPct  = c.count > 0 ? (c.negative / c.count) * 100 : 0;
    const posPct  = c.count > 0 ? (c.positive / c.count) * 100 : 0;
    const keywords = c.top_keywords || [];
    const name    = clusterName(keywords);
    const topic   = detectTopic(keywords);

    if (negPct >= 40) {
      const templates = {
        pengiriman: { icon: Truck, type: "warning", title: "Evaluasi Layanan Pengiriman", desc: `${c.negative} dari ${c.count} ulasan mengeluhkan pengiriman. Evaluasi mitra logistik.` },
        efek_produk: { icon: FlaskConical, type: "danger", title: "Perhatikan Efek Produk", desc: `${c.negative} ulasan menyebutkan masalah kecocokan di kulit (iritasi/breakout).` },
        harga: { icon: Banknote, type: "warning", title: "Evaluasi Strategi Harga", desc: `Konsumen merasa harga kurang sepadan dengan produk yang didapatkan.` },
        kemasan: { icon: Package, type: "warning", title: "Perbaiki Standar Pengemasan", desc: `${c.negative} ulasan mengeluhkan kondisi kemasan (bocor/rusak).` },
        pembelian: { icon: ShieldCheck, type: "warning", title: "Waspadai Isu Keaslian", desc: `Terdapat ulasan yang mempertanyakan keaslian produk dari seller.` },
        umum: { icon: ClipboardList, type: "info", title: `Tindak Lanjuti Keluhan: ${name}`, desc: `${c.negative} ulasan negatif pada topik ini perlu diperhatikan manual.` },
      };
      const t = templates[topic] || templates.umum;
      recs.push({ ...t, cluster: name, count: c.negative });
    } else if (posPct >= 80 && c.count >= 10) {
      recs.push({ icon: CheckCircle2, type: "success", title: `Pertahankan: ${name}`, desc: `${c.positive} dari ${c.count} ulasan memuji aspek ini. Jadikan fokus marketing.`, cluster: name, count: c.positive });
    }
  });

  return recs.slice(0, 5);
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

  const positivePct = session.total_reviews > 0 ? ((session.positive_count / session.total_reviews) * 100).toFixed(1) : 0;

  const pieData = [
    { name: "Positif", value: session.positive_count, color: "#22c55e" },
    { name: "Negatif", value: session.negative_count, color: "#ef4444" },
  ];

  const clusterList    = Object.values(clusters);
  const clusterBarData = clusterList.map(c => ({
    name:    clusterName(c.top_keywords),
    Positif: c.positive,
    Negatif: c.negative,
  }));

  const filtered = results.filter(r => {
    const matchSearch    = !search || r.review_text.toLowerCase().includes(search.toLowerCase());
    const matchSentiment = filterSentiment === "all" || r.sentiment === filterSentiment;
    const matchCluster   = filterCluster === "all" || String(r.cluster_id) === filterCluster;
    return matchSearch && matchSentiment && matchCluster;
  });

  const totalPages      = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated       = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const insight         = generateInsight(session, clusters);
  const recommendations = generateRecommendations(clusters);

  const TABS = [
    { id: "overview", label: "Ringkasan Eksekutif", icon: PieChartIcon },
    { id: "insight",  label: "Insight & Rekomendasi", icon: Lightbulb },
    { id: "clusters", label: `Topik Cluster (${clusterList.length})`, icon: BarChart2 },
    { id: "reviews",  label: `Data Ulasan (${session.total_reviews})`, icon: MessageSquare },
  ];

  const recStyle = {
    warning: "border-yellow-200 bg-yellow-50 text-yellow-800",
    danger:  "border-red-200 bg-red-50 text-red-800",
    success: "border-green-200 bg-green-50 text-green-800",
    info:    "border-blue-200 bg-blue-50 text-blue-800"
  };

  return (
    <div className="max-w-6xl mx-auto">
      <button onClick={() => navigate("/")} className="text-gray-500 hover:text-gray-900 text-sm mb-6 flex items-center gap-1.5 font-medium transition-colors">
        <ArrowLeft className="w-4 h-4" /> Kembali ke Dashboard
      </button>

      {/* Header Info Sesi */}
      <div className="bg-white border border-gray-200 rounded-xl p-6 mb-6 shadow-sm">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">{session.filename}</h1>
            <div className="flex items-center gap-4 flex-wrap">
              {session.brand_name && (
                <span className="bg-blue-50 text-blue-700 text-xs px-3 py-1 rounded-md border border-blue-200 flex items-center gap-1.5 font-medium">
                  <Tag className="w-3.5 h-3.5" /> {session.brand_name}
                </span>
              )}
              <span className="text-gray-500 text-sm flex items-center gap-1.5">
                <Calendar className="w-4 h-4" />
                {new Date(session.created_at).toLocaleDateString("id-ID", {
                  weekday:"long", day:"numeric", month:"long", year:"numeric"
                })}
              </span>
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
            { label: "Total Ulasan", value: session.total_reviews, color: "text-gray-900" },
            { label: "Sentimen Positif", value: session.positive_count, color: "text-green-600" },
            { label: "Sentimen Negatif", value: session.negative_count, color: "text-red-600" },
            { label: "Jumlah Cluster", value: session.n_clusters, color: "text-blue-600" },
          ].map((s, i) => (
            <div key={i} className="text-center">
              <div className={`text-xl font-bold ${s.color}`}>{s.value.toLocaleString()}</div>
              <div className="text-gray-500 text-xs mt-1 font-medium">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="flex gap-2 mb-6 border-b border-gray-200 pb-px overflow-x-auto">
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
              tab === t.id ? "border-blue-600 text-blue-600" : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
            }`}>
            <t.icon className="w-4 h-4" />
            {t.label}
          </button>
        ))}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === "overview" && (
        <div className="space-y-6">
          
          {/* Ringkasan Otomatis AI-like Box */}
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-5 shadow-sm">
            <div className="flex items-start gap-3">
              <Lightbulb className="text-blue-600 w-6 h-6 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-blue-900 font-bold text-sm mb-1">Ringkasan Otomatis</p>
                <p className="text-blue-800 text-sm leading-relaxed">
                  Berdasarkan <span className="font-bold">{session.total_reviews.toLocaleString()} ulasan</span> yang dianalisis,
                  sentimen konsumen terhadap produk ini terpantau{" "}
                  <span className={positivePct >= 70 ? "text-green-700 font-bold" : "text-red-700 font-bold"}>
                    {insight.sentiment_text}
                  </span>.{" "}
                  {insight.cluster_text} {insight.positive_text}
                </p>
                <button onClick={() => setTab("insight")} className="mt-3 text-blue-600 font-medium text-xs hover:underline flex items-center gap-1">
                  Lihat insight & rekomendasi lengkap <ArrowLeft className="w-3 h-3 rotate-180" />
                </button>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white border border-gray-200 shadow-sm rounded-xl p-6">
              <h3 className="font-bold text-gray-900 mb-4 text-sm uppercase tracking-wide">Distribusi Sentimen Keseluruhan</h3>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={2} dataKey="value">
                    {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)' }} formatter={(v, n) => [v.toLocaleString() + ' ulasan', n]} />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white border border-gray-200 shadow-sm rounded-xl p-6">
              <h3 className="font-bold text-gray-900 mb-4 text-sm uppercase tracking-wide">Distribusi per Topik Cluster</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={clusterBarData} layout="vertical" margin={{ left:0, right:16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={true} vertical={false} />
                  <XAxis type="number" tick={{ fill:"#6b7280", fontSize:11 }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="name" tick={{ fill:"#374151", fontSize:11, fontWeight: 500 }} width={120} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)' }} cursor={{fill: '#f3f4f6'}} />
                  <Bar dataKey="Positif" fill="#22c55e" radius={[0,4,4,0]} stackId="a" />
                  <Bar dataKey="Negatif" fill="#ef4444" radius={[0,4,4,0]} stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* ── INSIGHT ── */}
      {tab === "insight" && (
        <div className="space-y-6">
          <div className="bg-white border border-gray-200 shadow-sm rounded-xl p-6">
            <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-yellow-500" /> Interpretasi Sentimen Sentral
            </h3>
            <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
              <p className="text-gray-700 text-sm leading-relaxed">
                Dari total <span className="font-bold text-gray-900">{session.total_reviews.toLocaleString()} data uji ulasan</span> konsumen terhadap produk 
                <span className="font-bold text-blue-600"> {session.brand_name || "ini"}</span>, 
                model Machine Learning mendeteksi bahwa <span className="font-bold">{positivePct}% ulasan mengandung sentimen positif</span>. 
                {positivePct >= 80 ? " Tingkat kepuasan ini sangat tinggi dan mengindikasikan penerimaan pasar (Product-Market Fit) yang sangat baik." 
                  : positivePct >= 60 ? " Tingkat kepuasan ini cukup baik, namun masih ada area abu-abu yang perlu diidentifikasi dari ulasan negatif." 
                  : " Tingkat sentimen ini menandakan adanya masalah mendasar pada produk atau layanan yang perlu dievaluasi secepatnya."}
              </p>
            </div>
          </div>

          <div className="bg-white border border-gray-200 shadow-sm rounded-xl p-6">
            <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-blue-600" /> Rekomendasi Tindakan (Actionable Insights)
            </h3>
            <p className="text-gray-500 text-sm mb-5">
              Rekomendasi di bawah dihasilkan dengan memetakan hasil Word2Vec Clustering ke dalam kategori domain bisnis skincare.
            </p>
            
            {recommendations.length === 0 ? (
              <p className="text-gray-500 text-sm text-center py-8 bg-gray-50 rounded-lg border border-dashed border-gray-300">Tidak ada anomali sentimen negatif yang signifikan. Kondisi produk terpantau stabil.</p>
            ) : (
              <div className="space-y-3">
                {recommendations.map((rec, i) => (
                  <div key={i} className={`rounded-lg p-4 border ${recStyle[rec.type]}`}>
                    <div className="flex items-start gap-4">
                      <div className={`p-2 rounded-full bg-white/60 shadow-sm mt-0.5`}>
                        <rec.icon className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="font-bold text-sm mb-1">{rec.title}</h4>
                        <p className="text-sm opacity-90 leading-relaxed mb-2">{rec.desc}</p>
                        <span className="inline-block bg-white/60 px-2 py-1 rounded text-xs font-semibold opacity-80 border border-black/5">
                          Topik Rujukan: {rec.cluster}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── CLUSTERS ── */}
      {tab === "clusters" && (
        <div className="space-y-4">
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
            <p className="text-gray-600 text-sm">
              <span className="font-semibold text-gray-900">Agglomerative Hierarchical Clustering (AHC):</span> Algoritma secara mandiri mengelompokkan ulasan menjadi {session.n_clusters} cluster berdasarkan kedekatan jarak semantik antar kata (Word Embedding).
              {session.silhouette_score && <span className="block mt-1">Skor validasi Silhouette: <span className="font-bold">{Number(session.silhouette_score).toFixed(4)}</span></span>}
            </p>
          </div>

          {clusterList.map((cluster, i) => {
            const pct = cluster.count > 0 ? ((cluster.positive / cluster.count) * 100).toFixed(1) : 0;
            const name = clusterName(cluster.top_keywords);
            
            return (
              <div key={i} className="bg-white border border-gray-200 shadow-sm rounded-xl p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="font-bold text-gray-900 text-lg flex items-center gap-2">
                      Cluster {cluster.cluster_id} <span className="text-gray-400 font-normal">|</span> <span className="text-blue-600">{name}</span>
                    </h3>
                    <p className="text-gray-500 text-sm mt-1">{cluster.count.toLocaleString()} ulasan termuat dalam kelompok ini.</p>
                  </div>
                  <div className="text-right">
                    <span className={`text-sm font-bold px-3 py-1 rounded-full border ${pct >= 50 ? "bg-green-50 text-green-700 border-green-200" : "bg-red-50 text-red-700 border-red-200"}`}>
                      {pct}% Positif
                    </span>
                  </div>
                </div>

                <div className="flex h-2 rounded-full overflow-hidden mb-3 bg-gray-100">
                  <div className="bg-green-500" style={{ width:`${pct}%` }} />
                  <div className="bg-red-500" style={{ width:`${100-pct}%` }} />
                </div>
                
                <div className="mb-4">
                  <p className="text-gray-500 text-xs font-semibold mb-2 uppercase tracking-wide">Kata Kunci Pembentuk Topik (Top Keywords)</p>
                  <div className="flex flex-wrap gap-2">
                    {(cluster.top_keywords || []).map((kw, j) => (
                      <span key={j} className="bg-gray-100 text-gray-700 text-xs px-2.5 py-1 rounded border border-gray-200">{kw}</span>
                    ))}
                  </div>
                </div>

                <button
                  onClick={() => { setTab("reviews"); setFilterCluster(String(cluster.cluster_id)); setPage(1); }}
                  className="text-sm font-medium text-blue-600 hover:text-blue-800 hover:underline flex items-center gap-1"
                >
                  Inspeksi isi ulasan pada cluster ini <ArrowLeft className="w-3 h-3 rotate-180" />
                </button>
              </div>
            );
          })}
        </div>
      )}

      {/* ── REVIEWS ── */}
      {tab === "reviews" && (
        <div className="bg-white border border-gray-200 shadow-sm rounded-xl p-6">
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
              <option value="all">Semua Cluster Topik</option>
              {clusterList.map((c, i) => (
                <option key={i} value={String(c.cluster_id)}>Cluster {c.cluster_id}: {clusterName(c.top_keywords)}</option>
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
                  <th className="px-4 py-3 font-semibold text-center w-32">Topik Cluster</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {paginated.length === 0 ? (
                  <tr><td colSpan={4} className="text-center py-8 text-gray-500">Tidak ada ulasan yang cocok dengan pencarian.</td></tr>
                ) : paginated.map((r, i) => (
                  <tr key={i} className="hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3">
                      <p className="text-gray-900 leading-relaxed">{r.review_text}</p>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${
                        r.sentiment === "positive" ? "bg-green-50 text-green-700 border-green-200" : "bg-red-50 text-red-700 border-red-200"
                      }`}>
                        {r.sentiment === "positive" ? "Positif" : "Negatif"}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center text-gray-600 font-medium">
                      {(r.confidence * 100).toFixed(0)}%
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

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-1 mt-6">
              <button onClick={() => setPage(p => Math.max(1, p-1))} disabled={page === 1}
                className="px-3 py-1.5 rounded text-sm font-medium text-gray-600 border border-gray-300 hover:bg-gray-50 disabled:opacity-50 transition-colors">
                Prev
              </button>
              {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                const p = totalPages <= 7 ? i+1 : page <= 4 ? i+1 : page >= totalPages-3 ? totalPages-6+i : page-3+i;
                return (
                  <button key={p} onClick={() => setPage(p)}
                    className={`w-8 h-8 rounded text-sm font-medium border transition-colors ${
                      page === p ? "bg-blue-600 text-white border-blue-600" : "text-gray-600 border-gray-300 hover:bg-gray-50"
                    }`}>
                    {p}
                  </button>
                );
              })}
              <button onClick={() => setPage(p => Math.min(totalPages, p+1))} disabled={page === totalPages}
                className="px-3 py-1.5 rounded text-sm font-medium text-gray-600 border border-gray-300 hover:bg-gray-50 disabled:opacity-50 transition-colors">
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}