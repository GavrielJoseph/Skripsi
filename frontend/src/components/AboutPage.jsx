export default function AboutPage() {
  return (
    <>
      {/* Background Wardah — sama persis dengan halaman lain */}
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

      <div className="max-w-3xl mx-auto pb-16">
        <div className="bg-white/90 backdrop-blur-sm border border-sky-100 rounded-2xl p-8 shadow-sm">

          {/* Label + Judul */}
          <p className="text-xs font-bold text-blue-600 uppercase tracking-widest mb-1">Rancangan</p>
          <h1 className="text-xl font-bold text-gray-900 mb-6 leading-snug">
            Perancangan Sistem Analisis Opini Konsumen Terhadap Produk Skincare
            Menggunakan Regresi Logistik dan Klasterisasi Hierarki Aglomeratif
          </h1>

          <hr className="border-gray-200 mb-6" />

          {/* Konten */}
          <div className="space-y-4 text-gray-700 text-sm leading-relaxed">
            <p>
              Industri skincare di Indonesia terus berkembang, dan konsumen semakin aktif
              meninggalkan ulasan di platform seperti Tokopedia — mulai dari kesan pemakaian,
              cocok atau tidaknya di kulit, harga, kemasan, sampai pengalaman belanja. Tapi
              ulasan-ulasan itu datang dalam bahasa yang tidak formal, penuh singkatan dan
              kata slang, dan jumlahnya terus bertambah setiap hari. Membacanya satu per satu
              jelas bukan pilihan yang efisien.
            </p>
            <p>
              SkinSentiment dibuat untuk menjawab masalah itu. Sistem ini menerima data ulasan
              dalam bentuk file CSV, lalu mengolahnya secara otomatis: mengklasifikasikan
              sentimen tiap ulasan menjadi positif atau negatif menggunakan Regresi Logistik,
              kemudian mengelompokkannya berdasarkan kesamaan topik pembahasan menggunakan
              Klasterisasi Hierarki Aglomeratif.
            </p>
            <p>
              Kenapa perlu dua pendekatan? Karena label sentimen saja tidak cukup untuk
              menjelaskan apa yang sebenarnya dikeluhkan. Dua ulasan negatif bisa bicara soal
              hal yang sama sekali berbeda — satu tentang iritasi kulit, satu lagi tentang
              paket yang datang rusak. Dengan clustering ditambah deteksi aspek berbasis
              rule based, sistem bisa langsung menunjukkan isu mana yang paling dominan
              tanpa perlu membaca ulasan satu per satu.
            </p>

            {/* Info pembuat */}
            <div className="pt-4 border-t border-gray-100 text-gray-600 text-sm space-y-0.5">
              <p>
                <span className="font-semibold text-gray-800">👤 Dikembangkan oleh:</span>{" "}
                Gavriel Joseph Lim (NIM. 535220049)
              </p>
              <p>Program Studi Teknik Informatika</p>
              <p>Universitas Tarumanagara (2026)</p>
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
