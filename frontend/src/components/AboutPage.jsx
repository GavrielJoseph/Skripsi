export default function AboutPage() {
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

      <div className="max-w-3xl mx-auto pb-16">
        <div className="bg-white/90 backdrop-blur-sm border border-sky-100 rounded-2xl p-8 shadow-sm">

          <div className="flex items-start justify-between mb-1">
            <p className="text-xs font-bold text-blue-600 uppercase tracking-widest">Tentang Sistem</p>
            <img src="/logofti.png" alt="Logo FTI UNTAR" className="h-12 w-auto" />
          </div>

          <h1 className="text-xl font-bold text-gray-900 mb-6 leading-snug">
            Perancangan Sistem Analisis Opini Konsumen Terhadap Produk Skincare
            Menggunakan Regresi Logistik dan Klasterisasi Hierarki Aglomeratif
          </h1>

          <hr className="border-gray-200 mb-6" />

          <div className="space-y-4 text-gray-700 text-sm leading-relaxed">
            <p>
              SkinSentiment adalah sistem berbasis website yang dirancang untuk melakukan
              analisis opini konsumen terhadap produk skincare dari platform Tokopedia.
              Sistem menerima input berupa dataset ulasan dalam format CSV atau URL produk
              Tokopedia untuk dilakukan scraping secara otomatis.
            </p>
            <p>
              Proses analisis dilakukan melalui dua tahap utama. Pertama, klasifikasi
              sentimen menggunakan Regresi Logistik untuk menentukan apakah setiap ulasan
              bersifat positif atau negatif. Kedua, pengelompokan ulasan berdasarkan
              kesamaan topik pembahasan menggunakan Klasterisasi Hierarki Aglomeratif.
              Selain itu, sistem juga melakukan identifikasi aspek produk yang dibahas
              konsumen menggunakan pendekatan rule-based berbasis pencocokan kata kunci
              domain skincare.
            </p>
            <p>
              Hasil analisis ditampilkan dalam bentuk visualisasi distribusi sentimen,
              ringkasan per aspek produk, daftar cluster topik, dan daftar ulasan beserta
              label sentimen dan nilai confidence dari setiap ulasan.
            </p>

            <div className="pt-4 border-t border-gray-100 text-gray-600 text-sm space-y-0.5">
              <p>
                <span className="font-semibold text-gray-800">Dikembangkan oleh:</span>{" "}
                <span className="font-bold text-gray-900">Gavriel Joseph Lim</span> (NIM. 535220049)
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