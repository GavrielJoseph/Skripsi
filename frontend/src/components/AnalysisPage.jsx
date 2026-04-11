import { useState } from "react";

function AnalysisPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState([]);
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);

  const steps = [
    "Preprocessing",
    "Word2Vec Embedding",
    "Sentiment Classification",
    "Clustering",
    "Keyword Extraction",
  ];

  // 📌 Handle file upload + preview CSV
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    if (!selectedFile) return;

    const reader = new FileReader();

    reader.onload = (event) => {
      const text = event.target.result;

      // split CSV (simple version)
      const rows = text.split("\n").slice(0, 5); // ambil 5 baris saja
      setPreview(rows);
    };

    reader.readAsText(selectedFile);
  };

  // 📌 Simulasi proses ML
  const handleAnalyze = async () => {
    if (!file) {
      alert("Upload file dulu");
      return;
    }

    setLoading(true);
    setStep(0);

    for (let i = 0; i < steps.length; i++) {
      await new Promise((res) => setTimeout(res, 1000)); // delay simulasi
      setStep(i + 1);
    }

    setLoading(false);
    alert("Analisis selesai (nanti connect backend)");
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Title */}
      <h1 className="text-3xl font-bold text-blue-400 mb-2">
        Skincare Review Analysis
      </h1>
      <p className="text-gray-400 mb-6">Upload Dataset</p>

      {/* Upload Section */}
      <div className="bg-gray-800 p-6 rounded-xl shadow mb-6">
        <h2 className="text-xl font-semibold mb-4">
          Upload Review Dataset (CSV)
        </h2>

        <input
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="mb-4"
        />

        {file && (
          <p className="text-green-400 text-sm">
            File: {file.name}
          </p>
        )}

        {/* Preview REAL */}
        {preview.length > 0 && (
          <div className="mt-6">
            <h3 className="mb-2 font-semibold">Preview Dataset:</h3>
            <div className="bg-gray-700 rounded p-3 text-sm space-y-1">
              {preview.map((row, index) => (
                <p key={index}>{row}</p>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Process Section */}
      <div className="bg-gray-800 p-6 rounded-xl shadow">
        <h2 className="text-xl font-semibold mb-4">Process:</h2>

        <div className="space-y-2 text-sm">
          {steps.map((s, index) => (
            <div
              key={index}
              className={`p-2 rounded ${
                index < step
                  ? "bg-green-600"
                  : index === step && loading
                  ? "bg-yellow-500"
                  : "bg-gray-700"
              }`}
            >
              {index + 1}. {s}
            </div>
          ))}
        </div>

        {/* Button */}
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="mt-6 w-full bg-blue-500 hover:bg-blue-600 py-2 rounded-lg font-semibold disabled:bg-gray-500"
        >
          {loading ? "Processing..." : "Start Analyze"}
        </button>
      </div>
    </div>
  );
}

export default AnalysisPage;