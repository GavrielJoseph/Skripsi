<?php

namespace App\Http\Controllers;

use App\Models\AnalysisSession;
use App\Models\AnalysisResult;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;

class AnalysisController extends Controller
{
    // =========================================================================
    // ANALYZE — Upload CSV lalu analisis
    // =========================================================================

    public function analyze(Request $request)
    {
        $request->validate([
            'file'           => 'required|file|mimes:csv,txt|max:10240',
            'brand_name'     => 'nullable|string|max:100',
            'text_column'    => 'required|string',
            'product_column' => 'nullable|string',
        ]);

        $file = $request->file('file');

        // Validasi magic bytes — tangkap Excel yang di-rename jadi .csv
        $firstBytes = file_get_contents($file->getRealPath(), false, null, 0, 8);
        if (
            str_starts_with($firstBytes, "\xD0\xCF\x11\xE0") ||  // .xls
            str_starts_with($firstBytes, "PK\x03\x04")            // .xlsx (zip-based)
        ) {
            return response()->json([
                'status'  => 'error',
                'message' => 'File yang diupload terdeteksi sebagai Excel (.xls/.xlsx). Harap konversi ke format CSV terlebih dahulu melalui Excel: File → Save As → CSV (Comma delimited).',
            ], 422);
        }

        // Validasi isi — minimal ada baris pertama yang readable sebagai teks
        $handle    = fopen($file->getRealPath(), 'r');
        $firstLine = fgets($handle);
        fclose($handle);

        if (empty(trim($firstLine))) {
            return response()->json([
                'status'  => 'error',
                'message' => 'File CSV kosong atau tidak dapat dibaca.',
            ], 422);
        }

        $filename  = time() . '_' . $file->getClientOriginalName();
        $brandName = $request->input('brand_name', '');
        $textCol   = $request->input('text_column');
        $prodCol   = $request->input('product_column', '');

        $csvPath  = $file->storeAs('uploads', $filename, 'local');
        $fullPath = storage_path('app' . DIRECTORY_SEPARATOR . 'private' . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . $filename);

        $session = AnalysisSession::create([
            'filename'         => $filename,
            'brand_name'       => $brandName,
            'status'           => 'processing',
            'total_reviews'    => 0,
            'positive_count'   => 0,
            'negative_count'   => 0,
            'n_clusters'       => 0,
            'silhouette_score' => null,
        ]);

        if (!file_exists($fullPath)) {
            $session->update(['status' => 'failed']);
            return response()->json(['status' => 'error', 'message' => 'File gagal disimpan ke: ' . $fullPath], 500);
        }

        $pythonBin  = PHP_OS_FAMILY === 'Windows' ? 'python' : 'python3';
        $scriptPath = base_path('ml' . DIRECTORY_SEPARATOR . 'scripts' . DIRECTORY_SEPARATOR . 'predict.py');

        $command = $pythonBin . ' ' . escapeshellarg($scriptPath) . ' '
                 . escapeshellarg($fullPath) . ' '
                 . escapeshellarg($textCol) . ' '
                 . escapeshellarg($prodCol) . ' 2>&1';

        $output = shell_exec($command);

        if (empty($output)) {
            $session->update(['status' => 'failed']);
            return response()->json(['status' => 'error', 'message' => 'Python script gagal dijalankan.'], 500);
        }

        $jsonStart = strpos($output, '{');
        if ($jsonStart === false) {
            $session->update(['status' => 'failed']);
            return response()->json(['status' => 'error', 'message' => 'Output Python tidak valid: ' . substr($output, 0, 200)], 500);
        }

        $jsonOutput = substr($output, $jsonStart);
        $result     = json_decode($jsonOutput, true);

        if (!$result || $result['status'] !== 'success') {
            $session->update(['status' => 'failed']);
            return response()->json(['status' => 'error', 'message' => $result['message'] ?? 'Analisis gagal.'], 500);
        }

        $this->saveResults($session, $result);

        return response()->json([
            'status'     => 'success',
            'session_id' => $session->id,
            'summary'    => $result['summary'],
            'total'      => $result['total'],
        ]);
    }

    // =========================================================================
    // SCRAPE AND ANALYZE
    // =========================================================================

    public function scrapeAndAnalyze(Request $request)
    {
        $request->validate([
            'url'          => 'required|string|url',
            'brand_name'   => 'nullable|string|max:100',
            'product_name' => 'required|string|max:200',
            'max_pages'    => 'nullable|integer|min:1|max:100',
        ]);

        $url = $request->input('url');

        // Validasi URL harus dari tokopedia.com — tangkap di backend
        $parsedHost = parse_url($url, PHP_URL_HOST);
        if (!$parsedHost || !str_contains(strtolower($parsedHost), 'tokopedia.com')) {
            return response()->json([
                'status'  => 'error',
                'message' => 'URL harus berasal dari tokopedia.com. URL dari Shopee, Lazada, atau platform lain tidak didukung.',
            ], 422);
        }

        // Pastikan URL mengarah ke halaman produk (bukan search, home, dll)
        $parsedPath = parse_url($url, PHP_URL_PATH);
        $pathParts  = array_filter(explode('/', $parsedPath ?? ''));
        if (count($pathParts) < 2) {
            return response()->json([
                'status'  => 'error',
                'message' => 'URL tidak mengarah ke halaman produk. Salin URL dari halaman produk Tokopedia secara langsung.',
            ], 422);
        }

        $brandName   = $request->input('brand_name', '');
        $productName = $request->input('product_name');
        $maxPages    = $request->input('max_pages', 50);

        set_time_limit(0);
        ini_set('max_execution_time', 0);

        $safeName = preg_replace('/[^a-zA-Z0-9]/', '_', $productName);
        $filename = time() . '_scraped_' . $safeName . '.csv';
        $csvPath  = storage_path(
            'app' . DIRECTORY_SEPARATOR . 'private' . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . $filename
        );

        $uploadDir = storage_path('app' . DIRECTORY_SEPARATOR . 'private' . DIRECTORY_SEPARATOR . 'uploads');
        if (!is_dir($uploadDir)) {
            mkdir($uploadDir, 0755, true);
        }

        $session = AnalysisSession::create([
            'filename'         => $filename,
            'brand_name'       => $brandName,
            'status'           => 'scraping',
            'total_reviews'    => 0,
            'positive_count'   => 0,
            'negative_count'   => 0,
            'n_clusters'       => 0,
            'silhouette_score' => null,
        ]);

        $pythonBin   = PHP_OS_FAMILY === 'Windows' ? 'python' : 'python3';
        $scraperPath = base_path('ml' . DIRECTORY_SEPARATOR . 'scripts' . DIRECTORY_SEPARATOR . 'scraping.py');
        $predictPath = base_path('ml' . DIRECTORY_SEPARATOR . 'scripts' . DIRECTORY_SEPARATOR . 'predict.py');

        $scrapeCommand = $pythonBin . ' ' . escapeshellarg($scraperPath) . ' '
                       . escapeshellarg($url) . ' '
                       . escapeshellarg($productName) . ' '
                       . escapeshellarg((string)$maxPages) . ' '
                       . escapeshellarg($csvPath) . ' 2>&1';

        $scrapeOutput = shell_exec($scrapeCommand);

        Log::info("Output Scraper: " . $scrapeOutput);

        if (empty($scrapeOutput)) {
            $session->update(['status' => 'failed']);
            return response()->json([
                'status'  => 'error',
                'message' => 'Scraper gagal dijalankan. Pastikan Selenium dan Chrome sudah terinstall.',
            ], 500);
        }

        $scrapeLines = array_filter(explode("\n", trim($scrapeOutput)));
        $scrapeJson  = null;
        foreach (array_reverse($scrapeLines) as $line) {
            $decoded = json_decode(trim($line), true);
            if ($decoded && isset($decoded['status'])) {
                $scrapeJson = $decoded;
                break;
            }
        }

        if (!$scrapeJson || $scrapeJson['status'] !== 'success') {
            $session->update(['status' => 'failed']);
            $errMsg = $scrapeJson['message'] ?? 'Scraping gagal. Cek URL atau koneksi internet.';
            return response()->json(['status' => 'error', 'message' => $errMsg], 500);
        }

        if (!file_exists($csvPath)) {
            $session->update(['status' => 'failed']);
            return response()->json([
                'status'  => 'error',
                'message' => 'File CSV tidak ditemukan setelah scraping.',
            ], 500);
        }

        $session->update(['status' => 'processing']);

        $predictCommand = $pythonBin . ' ' . escapeshellarg($predictPath) . ' '
                        . escapeshellarg($csvPath) . ' '
                        . escapeshellarg('Comment') . ' '
                        . escapeshellarg('ProductName') . ' 2>&1';

        $predictOutput = shell_exec($predictCommand);

        if (empty($predictOutput)) {
            $session->update(['status' => 'failed']);
            return response()->json(['status' => 'error', 'message' => 'Analisis ML gagal dijalankan.'], 500);
        }

        $jsonStart = strpos($predictOutput, '{');
        if ($jsonStart === false) {
            $session->update(['status' => 'failed']);
            return response()->json([
                'status'  => 'error',
                'message' => 'Output analisis tidak valid: ' . substr($predictOutput, 0, 200),
            ], 500);
        }

        $jsonOutput = substr($predictOutput, $jsonStart);
        $result     = json_decode($jsonOutput, true);

        if (!$result || $result['status'] !== 'success') {
            $session->update(['status' => 'failed']);
            return response()->json([
                'status'  => 'error',
                'message' => $result['message'] ?? 'Analisis ML gagal.',
            ], 500);
        }

        $this->saveResults($session, $result);

        return response()->json([
            'status'         => 'success',
            'session_id'     => $session->id,
            'total_scraped'  => $scrapeJson['total'],
            'total_analyzed' => $result['total'],
            'summary'        => $result['summary'],
        ]);
    }

    // =========================================================================
    // HELPER — Simpan hasil
    // =========================================================================

    private function saveResults(AnalysisSession $session, array $result): void
    {
        $rows = [];
        foreach ($result['results'] as $item) {
            $rows[] = [
                'session_id'   => $session->id,
                'product_name' => $item['product_name'] ?? '',
                'review_text'  => $item['review_text'],
                'sentiment'    => $item['sentiment'],
                'confidence'   => $item['confidence'] ?? 0,
                'cluster_id'   => $item['cluster_id'],
                'keywords'     => json_encode($item['keywords'] ?? []),
                'created_at'   => now(),
                'updated_at'   => now(),
            ];
        }

        foreach (array_chunk($rows, 500) as $chunk) {
            AnalysisResult::insert($chunk);
        }

        $topicOverview  = $result['summary']['topic_overview'] ?? [];
        $clusterSummary = $result['summary']['clusters'] ?? [];

        $session->update([
            'status'                  => 'done',
            'total_reviews'           => $result['total'],
            'positive_count'          => $result['summary']['positive'],
            'negative_count'          => $result['summary']['negative'],
            'n_clusters'              => $result['summary']['n_clusters'],
            'silhouette_score'        => $result['summary']['silhouette_score'] ?? null,
            'confidence_distribution' => $result['summary']['confidence_distribution'] ?? [],
            'topic_overview'          => $topicOverview,
            'cluster_summary'         => $clusterSummary,
        ]);
    }

    // =========================================================================
    // RESULTS & SESSIONS
    // =========================================================================

    public function results($id)
    {
        $session = AnalysisSession::find($id);
        if (!$session) {
            return response()->json(['status' => 'error', 'message' => 'Session tidak ditemukan.'], 404);
        }

        $results  = AnalysisResult::where('session_id', $id)->get();
        $clusters = [];

        if (!empty($session->cluster_summary)) {
            foreach ($session->cluster_summary as $clusterId => $clusterData) {
                $clusters[$clusterId] = $clusterData;
                if (!isset($clusters[$clusterId]['cluster_id'])) {
                    $clusters[$clusterId]['cluster_id'] = (int) $clusterId;
                }
            }
        } else {
            foreach ($results->groupBy('cluster_id') as $clusterId => $items) {
                $allKeywords = [];
                foreach ($items as $item) {
                    $allKeywords = array_merge($allKeywords, $item->keywords ?? []);
                }
                $freq = array_count_values($allKeywords);
                arsort($freq);
                $topKeywords = array_keys(array_slice($freq, 0, 10, true));

                $clusters[$clusterId] = [
                    'cluster_id'      => $clusterId,
                    'count'           => $items->count(),
                    'positive'        => $items->where('sentiment', 'positive')->count(),
                    'negative'        => $items->where('sentiment', 'negative')->count(),
                    'avg_confidence'  => round($items->avg('confidence'), 4),
                    'top_keywords'    => $topKeywords,
                    'topic_breakdown' => [],
                    'sample_reviews'  => [],
                ];
            }
        }

        $confDist = [];
        if ($session->confidence_distribution) {
            $confDist = is_array($session->confidence_distribution)
                ? $session->confidence_distribution
                : json_decode($session->confidence_distribution, true);
        }

        $topicOverview = [];
        if ($session->topic_overview) {
            $topicOverview = is_array($session->topic_overview)
                ? $session->topic_overview
                : json_decode($session->topic_overview, true);
        }

        return response()->json([
            'status'  => 'success',
            'session' => [
                'id'                      => $session->id,
                'filename'                => $session->filename,
                'brand_name'              => $session->brand_name,
                'status'                  => $session->status,
                'total_reviews'           => $session->total_reviews,
                'positive_count'          => $session->positive_count,
                'negative_count'          => $session->negative_count,
                'n_clusters'              => $session->n_clusters,
                'silhouette_score'        => $session->silhouette_score,
                'confidence_distribution' => $confDist,
                'topic_overview'          => $topicOverview,
                'created_at'              => $session->created_at,
            ],
            'clusters' => $clusters,
            'results'  => $results->map(function ($r) {
                return [
                    'id'           => $r->id,
                    'product_name' => $r->product_name,
                    'review_text'  => $r->review_text,
                    'sentiment'    => $r->sentiment,
                    'confidence'   => $r->confidence,
                    'cluster_id'   => $r->cluster_id,
                    'keywords'     => $r->keywords,
                ];
            }),
        ]);
    }

    public function sessions()
    {
        $sessions = AnalysisSession::orderBy('created_at', 'desc')->get();
        return response()->json(['status' => 'success', 'sessions' => $sessions]);
    }

    public function sessionsByBrand()
    {
        $sessions = AnalysisSession::where('status', 'done')->orderBy('created_at', 'desc')->get();

        $brands = [];
        foreach ($sessions as $session) {
            $brand = $session->brand_name ?: 'Lainnya';
            if (!isset($brands[$brand])) {
                $brands[$brand] = [
                    'brand_name'     => $brand,
                    'total_sessions' => 0,
                    'total_reviews'  => 0,
                    'positive_count' => 0,
                    'negative_count' => 0,
                    'sessions'       => [],
                ];
            }
            $brands[$brand]['total_sessions']++;
            $brands[$brand]['total_reviews']  += $session->total_reviews;
            $brands[$brand]['positive_count'] += $session->positive_count;
            $brands[$brand]['negative_count'] += $session->negative_count;
            $brands[$brand]['sessions'][]      = [
                'id'               => $session->id,
                'filename'         => $session->filename,
                'total_reviews'    => $session->total_reviews,
                'positive_count'   => $session->positive_count,
                'negative_count'   => $session->negative_count,
                'n_clusters'       => $session->n_clusters,
                'silhouette_score' => $session->silhouette_score,
                'created_at'       => $session->created_at,
            ];
        }

        return response()->json(['status' => 'success', 'brands' => array_values($brands)]);
    }

    public function modelPerformance()
    {
        $perfPath = base_path('ml' . DIRECTORY_SEPARATOR . 'models' . DIRECTORY_SEPARATOR . 'model_performance.json');
        if (!file_exists($perfPath)) {
            return response()->json([
                'status'  => 'error',
                'message' => 'File model_performance.json belum ada. Jalankan training dulu.',
            ], 404);
        }
        $performance = json_decode(file_get_contents($perfPath), true);
        return response()->json(['status' => 'success', 'performance' => $performance]);
    }

    public function deleteBrand($brandName)
    {
        $sessions = AnalysisSession::where('brand_name', $brandName)->get();

        if ($sessions->isEmpty()) {
            return response()->json(['status' => 'error', 'message' => 'Brand tidak ditemukan'], 404);
        }

        foreach ($sessions as $session) {
            $this->deleteSessionData($session);
        }

        return response()->json(['status' => 'success', 'message' => 'Brand dan seluruh data sesinya berhasil dihapus']);
    }

    public function deleteSession($id)
    {
        $session = AnalysisSession::find($id);

        if (!$session) {
            return response()->json(['status' => 'error', 'message' => 'Sesi tidak ditemukan'], 404);
        }

        $this->deleteSessionData($session);

        return response()->json(['status' => 'success', 'message' => 'Sesi berhasil dihapus']);
    }

    private function deleteSessionData($session): void
    {
        AnalysisResult::where('session_id', $session->id)->delete();

        $filePath = storage_path(
            'app' . DIRECTORY_SEPARATOR . 'private' . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . $session->filename
        );
        if (file_exists($filePath)) {
            @unlink($filePath);
        }

        $session->delete();
    }

    // =========================================================================
    // DOWNLOAD CSV RAW
    // =========================================================================

    public function downloadCsv($id)
    {
        $session = AnalysisSession::find($id);

        if (!$session) {
            return response()->json(['status' => 'error', 'message' => 'Sesi tidak ditemukan'], 404);
        }

        $filePath = storage_path(
            'app' . DIRECTORY_SEPARATOR . 'private' . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . $session->filename
        );

        if (!file_exists($filePath)) {
            return response()->json(['status' => 'error', 'message' => 'File CSV sudah tidak ada di server'], 404);
        }

        return response()->download($filePath, $session->filename);
    }
}