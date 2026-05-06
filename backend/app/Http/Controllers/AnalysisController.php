<?php

namespace App\Http\Controllers;

use App\Models\AnalysisSession;
use App\Models\AnalysisResult;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;

class AnalysisController extends Controller
{
    public function analyze(Request $request)
    {
        // 1. Validasi input: sekarang wajib menerima text_column
        $request->validate([
            'file'           => 'required|file|mimes:csv,txt|max:10240',
            'brand_name'     => 'nullable|string|max:100',
            'text_column'    => 'required|string',
            'product_column' => 'nullable|string',
        ]);

        $file      = $request->file('file');
        $filename  = time() . '_' . $file->getClientOriginalName(); // Tambah timestamp agar nama unik
        $brandName = $request->input('brand_name', '');
        $textCol   = $request->input('text_column');
        $prodCol   = $request->input('product_column', '');
        
        $csvPath   = $file->storeAs('uploads', $filename, 'local');
        $fullPath  = storage_path('app' . DIRECTORY_SEPARATOR . 'private' . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . $filename);

        $session = AnalysisSession::create([
            'filename'       => $filename,
            'brand_name'     => $brandName,
            'status'         => 'processing',
            'total_reviews'  => 0,
            'positive_count' => 0,
            'negative_count' => 0,
            'n_clusters'     => 0,
            'silhouette_score' => null,
        ]);

        if (!file_exists($fullPath)) {
            $session->update(['status' => 'failed']);
            return response()->json(['status' => 'error', 'message' => 'File gagal disimpan ke: ' . $fullPath], 500);
        }

        $pythonBin  = PHP_OS_FAMILY === 'Windows' ? 'python' : 'python3';
        $scriptPath = base_path('ml' . DIRECTORY_SEPARATOR . 'scripts' . DIRECTORY_SEPARATOR . 'predict.py');
        
        // 2. Modifikasi Command: Melempar variabel text_column dan product_column ke Python
        $command    = $pythonBin . ' ' . escapeshellarg($scriptPath) . ' ' 
                    . escapeshellarg($fullPath) . ' ' 
                    . escapeshellarg($textCol) . ' ' 
                    . escapeshellarg($prodCol) . ' 2>&1';
                    
        $output     = shell_exec($command);

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

        // Simpan hasil ke analysis_results
        $rows = [];
        foreach ($result['results'] as $item) {
            $rows[] = [
                'session_id'   => $session->id,
                'product_name' => $item['product_name'] ?? '',
                'review_text'  => $item['review_text'],
                'sentiment'    => $item['sentiment'],
                'confidence'   => $item['confidence'] ?? 0,
                'cluster_id'   => $item['cluster_id'],
                'keywords'     => json_encode($item['keywords']),
                'created_at'   => now(),
                'updated_at'   => now(),
            ];
        }
        
        foreach (array_chunk($rows, 500) as $chunk) {
            AnalysisResult::insert($chunk);
        }

        // Update session
        $session->update([
            'status'               => 'done',
            'total_reviews'        => $result['total'],
            'positive_count'       => $result['summary']['positive'],
            'negative_count'       => $result['summary']['negative'],
            'n_clusters'           => $result['summary']['n_clusters'],
            'silhouette_score'     => $result['summary']['silhouette_score'] ?? null,
            'confidence_distribution' => json_encode($result['summary']['confidence_distribution'] ?? []),
        ]);

        return response()->json([
            'status'     => 'success',
            'session_id' => $session->id,
            'summary'    => $result['summary'],
            'total'      => $result['total'],
        ]);
    }

    public function results($id)
    {
        $session = AnalysisSession::find($id);
        if (!$session) {
            return response()->json(['status' => 'error', 'message' => 'Session tidak ditemukan.'], 404);
        }

        $results = AnalysisResult::where('session_id', $id)->get();

        $clusters = [];
        foreach ($results->groupBy('cluster_id') as $clusterId => $items) {
            $allKeywords = [];
            foreach ($items as $item) {
                $allKeywords = array_merge($allKeywords, $item->keywords ?? []);
            }
            $freq = array_count_values($allKeywords);
            arsort($freq);
            $topKeywords = array_keys(array_slice($freq, 0, 10, true));

            $clusters[$clusterId] = [
                'cluster_id'     => $clusterId,
                'count'          => $items->count(),
                'positive'       => $items->where('sentiment', 'positive')->count(),
                'negative'       => $items->where('sentiment', 'negative')->count(),
                'avg_confidence' => round($items->avg('confidence'), 4),
                'top_keywords'   => $topKeywords,
            ];
        }

        $confDist = [];
        if ($session->confidence_distribution) {
            $confDist = is_string($session->confidence_distribution)
                ? json_decode($session->confidence_distribution, true)
                : $session->confidence_distribution;
        }

        return response()->json([
            'status'  => 'success',
            'session' => [
                'id'                       => $session->id,
                'filename'                 => $session->filename,
                'brand_name'               => $session->brand_name,
                'status'                   => $session->status,
                'total_reviews'            => $session->total_reviews,
                'positive_count'           => $session->positive_count,
                'negative_count'           => $session->negative_count,
                'n_clusters'               => $session->n_clusters,
                'silhouette_score'         => $session->silhouette_score,
                'confidence_distribution'  => $confDist,
                'created_at'               => $session->created_at,
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
            return response()->json(['status' => 'error', 'message' => 'File model_performance.json belum ada. Jalankan training dulu.'], 404);
        }
        $performance = json_decode(file_get_contents($perfPath), true);
        return response()->json(['status' => 'success', 'performance' => $performance]);
    }

    // =========================================================================
    // FUNGSI BARU UNTUK MENGHAPUS DATA (DIPANGGIL DARI UI DASHBOARD)
    // =========================================================================

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

    // Private helper untuk menghapus data secara bersih (Database + File Fisik)
    private function deleteSessionData($session)
    {
        // 1. Hapus semua ulasan (results) yang terkait dengan session ini
        AnalysisResult::where('session_id', $session->id)->delete();
        
        // 2. Hapus file CSV fisik di storage agar hard disk server tidak penuh
        $filePath = storage_path('app' . DIRECTORY_SEPARATOR . 'private' . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . $session->filename);
        if (file_exists($filePath)) {
            @unlink($filePath);
        }

        // 3. Hapus record session
        $session->delete();
    }
}