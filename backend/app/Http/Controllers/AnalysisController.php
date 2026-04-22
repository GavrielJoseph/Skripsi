<?php

namespace App\Http\Controllers;

use App\Models\AnalysisSession;
use App\Models\AnalysisResult;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

class AnalysisController extends Controller
{
    /**
     * POST /api/analyze
     * Terima file CSV + brand_name dari React, jalankan Python, simpan hasil ke DB.
     */
    public function analyze(Request $request)
    {
        // 1. Validasi
        $request->validate([
            'file'       => 'required|file|mimes:csv,txt|max:10240',
            'brand_name' => 'nullable|string|max:100',
        ]);

        // 2. Simpan file CSV ke storage
        $file       = $request->file('file');
        $filename   = $file->getClientOriginalName();
        $brandName  = $request->input('brand_name', '');
        $csvPath    = $file->storeAs('uploads', $filename, 'local');
        $fullPath   = storage_path('app' . DIRECTORY_SEPARATOR . 'private' . DIRECTORY_SEPARATOR . 'uploads' . DIRECTORY_SEPARATOR . $filename);

        // 3. Buat session record dengan status 'processing'
        $session = AnalysisSession::create([
            'filename'       => $filename,
            'brand_name'     => $brandName,
            'status'         => 'processing',
            'total_reviews'  => 0,
            'positive_count' => 0,
            'negative_count' => 0,
            'n_clusters'     => 0,
        ]);

        // 4. Cek file ada
        if (!file_exists($fullPath)) {
            $session->update(['status' => 'failed']);
            return response()->json([
                'status'  => 'error',
                'message' => 'File gagal disimpan ke: ' . $fullPath,
            ], 500);
        }

        // 5. Tentukan path Python dan script
        $pythonBin  = PHP_OS_FAMILY === 'Windows' ? 'python' : 'python3';
        $scriptPath = base_path('ml' . DIRECTORY_SEPARATOR . 'scripts' . DIRECTORY_SEPARATOR . 'predict.py');

        // 6. Jalankan Python script
        $command = $pythonBin . ' ' . escapeshellarg($scriptPath) . ' ' . escapeshellarg($fullPath) . ' 2>&1';
        $output  = shell_exec($command);

        // 7. Kalau output kosong = Python error
        if (empty($output)) {
            $session->update(['status' => 'failed']);
            return response()->json([
                'status'  => 'error',
                'message' => 'Python script gagal dijalankan. Cek apakah model sudah ditraining.',
            ], 500);
        }

        // 8. Parse JSON output dari Python
        // Cari JSON valid dari output (abaikan warning/print Python lainnya)
        $jsonStart = strpos($output, '{');
        if ($jsonStart === false) {
            $session->update(['status' => 'failed']);
            return response()->json([
                'status'  => 'error',
                'message' => 'Output Python tidak valid: ' . substr($output, 0, 200),
            ], 500);
        }
        $jsonOutput = substr($output, $jsonStart);
        $result     = json_decode($jsonOutput, true);

        if (!$result || $result['status'] !== 'success') {
            $session->update(['status' => 'failed']);
            return response()->json([
                'status'  => 'error',
                'message' => $result['message'] ?? 'Analisis gagal.',
            ], 500);
        }

        // 9. Simpan hasil ke tabel analysis_results
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

        // 10. Update session dengan summary
        $session->update([
            'status'         => 'done',
            'total_reviews'  => $result['total'],
            'positive_count' => $result['summary']['positive'],
            'negative_count' => $result['summary']['negative'],
            'n_clusters'     => $result['summary']['n_clusters'],
        ]);

        return response()->json([
            'status'     => 'success',
            'session_id' => $session->id,
            'summary'    => $result['summary'],
            'total'      => $result['total'],
        ]);
    }

    /**
     * GET /api/results/{id}
     * Ambil hasil analisis berdasarkan session ID.
     */
    public function results($id)
    {
        $session = AnalysisSession::find($id);

        if (!$session) {
            return response()->json([
                'status'  => 'error',
                'message' => 'Session tidak ditemukan.',
            ], 404);
        }

        $results = AnalysisResult::where('session_id', $id)->get();

        // Kelompokkan per cluster
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
                'cluster_id'      => $clusterId,
                'count'           => $items->count(),
                'positive'        => $items->where('sentiment', 'positive')->count(),
                'negative'        => $items->where('sentiment', 'negative')->count(),
                'avg_confidence'  => round($items->avg('confidence'), 4),
                'top_keywords'    => $topKeywords,
            ];
        }

        return response()->json([
            'status'  => 'success',
            'session' => [
                'id'             => $session->id,
                'filename'       => $session->filename,
                'brand_name'     => $session->brand_name,
                'status'         => $session->status,
                'total_reviews'  => $session->total_reviews,
                'positive_count' => $session->positive_count,
                'negative_count' => $session->negative_count,
                'n_clusters'     => $session->n_clusters,
                'created_at'     => $session->created_at,
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

    /**
     * GET /api/sessions
     * Ambil semua history sesi analisis (untuk dashboard).
     */
    public function sessions()
    {
        $sessions = AnalysisSession::orderBy('created_at', 'desc')->get();

        return response()->json([
            'status'   => 'success',
            'sessions' => $sessions,
        ]);
    }

    /**
     * GET /api/sessions/by-brand
     * Ambil sesi dikelompokkan per brand (untuk dashboard per brand).
     */
    public function sessionsByBrand()
    {
        $sessions = AnalysisSession::where('status', 'done')
            ->orderBy('created_at', 'desc')
            ->get();

        // Kelompokkan per brand_name
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
                'id'             => $session->id,
                'filename'       => $session->filename,
                'total_reviews'  => $session->total_reviews,
                'positive_count' => $session->positive_count,
                'negative_count' => $session->negative_count,
                'n_clusters'     => $session->n_clusters,
                'created_at'     => $session->created_at,
            ];
        }

        return response()->json([
            'status' => 'success',
            'brands' => array_values($brands),
        ]);
    }

    /**
     * GET /api/model-performance
     * Ambil hasil evaluasi model dari file JSON.
     */
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

        return response()->json([
            'status'      => 'success',
            'performance' => $performance,
        ]);
    }
}