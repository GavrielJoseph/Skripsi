<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\AnalysisController;

// Analisis — upload CSV dan jalankan ML
Route::post('/analyze', [AnalysisController::class, 'analyze']);
Route::post('/scrape-and-analyze', [AnalysisController::class, 'scrapeAndAnalyze']);

// Ambil hasil analisis berdasarkan session ID
Route::get('/results/{id}', [AnalysisController::class, 'results']);

// Ambil semua history sesi (halaman dashboard)
Route::get('/sessions', [AnalysisController::class, 'sessions']);

// Dashboard per brand
Route::get('/sessions/by-brand', [AnalysisController::class, 'sessionsByBrand']);

// Model performance
Route::get('/model-performance', [AnalysisController::class, 'modelPerformance']);

// Hapus Data (Delete)
Route::delete('/brands/{brand_name}', [AnalysisController::class, 'deleteBrand']);
Route::delete('/sessions/{id}', [AnalysisController::class, 'deleteSession']);

Route::get('/sessions/{id}/download-csv', [AnalysisController::class, 'downloadCsv']);