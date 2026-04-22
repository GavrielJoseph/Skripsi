<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\AnalysisController;

// Analisis — upload CSV dan jalankan ML
Route::post('/analyze', [AnalysisController::class, 'analyze']);

// Ambil hasil analisis berdasarkan session ID
Route::get('/results/{id}', [AnalysisController::class, 'results']);

// Ambil semua history sesi (untuk dashboard)
Route::get('/sessions', [AnalysisController::class, 'sessions']);

// Dashboard per brand
Route::get('/sessions/by-brand', [AnalysisController::class, 'sessionsByBrand']);

// Model performance
Route::get('/model-performance', [AnalysisController::class, 'modelPerformance']);