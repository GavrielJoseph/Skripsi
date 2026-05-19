<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class AnalysisSession extends Model
{
    protected $fillable = [
        'filename',
        'brand_name',
        'status',
        'total_reviews',
        'positive_count',
        'negative_count',
        'n_clusters',
        'silhouette_score',
        'confidence_distribution',
        'topic_overview',    // BARU: distribusi sentimen per aspek bisnis
        'cluster_summary',   // BARU: data cluster lengkap dari predict.py
    ];

    // Cast otomatis JSON string ↔ array PHP
    protected $casts = [
        'confidence_distribution' => 'array',
        'topic_overview'          => 'array',
        'cluster_summary'         => 'array',
    ];

    public function results()
    {
        return $this->hasMany(AnalysisResult::class, 'session_id');
    }
}