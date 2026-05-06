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
    ];

    // Satu session punya banyak results
    public function results()
    {
        return $this->hasMany(AnalysisResult::class, 'session_id');
    }
}