<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class AnalysisResult extends Model
{
    protected $fillable = [
        'session_id',
        'product_name',
        'review_text',
        'sentiment',
        'confidence',
        'cluster_id',
        'keywords',
    ];

    // Cast keywords dari JSON string ke array otomatis
    protected $casts = [
        'keywords' => 'array',
    ];

    // Setiap result milik satu session
    public function session()
    {
        return $this->belongsTo(AnalysisSession::class, 'session_id');
    }
}