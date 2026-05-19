<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('analysis_sessions', function (Blueprint $table) {
            // Simpan topic_overview dari predict.py:
            // distribusi sentimen per aspek bisnis + neg_share + negative_keywords + negative_samples
            $table->json('topic_overview')->nullable()->after('confidence_distribution');

            // Simpan cluster_summary dari predict.py:
            // label, count, positive, negative, top_keywords, topic_breakdown, sample_reviews
            // Disimpan di sini agar tidak perlu rebuild dari analysis_results setiap request
            $table->json('cluster_summary')->nullable()->after('topic_overview');
        });
    }

    public function down(): void
    {
        Schema::table('analysis_sessions', function (Blueprint $table) {
            $table->dropColumn(['topic_overview', 'cluster_summary']);
        });
    }
};