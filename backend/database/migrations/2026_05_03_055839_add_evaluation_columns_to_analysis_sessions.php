<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('analysis_sessions', function (Blueprint $table) {
            $table->float('silhouette_score')->nullable()->after('n_clusters');
            $table->text('confidence_distribution')->nullable()->after('silhouette_score');
        });
    }

    public function down(): void
    {
        Schema::table('analysis_sessions', function (Blueprint $table) {
            $table->dropColumn(['silhouette_score', 'confidence_distribution']);
        });
    }
};