<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('analysis_results', function (Blueprint $table) {
            $table->id();
            $table->foreignId('session_id')
                  ->constrained('analysis_sessions')
                  ->onDelete('cascade');          // kalau session dihapus, hasil ikut terhapus
            $table->text('review_text');          // teks ulasan asli
            $table->string('sentiment');          // positive / negative
            $table->integer('cluster_id');        // nomor cluster (0, 1, 2, ...)
            $table->json('keywords');             // ["bagus", "cocok", "suka"]
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('analysis_results');
    }
};