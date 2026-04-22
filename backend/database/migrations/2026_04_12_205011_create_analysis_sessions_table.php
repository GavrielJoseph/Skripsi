<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('analysis_sessions', function (Blueprint $table) {
            $table->id();
            $table->string('filename');           // nama file CSV yang diupload
            $table->string('status')->default('processing'); // processing / done / failed
            $table->integer('total_reviews')->default(0);    // jumlah ulasan
            $table->integer('positive_count')->default(0);   // jumlah sentimen positif
            $table->integer('negative_count')->default(0);   // jumlah sentimen negatif
            $table->integer('n_clusters')->default(0);       // jumlah cluster yang terbentuk
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('analysis_sessions');
    }
};