<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('analysis_sessions', function (Blueprint $table) {
            $table->string('brand_name')->nullable()->after('filename');
        });
    }

    public function down(): void
    {
        Schema::table('analysis_sessions', function (Blueprint $table) {
            $table->dropColumn('brand_name');
        });
    }
};