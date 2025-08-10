<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('daily_scan_logs', function (Blueprint $table) {
            $table->id();
            $table->date('log_date')->index();
            $table->integer('total_analysis')->default(0);
            $table->integer('defect_count')->default(0);
            $table->decimal('defect_rate', 5, 2)->default(0.00);
            $table->integer('good_count')->default(0);
            $table->decimal('good_rate', 5, 2)->default(0.00);
            $table->decimal('avg_processing_time', 8, 3)->nullable();
            $table->decimal('max_processing_time', 8, 3)->nullable();
            $table->decimal('min_processing_time', 8, 3)->nullable();
            $table->decimal('avg_anomaly_score', 6, 5)->nullable();
            $table->decimal('max_anomaly_score', 6, 5)->nullable();
            $table->decimal('min_anomaly_score', 6, 5)->nullable();
            $table->decimal('avg_classification_confidence', 6, 5)->nullable();
            $table->decimal('max_classification_confidence', 6, 5)->nullable();
            $table->decimal('min_classification_confidence', 6, 5)->nullable();
            $table->json('defect_type_distribution')->nullable();
            $table->json('severity_level_distribution')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('daily_scan_logs');
    }
};
