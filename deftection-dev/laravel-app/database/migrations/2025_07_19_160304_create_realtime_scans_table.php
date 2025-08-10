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
        Schema::create('realtime_scans', function (Blueprint $table) {
            $table->id();
            $table->foreignId('realtime_session_id')->constrained('realtime_sessions')->cascadeOnDelete();
            $table->string('filename');
            $table->timestamp('captured_at')->nullable()->index();
            $table->string('annotated_path')->nullable();
            $table->boolean('is_defect')->default(false);
            $table->decimal('anomaly_score', 6, 5)->nullable();
            $table->string('anomaly_confidence_level')->nullable();
            $table->decimal('anomaly_inference_time_ms', 8, 3)->nullable();
            $table->decimal('classification_inference_time_ms', 8, 3)->nullable();
            $table->decimal('preprocessing_time_ms', 8, 3)->nullable();
            $table->decimal('postprocessing_time_ms', 8, 3)->nullable();
            $table->decimal('anomaly_threshold', 6, 5)->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('realtime_scans');
    }
};
