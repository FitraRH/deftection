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

        //realtime_sessions	id	int	user_id	int	session_status	string	session_start	timestamp	session_end	timestamp	duration_seconds	int	camera_location	string	total_frames_processed	int	throughput_fps	decimal(6,3)	defect_count	int	defect_rate	decimal(5,2)	good_count	int	good_rate	decimal(5,2)	avg_processing_time	decimal(8,3)	max_processing_time	decimal(8,3)	min_processing_time	decimal(8,3)	avg_anomaly_score	decimal(6,5)	max_anomaly_score	decimal(6,5)	min_anomaly_score	decimal(6,5)	avg_classification_confident	decimal(6,5)	max_classification_confident	decimal(6,5)	min_classification_confident	decimal(6,5)	defect_type_distribution	json	severity_level_distribution	json	report_session_path	string	created_at	timestamp	updated_at	timstamp
        Schema::create('realtime_sessions', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->nullOnDelete();
            $table->string('session_status')->default('aborted')->index();
            $table->timestamp('session_start');
            $table->timestamp('session_end')->nullable();
            $table->integer('duration_seconds')->nullable();
            $table->string('camera_location')->nullable();
            $table->integer('total_frames_processed')->default(0);
            $table->decimal('throughput_fps', 6, 3)->default(0.000);
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
        Schema::dropIfExists('realtime_sessions');
    }
};
