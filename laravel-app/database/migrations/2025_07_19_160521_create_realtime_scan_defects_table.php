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
        Schema::create('realtime_scan_defects', function (Blueprint $table) {
            $table->id();
            $table->foreignId('realtime_scan_id')->constrained('realtime_scans')->cascadeOnDelete();
            $table->string('label')->index();
            $table->decimal('confidence_score', 6, 5)->default(0.00);
            $table->string('severity_level')->default('low');
            $table->decimal('area_percentage', 5, 2)->default(0.00);
            $table->json('box_location')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('realtime_scan_defects');
    }
};
