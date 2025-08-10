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
        Schema::create('scan_threats', function (Blueprint $table) {
            $table->id();
            $table->foreignId('scan_id')->constrained('scans')->cascadeOnDelete();
            $table->string('hash')->unique();
            $table->string('status')->default('clean')->index();
            $table->string('risk_level')->default('none')->index();
            $table->json('flags')->nullable();
            $table->json('details')->nullable();
            $table->json('possible_attack')->nullable();
            $table->decimal('processing_time_ms', 8, 3)->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('scan_threats');
    }
};
