<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class RealtimeSession extends Model
{
    /** @use HasFactory<\Database\Factories\RealtimeSessionFactory> */
    use HasFactory;

    protected $fillable = [
        'user_id',
        'session_status',
        'session_start',
        'session_end',
        'duration_seconds',
        'camera_location',
        'total_frames_processed',
        'throughput_fps',
        'defect_count',
        'defect_rate',
        'good_count',
        'good_rate',
        'avg_processing_time',
        'max_processing_time',
        'min_processing_time',
        'avg_anomaly_score',
        'max_anomaly_score',
        'min_anomaly_score',
        'avg_classification_confidence',
        'max_classification_confidence',
        'min_classification_confidence',
        'defect_type_distribution',
        'severity_level_distribution',
    ];

    public static array $sortable = [
        'session_start',
        'session_end',
        'duration_seconds',
        'camera_location',
        'total_frames_processed',
        'throughput_fps',
        'defect_count',
        'defect_rate',
        'good_count',
        'good_rate',
        'avg_processing_time',
        'max_processing_time',
        'min_processing_time',
        'avg_anomaly_score',
        'max_anomaly_score',
        'min_anomaly_score',
        'avg_classification_confidence',
        'max_classification_confidence',
        'min_classification_confidence',
        'created_at',
        'updated_at'
    ];

    protected $casts = [
        'defect_type_distribution' => 'array',
        'severity_level_distribution' => 'array',
    ];

    public function user()
    {
        return $this->belongsTo(User::class);
    }

    public function realtimeScans()
    {
        return $this->hasMany(RealtimeScan::class, 'realtime_session_id');
    }
}
