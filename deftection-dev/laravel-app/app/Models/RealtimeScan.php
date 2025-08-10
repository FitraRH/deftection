<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class RealtimeScan extends Model
{
    /** @use HasFactory<\Database\Factories\RealtimeScanFactory> */
    use HasFactory;
    //realtime_scans	id	int	realtime_session_id	int	filename	string	captured_at	timestamp	annotated_path	string	is_defect	boolean	anomaly_score	decimal(6,5)	anomaly_confidence_level	string	anomaly_inference_time_ms	decimal(8,3)	classification_interference_time_ms	decimal(8,3)	preprocessing_time_ms	decimal(8,3)	postprocessing_time_ms	decimal(8,3)	anomaly_threshold	decimal(6,5)	created_at	timestamp	updated_at	timestamp
    protected $fillable = [
        'realtime_session_id',
        'filename',
        'captured_at',
        'annotated_path',
        'is_defect',
        'anomaly_score',
        'anomaly_confidence_level',
        'anomaly_inference_time_ms',
        'classification_inference_time_ms',
        'preprocessing_time_ms',
        'postprocessing_time_ms',
        'anomaly_threshold',
    ];
    public static array $sortable = [
        'filename',
        'captured_at',
        'is_defect',
        'anomaly_score',
        'anomaly_confidence_level',
        'anomaly_inference_time_ms',
        'classification_inference_time_ms',
        'preprocessing_time_ms',
        'postprocessing_time_ms',
        'anomaly_threshold',
        'created_at',
        'updated_at'
    ];

    public function realtimeSession()
    {
        return $this->belongsTo(RealtimeSession::class, 'realtime_session_id');
    }
    public function realtimeScanDefects()
    {
        return $this->hasMany(RealtimeScanDefect::class, 'realtime_scan_id');
    }
}
