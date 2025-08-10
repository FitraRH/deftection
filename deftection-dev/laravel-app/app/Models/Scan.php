<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Scan extends Model
{
    /** @use HasFactory<\Database\Factories\ScanFactory> */
    use HasFactory;
    // scans	id	int	user_id	int	filename	string	original_path	string	annotated_path	string	is_defect	boolean	anomaly_score	decimal(6,5)	anomaly_confidence_level	string	anomaly_inference_time_ms	decimal(8,3)	classification_interference_time_ms	decimal(8,3)	preprocessing_time_ms	decimal(8,3)	postprocessing_time_ms	decimal(8,3)	anomaly_threshold	decimal(6,5)	created_at	datetime	updated_at	datetime
    protected $fillable = [
        'user_id',
        'filename',
        'original_path',
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

    public function user()
    {
        return $this->belongsTo(User::class);
    }
    public function scanDefects()
    {
        return $this->hasMany(ScanDefect::class);
    }
    public function scanThreat()
    {
        return $this->hasOne(ScanThreat::class);
    }
}
