<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class DailyScanLog extends Model
{
    /** @use HasFactory<\Database\Factories\DailyScanLogFactory> */
    use HasFactory;

    //daily_scan_logs	id	int	log_date	date	total_analysis	int	defect_count	int	defect_rate	decimal(5,2)	good_count	int	good_rate	decimal(5,2)	avg_processing_time	decimal(8,3)	max_processing_time	decimal(8,3)	min_processing_time	decimal(8,3)	avg_anomaly_score	decimal(6,5)	max_anomaly_score	decimal(6,5)	min_anomaly_score	decimal(6,5)	avg_classification_confident	decimal(6,5)	max_classification_confident	decimal(6,5)	min_classification_confident	decimal(6,5)	defect_type_distribution	json	severity_level_distribution	json	created_at	timestamp	updated_at	timestamp
    protected $fillable = [
        'log_date',
        'total_analysis',
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
        'log_date',
        'total_analysis',
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
}
