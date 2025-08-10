<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class RealtimeScanDefect extends Model
{
    /** @use HasFactory<\Database\Factories\RealtimeScanDefectFactory> */
    use HasFactory;

    //realtime_scan_defects	id	int	realtime_scan_id	int	label	string	confidence_score	decimal(6,5)	severity_level	string	area_percentage	decimal(5,2)	box_location	json	created_at	timestamp	updated_at	timestamp
    protected $fillable = [
        'realtime_scan_id',
        'label',
        'confidence_score',
        'severity_level',
        'area_percentage',
        'box_location',
    ];
    public static array $sortable = [
        'label',
        'confidence_score',
        'severity_level',
        'area_percentage',
        'created_at',
        'updated_at'
    ];
    public function realtimeScan()
    {
        return $this->belongsTo(RealtimeScan::class);
    }
}
