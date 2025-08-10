<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class ScanDefect extends Model
{
    /** @use HasFactory<\Database\Factories\ScanDefectFactory> */
    use HasFactory;

    protected $fillable = [
        'scan_id',
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
    public function scan()
    {
        return $this->belongsTo(Scan::class);
    }

    protected $casts = [
        'box_location' => 'array',
    ];

    protected $attributes = [
        'box_location' => '{"x":0,"y":0,"width":0,"height":0}',
    ];
}
