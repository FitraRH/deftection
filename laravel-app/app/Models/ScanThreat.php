<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class ScanThreat extends Model
{
    /** @use HasFactory<\Database\Factories\ScanThreatFactory> */
    use HasFactory;

    //scan_threats	id	int	scan_id	int	hash	string	status	string	risk_level	string	flags	json	details	json	possible_attack	json	processing_time_ms	decimal(8,3)	created_at	timestamp	updated_at	timestamp
    protected $fillable = [
        'scan_id',
        'hash',
        'status',
        'risk_level',
        'flags',
        'details',
        'possible_attack',
        'processing_time_ms',
    ];
    public static array $sortable = [
        'hash',
        'status',
        'risk_level',
        'processing_time_ms',
        'created_at',
        'updated_at'
    ];
    public function scan()
    {
        return $this->belongsTo(Scan::class);
    }

    protected $casts = [
        'flags' => 'array',
        'details' => 'array',
        'possible_attack' => 'array',
    ];
}
