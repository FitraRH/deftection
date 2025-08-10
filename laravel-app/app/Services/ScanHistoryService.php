<?php

namespace App\Services;

use App\Models\Scan;
use App\Models\ScanDefect;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class ScanHistoryService
{
    public function getScans(array $filters): LengthAwarePaginator
    {
        return Scan::query()
            ->where('user_id', auth()->id()) // Filter by authenticated user
            ->with(['scanDefects:scan_id,label,confidence_score,severity_level'])
            ->selectRaw('
                *,
                DATE_FORMAT(created_at, "%d-%m-%Y") AS analysis_date,
                DATE_FORMAT(created_at, "%H:%i") AS analysis_time,
                COALESCE(anomaly_inference_time_ms, 0) + COALESCE(classification_inference_time_ms, 0) +
                COALESCE(preprocessing_time_ms, 0) + COALESCE(postprocessing_time_ms, 0) AS total_processing_time,
                COALESCE((
                    SELECT COUNT(*) FROM scan_defects WHERE scan_id = scans.id
                ), 0) AS total_defect,
                COALESCE((
                    SELECT GROUP_CONCAT(DISTINCT label ORDER BY label SEPARATOR ", ") 
                    FROM scan_defects 
                    WHERE scan_id = scans.id
                ), "No defects") AS defect_scanned,
                CASE WHEN is_defect = 1 THEN "defect" ELSE "good" END AS status
            ')
            ->when(!empty($filters['status']), function ($query) use ($filters) {
                $statusValues = collect($filters['status'])->map(fn($s) => $s === 'defect' ? 1 : 0)->toArray();
                $query->whereIn('is_defect', $statusValues);
            })
            ->when(!empty($filters['defectTypes']), function ($query) use ($filters) {
                $query->whereHas('scanDefects', function ($q) use ($filters) {
                    $q->whereIn('label', $filters['defectTypes']);
                });
            })
            ->when(!empty($filters['search']), function ($query) use ($filters) {
                $searchTerm = trim($filters['search']);
                $query->where(function ($subQuery) use ($searchTerm) {
                    $subQuery->where('filename', 'like', '%' . $searchTerm . '%')
                        ->orWhere('anomaly_confidence_level', 'like', '%' . $searchTerm . '%')
                        ->orWhere('anomaly_score', 'like', '%' . $searchTerm . '%')
                        ->orWhereHas('scanDefects', function ($q) use ($searchTerm) {
                            $q->where('label', 'like', '%' . $searchTerm . '%');
                        });
                });
            })
            ->when(!empty($filters['dateFrom']), function ($query) use ($filters) {
                $query->whereDate('created_at', '>=', $filters['dateFrom']);
            })
            ->when(!empty($filters['dateTo']), function ($query) use ($filters) {
                $query->whereDate('created_at', '<=', $filters['dateTo']);
            })
            ->orderBy($filters['sortBy'], $filters['sortDir'])
            ->paginate($filters['perPage'], ['*'], 'page', $filters['page'])
            ->withQueryString()
            ->through(fn($scan) => [
                'id' => $scan->id,
                'filename' => $scan->filename,
                'original_path' => $scan->original_path,
                'annotated_path' => $scan->annotated_path,
                'status' => $scan->status,
                'anomaly_confidence_level' => $scan->anomaly_confidence_level,
                'anomaly_score' => number_format($scan->anomaly_score, 5),
                'analysis_date' => $scan->analysis_date,
                'analysis_time' => $scan->analysis_time,
                'total_processing_time' => number_format($scan->total_processing_time, 3),
                'defect_scanned' => $scan->defect_scanned,
                'total_defect' => $scan->total_defect,
                'created_at' => $scan->created_at->toISOString(),
                'updated_at' => $scan->updated_at->toISOString(),
            ]);
    }


    /**
     * Alternative: Even more stable checksum that only includes data that actually affects the UI
     */
    public function getStableDataChecksum($userId): string
    {
        // Get aggregated data that affects what user sees
        $scanData = Scan::where('user_id', $userId)
            ->selectRaw('
                COUNT(*) as total_scans,
                MAX(updated_at) as latest_update,
                SUM(CASE WHEN is_defect = 1 THEN 1 ELSE 0 END) as defect_count,
                SUM(CASE WHEN is_defect = 0 THEN 1 ELSE 0 END) as good_count
            ')
            ->first();

        $defectData = ScanDefect::whereHas('scan', function ($query) use ($userId) {
            $query->where('user_id', $userId);
        })
            ->selectRaw('
            COUNT(*) as total_defects,
            MAX(updated_at) as latest_defect_update,
            GROUP_CONCAT(DISTINCT label ORDER BY label) as defect_types
        ')
            ->first();

        // Only include data that affects the UI display
        $checksumData = [
            'total_scans' => $scanData->total_scans ?? 0,
            'defect_count' => $scanData->defect_count ?? 0,
            'good_count' => $scanData->good_count ?? 0,
            'total_defects' => $defectData->total_defects ?? 0,
            'defect_types' => $defectData->defect_types ?? '',

            'latest_scan_update' => $scanData->latest_update
                ? Carbon::parse($scanData->latest_update)->timestamp
                : 0,
            'latest_defect_update' => $defectData->latest_defect_update
                ? Carbon::parse($defectData->latest_defect_update)->timestamp
                : 0,
        ];

        return md5(json_encode($checksumData));
    }
    /**
     * Get all defect types with counts for the current authenticated user
     */
    public function getDefectTypesWithCounts(): array
    {
        return ScanDefect::select('label')
            ->selectRaw('COUNT(DISTINCT scan_defects.scan_id) as user_scan_count')
            ->join('scans', 'scan_defects.scan_id', '=', 'scans.id')
            ->where('scans.user_id', auth()->id())
            ->groupBy('label')
            ->orderBy('user_scan_count', 'desc')
            ->orderBy('label', 'asc')
            ->get()
            ->map(function ($defect) {
                return [
                    'label' => ucwords(str_replace(['_', '-'], ' ', $defect->label)),
                    'value' => $defect->label,
                    'count' => $defect->user_scan_count,
                ];
            })
            ->toArray();
    }

    /**
     * Get status counts for the current authenticated user
     */
    public function getStatusCounts(): array
    {
        $counts = Scan::where('user_id', auth()->id())
            ->selectRaw('
                SUM(CASE WHEN is_defect = 1 THEN 1 ELSE 0 END) as defect_count,
                SUM(CASE WHEN is_defect = 0 THEN 1 ELSE 0 END) as good_count,
                COUNT(*) as total_count
            ')
            ->first();

        return [
            [
                'label' => 'Good',
                'value' => 'good',
                'count' => $counts->good_count ?? 0,
            ],
            [
                'label' => 'Defect',
                'value' => 'defect',
                'count' => $counts->defect_count ?? 0,
            ]
        ];
    }

    /**
     * Get and validate filters from request
     */
    public function filters(Request $request): array
    {
        // Validate request parameters
        $validated = $request->validate([
            'search' => 'nullable|string|max:255',
            'sort_by' => 'nullable|string|in:' . implode(',', Scan::$sortable),
            'sort_dir' => 'nullable|string|in:asc,desc',
            'per_page' => 'nullable|integer|min:1|max:100',
            'page' => 'nullable|integer|min:1',
            'status' => 'nullable|array',
            'status.*' => 'string|in:good,defect',
            'defect_types' => 'nullable|array',
            'defect_types.*' => 'string',
            'date_from' => 'nullable|date',
            'date_to' => 'nullable|date|after_or_equal:date_from',
        ]);

        return [
            'search' => $validated['search'] ?? '',
            'sortBy' => $validated['sort_by'] ?? 'created_at',
            'sortDir' => $validated['sort_dir'] ?? 'desc',
            'perPage' => $validated['per_page'] ?? 10,
            'page' => $validated['page'] ?? 1,
            'status' => $validated['status'] ?? [],
            'defectTypes' => $validated['defect_types'] ?? [],
            'dateFrom' => $validated['date_from'] ?? null,
            'dateTo' => $validated['date_to'] ?? null,
        ];
    }
}
