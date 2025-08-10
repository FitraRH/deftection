<?php

namespace App\Services;

use App\Models\DefectType;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class DefectTypeQueryService
{
    public function getAllDefectTypes(array $filters): LengthAwarePaginator
    {
        return DefectType::query()
            ->where(function ($query) use ($filters) {
                $query->where('name', 'like', '%' . $filters['search'] . '%')
                    ->orWhere('description', 'like', '%' . $filters['search'] . '%');
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
            ->through(fn($defectType) => [
                'id' => $defectType->id,
                'name' => $defectType->name,
                'slug' => $defectType->slug,
                'description' => $defectType->description,
                'created_at' => $defectType->created_at->format('Y-m-d H:i:s'),
                'updated_at' => $defectType->updated_at->format('Y-m-d H:i:s'),
            ]);
    }

    public function getChecksumAllDefectTypes(): string
    {
        $defectTypeData = DefectType::selectRaw('
            COUNT(*) as total_defect_types,
            MAX(updated_at) as latest_update
        ')->first();

        $checksumData = [
            'total_defect_types' => $defectTypeData->total_defect_types ?? 0,
            'latest_update' => $defectTypeData->latest_update
                ? Carbon::parse($defectTypeData->latest_update)->timestamp
                : 0,
        ];

        return md5(json_encode($checksumData));
    }

    public function filters(Request $request): array
    {
        $validated = $request->validate([
            'search' => 'nullable|string|max:255',
            'sort_by' => 'nullable|string|in:' . implode(',', DefectType::$sortable),
            'sort_dir' => 'nullable|string|in:asc,desc',
            'per_page' => 'nullable|integer|min:1|max:100',
            'page' => 'nullable|integer|min:1',
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
            'dateFrom' => $validated['date_from'] ?? null,
            'dateTo' => $validated['date_to'] ?? null,
        ];
    }
}
