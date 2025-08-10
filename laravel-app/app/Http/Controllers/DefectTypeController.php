<?php

namespace App\Http\Controllers;

use Inertia\Inertia;
use App\Models\DefectType;
use Illuminate\Support\Str;
use Illuminate\Http\Request;
use Illuminate\Validation\Rule;
use Illuminate\Support\Facades\Log;
use App\Services\DefectTypeQueryService;


class DefectTypeController extends Controller
{

    public function __construct(
        protected DefectTypeQueryService $queryService
    ) {}


    public function index(Request $request)
    {
        try {
            $filters = $this->queryService->filters($request);
            $defectTypes = $this->queryService->getAllDefectTypes($filters);
            $initialChecksum = $this->queryService->getChecksumAllDefectTypes();

            return Inertia::render(
                'Database/DefectType/Index',
                [
                    'defect_types' => $defectTypes->items(),
                    'filters' => [
                        "current" => $filters,
                        'options' => [
                            'perPageOptions' => [5, 10, 25, 50, 100],
                        ]
                    ],
                    'meta' => [
                        'total' => $defectTypes->total(),
                        'per_page' => $defectTypes->perPage(),
                        'current_page' => $defectTypes->currentPage(),
                        'last_page' => $defectTypes->lastPage(),
                        'from' => $defectTypes->firstItem(),
                        'to' => $defectTypes->lastItem(),
                    ],
                    'initialChecksum' => $initialChecksum
                ]
            );
        } catch (\Exception $e) {
            Log::error('Error fetching defect types: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
            ]);
            return Inertia::render(
                'Database/DefectType/Index',
                [
                    'defect_types' => [],
                    'filters' => [
                        "current" => [],
                        'options' => [
                            'perPageOptions' => [5, 10, 25, 50, 100],
                        ]
                    ],
                    'meta' => [
                        'total' => 0,
                        'per_page' => 0,
                        'current_page' => 0,
                        'last_page' => 0,
                        'from' => 0,
                        'to' => 0,
                    ],
                    'initialChecksum' => ''
                ]
            );
        }
    }

    public function indexCheck(Request $request)
    {
        try {
            $lastChecksum = $request->input('checksum', '');
            $currentChecksum = $this->queryService->getChecksumAllDefectTypes();
            $hasChanges = $lastChecksum !== $currentChecksum;

            return response()->json([
                'has_changes' => $hasChanges,
                'checksum' => $currentChecksum,
                'timestamp' => now()->toISOString(),
            ]);
        } catch (\Exception $e) {
            Log::error('Error checking updates: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
                'last_checksum' => $request->input('checksum', ''),
            ]);
            return response()->json([
                'error' => 'Failed to check updates',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }

    // data for polling
    public function indexApi(Request $request)
    {
        try {
            $filters = $this->queryService->filters($request);
            $defectTypes = $this->queryService->getAllDefectTypes($filters);
            $currentChecksum = $this->queryService->getChecksumAllDefectTypes();

            $data = [
                'defect_types' => $defectTypes->items(),
                'filters' => [
                    'current' => $filters,
                    'options' => [
                        'perPageOptions' => [5, 10, 25, 50, 100],
                    ]
                ],
                'meta' => [
                    'total' => $defectTypes->total(),
                    'per_page' => $defectTypes->perPage(),
                    'current_page' => $defectTypes->currentPage(),
                    'last_page' => $defectTypes->lastPage(),
                    'from' => $defectTypes->firstItem(),
                    'to' => $defectTypes->lastItem(),
                ],
                'checksum' => $currentChecksum,
                'timestamp' => now()->toISOString(),
            ];

            return response()->json($data);
        } catch (\Exception $e) {
            Log::error('Error in poll data: ' . $e->getMessage(), [
                'filters' => $request->all(),
            ]);
            return response()->json([
                'user_id' => auth()->id(),
                'error' => 'Failed to fetch data',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }

    public function indexRefresh(Request $request)
    {
        try {
            // This is essentially the same as pollData but ensures fresh data
            return $this->indexApi($request);
        } catch (\Exception $e) {
            Log::error('Error in force refresh: ' . $e->getMessage());
            return response()->json(['error' => 'Failed to refresh data'], 500);
        }
    }

    public function store(Request $request)
    {
        $request->validate([
            'name' => ['required', 'string', 'max:255', 'unique:defect_types,name'],
            'description' => ['required', 'string', 'max:255'],
        ]);

        $defectTypes = DefectType::create([
            'name' => $request->name,
            'slug' => Str::slug($request->name),
            'description' => $request->description
        ]);

        // Log successful Defect type creation
        Log::info('Defect type created', [
            'defect_type_id' => $defectTypes->id,
            'name' => $defectTypes->name,
            'slug' => $defectTypes->slug,
            'description' => $defectTypes->description,
            'user_id' => auth()->id()
        ]);

        return redirect()
            ->back()
            ->with('success', 'Defect type created successfully.');
    }

    public function showApi(DefectType $defectType)
    {
        try {
            return response()->json([
                'success' => true,
                'message' => 'Defect type retrieved successfully',
                'data' => $defectType
            ]);
        } catch (\Exception $e) {
            Log::error('Error showing defect type: ' . $e->getMessage(), [
                'defect_type_id' => $defectType->id,
                'user_id' => auth()->id(),
            ]);
            return response()->json([
                'success' => false,
                'message' => 'Failed to show defect type',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, DefectType $defectType)
    {
        $request->validate([
            'name' => [
                'required',
                'string',
                'max:255',
                Rule::unique('defect_types', 'name')->ignore($defectType->slug, 'slug'),
            ],
            'description' => ['required', 'string', 'max:255'],
        ]);

        // Store original values for logging
        $originalName = $defectType->name;
        $originalSlug = $defectType->slug;
        $originalDescription = $defectType->description;

        $defectType->update([
            'name' => $request->name,
            'slug' => Str::slug($request->name),
            'description' => $request->description
        ]);

        // Log successful Defect type update
        Log::info('Defect type updated', [
            'defect_type_id' => $defectType->id,
            'old_name' => $originalName,
            'new_name' => $defectType->name,
            'old_slug' => $originalSlug,
            'new_slug' => $defectType->slug,
            'old_description' => $originalDescription,
            'new_description' => $defectType->description,
            'user_id' => auth()->id()
        ]);


        return redirect()
            ->back()
            ->with('success', 'Defect type updated successfully.');
    }

    public function destroy(DefectType $defectType)
    {
        // Store defect type details for logging before deletion
        $defectTypesData = [
            'defect_type_id' => $defectType->id,
            'name' => $defectType->name,
            'slug' => $defectType->slug,
            'description' => $defectType->description,
            'user_id' => auth()->id()
        ];

        $defectType->delete();

        // Log successful defect type deletion
        Log::info('Defect type deleted', $defectTypesData);

        return redirect()
            ->back()
            ->with('success', 'Defect type deleted successfully.');
    }
}
