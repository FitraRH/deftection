<?php

namespace App\Services;

use App\Models\Scan;
use App\Models\DefectType;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;

class ReportService
{
    /**
     * Prepare data for single scan report
     */
    public function prepareSingleScanReport(Scan $scan): array
    {
        // Load relationships - they might be empty collections/null
        $scan->load(['user', 'scanDefects', 'scanThreat']);

        // Get defect types for explanations
        $defectTypes = DefectType::all()->keyBy('slug');

        // Process defects with explanations (handle empty collection)
        $processedDefects = $this->processDefectsWithExplanations($scan->scanDefects ?? collect(), $defectTypes);

        // Generate charts data
        $chartsData = $this->generateSingleScanCharts($scan);

        // Generate chart images
        $chartImages = $this->generateChartImages($chartsData);

        // Get image paths
        $imagePaths = $this->getImagePaths($scan);

        return [
            'scan' => $scan,
            'defects' => $processedDefects,  // Will be empty collection if no defects
            'chartsData' => $chartsData,
            'chartImages' => $chartImages, // NEW: Base64 chart images
            'imagePaths' => $imagePaths,
            'summary' => $this->generateScanSummary($scan),
            'generatedAt' => now(),
            'reportTitle' => "Scan Analysis Report - {$scan->filename}",
        ];
    }

    /**
     * Process defects and match with defect type explanations
     */
    private function processDefectsWithExplanations(Collection $scanDefects, Collection $defectTypes): Collection
    {
        return $scanDefects->map(function ($defect) use ($defectTypes) {
            $defectSlug = str()->slug($defect->label);
            $defectType = $defectTypes->get($defectSlug);

            // Handle double-encoded JSON for box_location
            $boxLocation = $defect->box_location;
            if (is_string($boxLocation)) {
                // Try to decode if it's a string (might be double-encoded)
                $decoded = json_decode($boxLocation, true);
                if (json_last_error() === JSON_ERROR_NONE) {
                    $boxLocation = $decoded;
                } else {
                    // If first decode fails, try double-decode
                    $doubleDecoded = json_decode(json_decode($boxLocation, true), true);
                    if (json_last_error() === JSON_ERROR_NONE) {
                        $boxLocation = $doubleDecoded;
                    } else {
                        // Fallback to default values
                        $boxLocation = ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0];
                    }
                }
            }

            return [
                'id' => $defect->id,
                'label' => $defect->label,
                'confidence_score' => $defect->confidence_score,
                'severity_level' => $defect->severity_level,
                'area_percentage' => $defect->area_percentage,
                'box_location' => $boxLocation,
                'explanation' => $defectType ? $defectType->description : 'No detailed explanation available for this defect type.',
                'defect_type_found' => (bool) $defectType,
                'created_at' => $defect->created_at,
            ];
        });
    }

    /**
     * Generate chart data for single scan
     */
    private function generateSingleScanCharts(Scan $scan): array
    {
        $defects = $scan->scanDefects;

        // Defect distribution by type
        $defectDistribution = $defects->groupBy('label')->map(function ($group) {
            return [
                'label' => $group->first()->label,
                'count' => $group->count(),
                'avg_confidence' => round($group->avg('confidence_score'), 4),
                'total_area' => round($group->sum('area_percentage'), 2),
            ];
        })->values()->toArray(); // Convert to array

        // Severity distribution
        $severityDistribution = $defects->groupBy('severity_level')->map(function ($group, $severity) use ($defects) {
            return [
                'severity' => $severity ?: 'Unknown',
                'count' => $group->count(),
                'percentage' => $defects->count() > 0 ? round(($group->count() / $defects->count()) * 100, 1) : 0,
            ];
        })->values()->toArray(); // Convert to array

        // Processing time breakdown
        $processingTimes = [
            'preprocessing' => $scan->preprocessing_time_ms ?? 0,
            'anomaly_inference' => $scan->anomaly_inference_time_ms ?? 0,
            'classification_inference' => $scan->classification_inference_time_ms ?? 0,
            'postprocessing' => $scan->postprocessing_time_ms ?? 0,
        ];

        $totalProcessingTime = array_sum(array_filter($processingTimes));

        $processingTimeBreakdown = collect($processingTimes)->map(function ($time, $stage) use ($totalProcessingTime) {
            return [
                'stage' => str_replace('_', ' ', ucwords(str_replace('_', ' ', $stage))),
                'time_ms' => $time,
                'percentage' => $totalProcessingTime > 0 ? round(($time / $totalProcessingTime) * 100, 1) : 0,
            ];
        })->filter(fn($item) => $item['time_ms'] > 0)->values()->toArray(); // Convert to array

        return [
            'defect_distribution' => $defectDistribution,
            'severity_distribution' => $severityDistribution,
            'processing_time_breakdown' => $processingTimeBreakdown,
            'total_processing_time' => $totalProcessingTime,
        ];
    }

    /**
     * Get image paths for the report
     */
    private function getImagePaths(Scan $scan): array
    {
        return [
            'original' => Storage::disk('public')->exists($scan->original_path)
                ? Storage::disk('public')->path($scan->original_path)
                : null,
            'annotated' => Storage::disk('public')->exists($scan->annotated_path)
                ? Storage::disk('public')->path($scan->annotated_path)
                : null,
        ];
    }

    /**
     * Generate scan summary
     */
    private function generateScanSummary(Scan $scan): array
    {
        $defects = $scan->scanDefects;
        $threat = $scan->scanThreat;

        return [
            'overall_status' => $scan->is_defect ? 'DEFECTIVE' : 'GOOD',
            'anomaly_score' => $scan->anomaly_score,
            'confidence_level' => $scan->anomaly_confidence_level,
            'threshold_used' => $scan->anomaly_threshold,
            'defects_found' => $defects->count(),
            'unique_defect_types' => $defects->pluck('label')->unique()->count(),
            'severity_breakdown' => $defects->groupBy('severity_level')->map->count(),
            'total_affected_area' => round($defects->sum('area_percentage'), 2),
            'avg_defect_confidence' => $defects->count() > 0 ? round($defects->avg('confidence_score'), 4) : 0,
            'threat_status' => $threat ? $threat->status : 'not_scanned',
            'risk_level' => $threat ? $threat->risk_level : 'none',
            'scan_date' => $scan->created_at,
            'processing_performance' => [
                'total_time' => ($scan->preprocessing_time_ms ?? 0) +
                    ($scan->anomaly_inference_time_ms ?? 0) +
                    ($scan->classification_inference_time_ms ?? 0) +
                    ($scan->postprocessing_time_ms ?? 0),
                'fastest_stage' => $this->getFastestProcessingStage($scan),
                'slowest_stage' => $this->getSlowestProcessingStage($scan),
            ],
        ];
    }

    /**
     * Get fastest processing stage
     */
    private function getFastestProcessingStage(Scan $scan): string
    {
        $times = [
            'preprocessing' => $scan->preprocessing_time_ms ?? PHP_FLOAT_MAX,
            'anomaly_inference' => $scan->anomaly_inference_time_ms ?? PHP_FLOAT_MAX,
            'classification_inference' => $scan->classification_inference_time_ms ?? PHP_FLOAT_MAX,
            'postprocessing' => $scan->postprocessing_time_ms ?? PHP_FLOAT_MAX,
        ];

        return array_search(min($times), $times);
    }

    /**
     * Get slowest processing stage
     */
    private function getSlowestProcessingStage(Scan $scan): string
    {
        $times = [
            'preprocessing' => $scan->preprocessing_time_ms ?? 0,
            'anomaly_inference' => $scan->anomaly_inference_time_ms ?? 0,
            'classification_inference' => $scan->classification_inference_time_ms ?? 0,
            'postprocessing' => $scan->postprocessing_time_ms ?? 0,
        ];

        return array_search(max($times), $times);
    }

    /**
     * Generate chart images as base64
     */
    private function generateChartImages(array $chartsData): array
    {
        // DEBUG: Check data types
        Log::info('Chart data debug:', [
            'defect_distribution_type' => gettype($chartsData['defect_distribution']),
            'defect_distribution_is_array' => is_array($chartsData['defect_distribution']),
            'defect_distribution_count' => count($chartsData['defect_distribution']),
            'defect_distribution_sample' => array_slice($chartsData['defect_distribution'], 0, 2),

            'severity_distribution_type' => gettype($chartsData['severity_distribution']),
            'severity_distribution_is_array' => is_array($chartsData['severity_distribution']),
            'severity_distribution_count' => count($chartsData['severity_distribution']),

            'processing_time_breakdown_type' => gettype($chartsData['processing_time_breakdown']),
            'processing_time_breakdown_is_array' => is_array($chartsData['processing_time_breakdown']),
            'processing_time_breakdown_count' => count($chartsData['processing_time_breakdown']),
        ]);

        $chartService = app(ChartGenerationService::class);

        return [
            'defect_distribution' => !empty($chartsData['defect_distribution'])
                ? $chartService->generateDefectDistributionChart($chartsData['defect_distribution'])
                : $chartService->generatePlaceholderChart('No Defects Found', 'All quality checks passed'),

            'severity_distribution' => !empty($chartsData['severity_distribution'])
                ? $chartService->generateSeverityDistributionChart($chartsData['severity_distribution'])
                : $chartService->generatePlaceholderChart('No Severity Data', 'No defects to analyze'),

            'processing_time' => !empty($chartsData['processing_time_breakdown'])
                ? $chartService->generateProcessingTimeChart($chartsData['processing_time_breakdown'])
                : $chartService->generatePlaceholderChart('Processing Time Chart', 'Total: ' . number_format($chartsData['total_processing_time'], 1) . 'ms'),
        ];
    }
}
