<?php

namespace App\Services;

use App\Models\Scan;
use Illuminate\Support\Facades\Storage;

class ScanDetailService
{
    /**
     * Transform scan data for the analysis detail page
     */
    public function transformScanForAnalysis(Scan $scan): array
    {
        // Load relationships
        $scan->load(['scanDefects', 'scanThreat', 'user']);

        // Calculate derived values
        $totalProcessingTime = $this->calculateTotalProcessingTime($scan);
        $aiInferenceTime = $this->calculateAiInferenceTime($scan);
        $status = $scan->is_defect ? 'defect' : 'good';
        $aiConfidence = round($scan->anomaly_score * 100);
        $confidenceLevel = $this->getConfidenceLevel($scan->anomaly_score);
        $analysisQuality = $this->getAnalysisQuality($confidenceLevel);
        $processingSpeed = $this->calculateProcessingSpeed($totalProcessingTime);

        // Get image URLs
        $originalImageUrl = $scan->original_path ? Storage::url($scan->original_path) : '';
        $analyzedImageUrl = $scan->annotated_path ? Storage::url($scan->annotated_path) : '';

        // Process defects data
        $defects = $this->transformDefects($scan->scanDefects);

        return [
            'id' => $scan->id,
            'status' => $status,
            'summary' => $this->buildSummary($scan, $totalProcessingTime, $confidenceLevel, $analysisQuality, $processingSpeed, $aiConfidence),
            'visuals' => $this->buildVisuals($status, $originalImageUrl, $analyzedImageUrl),
            'defects' => $defects,
            'performance' => $this->buildPerformance($scan, $aiInferenceTime),
            'technical' => $this->buildTechnical($scan, $totalProcessingTime, $aiInferenceTime, $aiConfidence, $defects),
        ];
    }

    /**
     * Calculate total processing time in milliseconds
     */
    private function calculateTotalProcessingTime(Scan $scan): float
    {
        return ($scan->preprocessing_time_ms ?? 0) +
            ($scan->anomaly_inference_time_ms ?? 0) +
            ($scan->classification_inference_time_ms ?? 0) +
            ($scan->postprocessing_time_ms ?? 0);
    }

    /**
     * Calculate AI inference time (anomaly + classification)
     */
    private function calculateAiInferenceTime(Scan $scan): float
    {
        return ($scan->anomaly_inference_time_ms ?? 0) +
            ($scan->classification_inference_time_ms ?? 0);
    }

    /**
     * Calculate processing speed (scans per second)
     */
    private function calculateProcessingSpeed(float $totalProcessingTime): float
    {
        return $totalProcessingTime > 0 ? round(1000 / $totalProcessingTime, 1) : 0;
    }

    /**
     * Determine confidence level based on anomaly score
     */
    private function getConfidenceLevel(float $anomalyScore): string
    {
        return match (true) {
            $anomalyScore >= 0.8 => 'Very High',
            $anomalyScore >= 0.7 => 'High',
            $anomalyScore >= 0.5 => 'Medium',
            $anomalyScore >= 0.3 => 'Low',
            default => 'Very Low'
        };
    }

    /**
     * Determine analysis quality based on confidence level
     */
    private function getAnalysisQuality(string $confidenceLevel): string
    {
        return match ($confidenceLevel) {
            'Very High', 'High' => 'High',
            'Medium' => 'Medium',
            'Low', 'Very Low' => 'Low',
            default => 'Medium'
        };
    }

    /**
     * Format defect label for display
     */
    private function formatDefectLabel(string $label): string
    {
        return ucwords(str_replace('_', ' ', $label));
    }

    /**
     * Get image dimensions from stored file
     */
    private function getImageDimensions(?string $imagePath): string
    {
        try {
            if (!$imagePath) return 'Unknown';

            $fullPath = Storage::disk('public')->path($imagePath);
            if (!file_exists($fullPath)) return 'Unknown';

            $imageInfo = getimagesize($fullPath);
            if (!$imageInfo) return 'Unknown';

            return $imageInfo[0] . 'x' . $imageInfo[1];
        } catch (\Exception $e) {
            return 'Unknown';
        }
    }

    /**
     * Transform defects collection for frontend
     */
    private function transformDefects($scanDefects): array
    {
        return $scanDefects->map(function ($defect) {
            return [
                'name' => $this->formatDefectLabel($defect->label),
                'confidence' => round($defect->confidence_score * 100),
                'coverage' => round($defect->area_percentage, 1),
                'regions' => 1 // Assuming each defect record represents one region
            ];
        })->toArray();
    }

    /**
     * Build summary section data
     */
    private function buildSummary(
        Scan $scan,
        float $totalProcessingTime,
        string $confidenceLevel,
        string $analysisQuality,
        float $processingSpeed,
        int $aiConfidence
    ): array {
        return [
            'imageName' => $scan->filename,
            'originalSize' => $this->getImageDimensions($scan->original_path),
            'analysisDate' => $scan->created_at->format('d/m/Y, H:i:s'),
            'processingTime' => number_format($totalProcessingTime / 1000, 3) . 's',
            'finalDecision' => $scan->is_defect ? 'DEFECT' : 'GOOD',
            'anomalyScore' => round($scan->anomaly_score, 4),
            'confidenceLevel' => $confidenceLevel,
            'status' => 'Completed',
            'processingSpeed' => $processingSpeed,
            'aiConfidence' => $aiConfidence,
            'analysisQuality' => $analysisQuality,
        ];
    }

    /**
     * Build visuals section data
     */
    private function buildVisuals(string $status, string $originalImageUrl, string $analyzedImageUrl): array
    {
        return [
            'status' => $status,
            'originalImageUrl' => $originalImageUrl,
            'analyzedImageUrl' => $status === 'defect' ? $analyzedImageUrl : '',
        ];
    }

    /**
     * Build performance section data
     */
    private function buildPerformance(Scan $scan, float $aiInferenceTime): array
    {
        return [
            'timeBreakdown' => [
                'preprocessing' => round(($scan->preprocessing_time_ms ?? 0) / 1000, 3),
                'aiInference' => round($aiInferenceTime / 1000, 3),
                'postprocessing' => round(($scan->postprocessing_time_ms ?? 0) / 1000, 3),
            ],
            'modelPerformance' => $this->getModelPerformanceMetrics($scan)
        ];
    }

    /**
     * Build technical section data
     */
    private function buildTechnical(
        Scan $scan,
        float $totalProcessingTime,
        float $aiInferenceTime,
        int $aiConfidence,
        array $defects
    ): array {
        return [
            'parameters' => [
                'Total Processing Time' => number_format($totalProcessingTime / 1000, 3) . 's',
                'Image Preprocessing' => number_format(($scan->preprocessing_time_ms ?? 0) / 1000, 3) . 's',
                'AI Inference' => number_format($aiInferenceTime / 1000, 3) . 's',
                'Result Processing' => number_format(($scan->postprocessing_time_ms ?? 0) / 1000, 3) . 's',
                'Model Used' => config('app.ai_model_name', 'HRNet + Anomalib'),
            ],
            'metrics' => [
                'Anomaly Detection' => $aiConfidence,
                'Classification Accuracy' => $this->getClassificationAccuracy($scan),
                'Overall Confidence' => $this->getOverallConfidence($scan),
            ],
            'rawData' => [
                'analysis_date' => $scan->created_at->format('Y-m-d H:i:s'),
                'anomaly_score' => $scan->anomaly_score,
                'final_decision' => $scan->is_defect ? 'DEFECT' : 'GOOD',
                'id' => $scan->id,
                'image_name' => $scan->filename,
                'defects' => collect($defects)->pluck('name')->toArray(),
            ],
        ];
    }

    /**
     * Calculate model performance metrics based on scan data
     */
    private function getModelPerformanceMetrics(Scan $scan): array
    {
        $confidenceScore = $scan->anomaly_score;
        $hasDefects = $scan->is_defect;

        // Base metrics - you can adjust these algorithms based on your requirements
        $accuracy = $hasDefects ?
            min(95, 80 + ($confidenceScore * 20)) :
            min(99, 85 + ($confidenceScore * 15));

        $precision = $hasDefects ?
            min(95, 70 + ($confidenceScore * 25)) :
            min(98, 85 + ($confidenceScore * 13));

        $reliability = min(98, 80 + ($confidenceScore * 18));

        // Speed based on processing time
        $totalTime = $this->calculateTotalProcessingTime($scan);
        $speed = $totalTime > 0 ? min(100, max(10, 100 - ($totalTime / 10))) : 90;

        return [
            'accuracy' => round($accuracy),
            'precision' => round($precision),
            'reliability' => round($reliability),
            'speed' => round($speed),
        ];
    }

    /**
     * Calculate classification accuracy
     */
    private function getClassificationAccuracy(Scan $scan): int
    {
        // Calculate average confidence from defects if any
        if ($scan->scanDefects->count() > 0) {
            $avgConfidence = $scan->scanDefects->avg('confidence_score');
            return round($avgConfidence * 100);
        }

        // For good scans, base it on anomaly score
        return round((1 - $scan->anomaly_score) * 100);
    }

    /**
     * Calculate overall confidence
     */
    private function getOverallConfidence(Scan $scan): int
    {
        $anomalyConfidence = $scan->anomaly_score * 100;
        $classificationAccuracy = $this->getClassificationAccuracy($scan);

        // Weighted average
        return round(($anomalyConfidence * 0.6) + ($classificationAccuracy * 0.4));
    }
}
