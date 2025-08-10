<?php

namespace App\Http\Controllers;

use App\Models\Scan;
use App\Services\ReportService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Barryvdh\Snappy\Facades\SnappyPdf;

class ReportController extends Controller
{
    public function __construct(protected ReportService $reportService) {}

    /**
     * Generate single scan report
     */
    public function generateSingleReport(Request $request, Scan $scan)
    {
        try {
            // Check if user can access this scan
            if ($scan->user_id !== auth()->id() && auth()->user()->role !== 'admin') {
                abort(403, 'Unauthorized to generate report for this scan.');
            }

            // Get report data
            $reportData = $this->reportService->prepareSingleScanReport($scan);

            // DEBUG: Log what's being passed to the blade template
            Log::info('Report data keys:', array_keys($reportData));
            Log::info('Defects data:', [
                'defects_exists' => isset($reportData['defects']),
                'defects_type' => isset($reportData['defects']) ? gettype($reportData['defects']) : 'not_set',
                'defects_count' => isset($reportData['defects']) ? (is_countable($reportData['defects']) ? count($reportData['defects']) : 'not_countable') : 'not_set',
                'defects_class' => isset($reportData['defects']) && is_object($reportData['defects']) ? get_class($reportData['defects']) : 'not_object',
            ]);

            // Generate PDF with Snappy
            $pdf = SnappyPdf::loadView('reports.single-scan', $reportData)
                ->setOptions([
                    'page-size' => 'A4',
                    'orientation' => 'Portrait',
                    'margin-top' => '20mm',
                    'margin-right' => '15mm',
                    'margin-bottom' => '20mm',
                    'margin-left' => '15mm',
                    'encoding' => 'UTF-8',
                    'enable-local-file-access' => true,
                ]);

            $filename = "scan-report-{$scan->id}-" . now()->format('Y-m-d-H-i-s') . ".pdf";

            return $pdf->download($filename);
        } catch (\Exception $e) {
            Log::error('Single scan report generation failed', [
                'scan_id' => $scan->id,
                'user_id' => auth()->id(),
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString()
            ]);

            return back()->with('error', 'Failed to generate report. Please try again.');
        }
    }

    /**
     * Preview single scan report (for testing)
     */
    public function previewSingleReport(Scan $scan)
    {
        try {
            // Check if user can access this scan
            if ($scan->user_id !== auth()->id() && auth()->user()->role !== 'admin') {
                abort(403, 'Unauthorized to preview report for this scan.');
            }

            // Get report data
            $reportData = $this->reportService->prepareSingleScanReport($scan);

            return view('reports.single-scan', $reportData);
        } catch (\Exception $e) {
            Log::error('Single scan report preview failed', [
                'scan_id' => $scan->id,
                'user_id' => auth()->id(),
                'error' => $e->getMessage()
            ]);

            return back()->with('error', 'Failed to preview report. Please try again.');
        }
    }
}
