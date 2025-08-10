<?php

namespace App\Http\Controllers;

use App\Services\DashboardService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use Inertia\Inertia;
use Inertia\Response;

class DashboardController extends Controller
{
    protected $dashboardService;

    public function __construct(DashboardService $dashboardService)
    {
        $this->dashboardService = $dashboardService;
    }

    /**
     * Display the dashboard page
     */
    public function index(Request $request): Response
    {
        $filters = [
            'defect_filter' => $request->get('defect_filter'),
            'date_range' => $request->get('date_range'),
        ];

        $dashboardData = $this->dashboardService->getDashboardData($filters);

        return Inertia::render('Dashboard/Index', $dashboardData);
    }

    /**
     * API endpoint for polling dashboard data
     */
    public function indexApi(Request $request): JsonResponse
    {
        $filters = [
            'defect_filter' => $request->get('defect_filter'),
            'date_range' => $request->get('date_range'),
        ];

        $dashboardData = $this->dashboardService->getDashboardData($filters);

        return response()->json([
            'status' => 'success',
            'data' => $dashboardData,
            'timestamp' => now()->toISOString(),
        ]);
    }

    /**
     * API endpoint for getting only card data (for frequent polling)
     */
    public function cardDataApi(): JsonResponse
    {
        $cardData = $this->dashboardService->getCardDataOnly();

        return response()->json([
            'status' => 'success',
            'data' => $cardData,
            'timestamp' => now()->toISOString(),
        ]);
    }

    /**
     * API endpoint for getting only overview data
     */
    public function overviewDataApi(): JsonResponse
    {
        $overviewData = $this->dashboardService->getOverviewDataOnly();

        return response()->json([
            'status' => 'success',
            'data' => $overviewData,
            'timestamp' => now()->toISOString(),
        ]);
    }

    /**
     * API endpoint for getting defect trend data
     */
    public function defectTrendApi(Request $request): JsonResponse
    {
        $filterMonth = $request->get('month');
        $trendData = $this->dashboardService->getDefectTrendOnly($filterMonth);

        return response()->json([
            'status' => 'success',
            'data' => $trendData,
            'timestamp' => now()->toISOString(),
        ]);
    }

    /**
     * API endpoint for getting recent analyses
     */
    public function recentAnalysesApi(Request $request): JsonResponse
    {
        $limit = $request->get('limit', 10);
        $recentData = $this->dashboardService->getRecentAnalysesOnly($limit);

        return response()->json([
            'status' => 'success',
            'data' => $recentData,
            'timestamp' => now()->toISOString(),
        ]);
    }
}
