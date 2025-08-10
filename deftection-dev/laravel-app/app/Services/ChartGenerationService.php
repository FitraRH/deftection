<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class ChartGenerationService
{
    /**
     * Generate chart as base64 image using QuickChart API
     */
    public function generateChartBase64(array $chartConfig, int $width = 400, int $height = 200): ?string
    {
        try {
            $response = Http::timeout(30)->post('https://quickchart.io/chart', [
                'chart' => $chartConfig,
                'width' => $width,
                'height' => $height,
                'format' => 'png',
                'backgroundColor' => 'white',
            ]);

            if ($response->successful()) {
                return 'data:image/png;base64,' . base64_encode($response->body());
            }

            Log::error('Chart generation failed', [
                'status' => $response->status(),
                'body' => $response->body()
            ]);

            return null;
        } catch (\Exception $e) {
            Log::error('Chart generation exception', [
                'error' => $e->getMessage()
            ]);
            return null;
        }
    }

    /**
     * Generate defect distribution doughnut chart
     */
    public function generateDefectDistributionChart(array $defectData): ?string
    {
        if (empty($defectData)) {
            return null;
        }

        $labels = array_column($defectData, 'label');
        $data = array_column($defectData, 'count');
        $colors = ['#dc3545', '#ffc107', '#28a745', '#007bff', '#6f42c1', '#fd7e14', '#20c997', '#e83e8c'];

        $chartConfig = [
            'type' => 'doughnut',
            'data' => [
                'labels' => $labels,
                'datasets' => [[
                    'data' => $data,
                    'backgroundColor' => array_slice($colors, 0, count($data)),
                    'borderWidth' => 2,
                    'borderColor' => '#fff'
                ]]
            ],
            'options' => [
                'responsive' => false,
                'plugins' => [
                    'legend' => [
                        'position' => 'bottom',
                        'labels' => [
                            'fontSize' => 12,
                            'padding' => 15
                        ]
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 350, 250);
    }

    /**
     * Generate severity distribution pie chart
     */
    public function generateSeverityDistributionChart(array $severityData): ?string
    {
        if (empty($severityData)) {
            return null;
        }

        $labels = array_column($severityData, 'severity');
        $data = array_column($severityData, 'count');
        $colors = ['#dc3545', '#ffc107', '#28a745']; // Red, Yellow, Green

        $chartConfig = [
            'type' => 'pie',
            'data' => [
                'labels' => $labels,
                'datasets' => [[
                    'data' => $data,
                    'backgroundColor' => array_slice($colors, 0, count($data)),
                    'borderWidth' => 2,
                    'borderColor' => '#fff'
                ]]
            ],
            'options' => [
                'responsive' => false,
                'plugins' => [
                    'legend' => [
                        'position' => 'bottom',
                        'labels' => [
                            'fontSize' => 12,
                            'padding' => 15
                        ]
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 350, 250);
    }

    /**
     * Generate processing time bar chart
     */
    public function generateProcessingTimeChart(array $processingData): ?string
    {
        if (empty($processingData)) {
            return null;
        }

        $labels = array_column($processingData, 'stage');
        $data = array_column($processingData, 'time_ms');
        $colors = ['#007bff', '#28a745', '#ffc107', '#6f42c1'];

        $chartConfig = [
            'type' => 'bar',
            'data' => [
                'labels' => $labels,
                'datasets' => [[
                    'label' => 'Time (ms)',
                    'data' => $data,
                    'backgroundColor' => array_slice($colors, 0, count($data)),
                    'borderWidth' => 1,
                    'borderColor' => '#fff'
                ]]
            ],
            'options' => [
                'responsive' => false,
                'scales' => [
                    'y' => [
                        'beginAtZero' => true,
                        'title' => [
                            'display' => true,
                            'text' => 'Time (milliseconds)'
                        ]
                    ]
                ],
                'plugins' => [
                    'legend' => [
                        'display' => false
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 600, 300);
    }

    /**
     * Generate fallback placeholder image
     */
    public function generatePlaceholderChart(string $title, string $subtitle = ''): string
    {
        // Create a simple SVG placeholder and convert to base64
        $svg = '<svg width="350" height="250" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2" rx="8"/>
            <text x="50%" y="40%" text-anchor="middle" font-family="Arial" font-size="16" fill="#495057" font-weight="bold">' . htmlspecialchars($title) . '</text>
            <text x="50%" y="60%" text-anchor="middle" font-family="Arial" font-size="12" fill="#6c757d">' . htmlspecialchars($subtitle) . '</text>
        </svg>';

        return 'data:image/svg+xml;base64,' . base64_encode($svg);
    }
}
