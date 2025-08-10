<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ $reportTitle }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'DejaVu Sans', Arial, sans-serif;
            font-size: 12px;
            line-height: 1.6;
            color: #333;
            margin: 20mm 15mm 20mm 15mm;
        }

        @page {
            margin: 20mm 15mm 20mm 15mm;
            size: A4;
        }

        .page-break {
            page-break-after: always;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 25px 0;
            border-bottom: 3px solid #007bff;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px 8px 0 0;
        }

        .header h1 {
            font-size: 28px;
            color: #007bff;
            margin-bottom: 15px;
            font-weight: 700;
        }

        .header .subtitle {
            font-size: 16px;
            color: #6c757d;
            font-weight: 500;
        }

        .meta-info {
            display: table;
            width: 100%;
            margin-bottom: 30px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
        }

        .meta-row {
            display: table-row;
        }

        .meta-label,
        .meta-value {
            display: table-cell;
            padding: 12px 15px;
            border-bottom: 1px solid #f1f3f4;
        }

        .meta-label {
            font-weight: 600;
            width: 30%;
            color: #495057;
            background-color: #f8f9fa;
        }

        .meta-value {
            width: 70%;
            background-color: #ffffff;
        }

        .meta-row:last-child .meta-label,
        .meta-row:last-child .meta-value {
            border-bottom: none;
        }

        .section {
            margin-bottom: 35px;
        }

        .section-title {
            font-size: 20px;
            font-weight: 700;
            color: #007bff;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid #007bff;
            position: relative;
        }

        .section-title::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 60px;
            height: 2px;
            background-color: #28a745;
        }

        .status-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 15px;
            font-weight: 500;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            min-width: 80px;
            text-align: center;
        }

        .status-good {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 2px solid #28a745;
        }

        .status-defective {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border: 2px solid #dc3545;
        }

        .summary-grid {
            display: table;
            width: 100%;
            margin-bottom: 25px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .summary-row {
            display: table-row;
        }

        .summary-cell {
            display: table-cell;
            padding: 15px 12px;
            border-right: 1px solid #dee2e6;
            text-align: center;
            vertical-align: middle;
        }

        .summary-cell:last-child {
            border-right: none;
        }

        .summary-cell.header {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            font-weight: 700;
            font-size: 11px;
            color: #495057;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .summary-cell .value {
            font-size: 20px;
            font-weight: 800;
            color: #007bff;
            margin-bottom: 5px;
        }

        .summary-cell .label {
            font-size: 10px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .chart-image {
            max-width: 100%;
            max-height: 250px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .defect-list {
            margin-top: 20px;
        }

        .defect-item {
            margin-bottom: 20px;
            padding: 20px;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .defect-header {
            display: table;
            width: 100%;
            margin-bottom: 15px;
        }

        .defect-info,
        .defect-metrics {
            display: table-cell;
            vertical-align: top;
        }

        .defect-info {
            width: 60%;
        }

        .defect-metrics {
            width: 40%;
            padding-left: 25px;
        }

        .defect-name {
            font-size: 16px;
            font-weight: 700;
            color: #dc3545;
            margin-bottom: 8px;
            text-transform: capitalize;
        }

        .defect-explanation {
            font-size: 12px;
            color: #6c757d;
            line-height: 1.6;
        }

        .metric {
            margin-bottom: 8px;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metric-label {
            font-weight: 600;
            color: #495057;
            min-width: 80px;
        }

        .metric-value {
            color: #007bff;
            font-weight: 600;
        }

        .severity-high {
            color: #dc3545;
            font-weight: 700;
            text-transform: uppercase;
        }

        .severity-medium {
            color: #ffc107;
            font-weight: 700;
            text-transform: uppercase;
        }

        .severity-low {
            color: #28a745;
            font-weight: 700;
            text-transform: uppercase;
        }

        .image-container {
            text-align: center;
            margin: 25px 0;
        }

        .scan-image {
            max-width: 100%;
            max-height: 400px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .image-caption {
            font-size: 12px;
            color: #6c757d;
            margin-top: 12px;
            font-weight: 500;
        }

        .image-placeholder {
            width: 100%;
            height: 250px;
            border: 2px dashed #dee2e6;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            color: #6c757d;
            border-radius: 8px;
            font-size: 14px;
            text-align: center;
        }

        .two-column {
            display: table;
            width: 100%;
        }

        .column {
            display: table-cell;
            width: 50%;
            vertical-align: top;
            padding-right: 20px;
        }

        .column:last-child {
            padding-right: 0;
            padding-left: 20px;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
        }

        .data-table th,
        .data-table td {
            border-bottom: 1px solid #dee2e6;
            padding: 12px 15px;
            text-align: left;
            font-size: 11px;
        }

        .data-table th {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            font-weight: 700;
            color: #495057;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .data-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        .data-table tr:last-child td {
            border-bottom: none;
        }

        .threat-info {
            padding: 25px;
            border-radius: 10px;
            margin-top: 15px;
            border: 2px solid;
        }

        .threat-clean {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-color: #28a745;
            color: #155724;
        }

        .threat-suspicious {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border-color: #ffc107;
            color: #856404;
        }

        .threat-malicious {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-color: #dc3545;
            color: #721c24;
        }

        .threat-status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            min-width: 70px;
            text-align: center;
        }

        .threat-status.clean {
            background-color: #28a745;
            color: white;
        }

        .threat-status.suspicious {
            background-color: #ffc107;
            color: #856404;
        }

        .threat-status.malicious {
            background-color: #dc3545;
            color: white;
        }

        .risk-level {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 10px;
            font-weight: 500;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .risk-none {
            background-color: #e9ecef;
            color: #495057;
        }

        .risk-low {
            background-color: #d1ecf1;
            color: #0c5460;
        }

        .risk-medium {
            background-color: #fff3cd;
            color: #856404;
        }

        .risk-high {
            background-color: #f8d7da;
            color: #721c24;
        }

        .security-flags {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }

        .security-flag {
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 4px solid;
            font-weight: 600;
            font-size: 11px;
            text-transform: capitalize;
        }

        .flag-clean {
            background-color: rgba(40, 167, 69, 0.1);
            border-color: #28a745;
            color: #155724;
        }

        .flag-suspicious {
            background-color: rgba(255, 193, 7, 0.1);
            border-color: #ffc107;
            color: #856404;
        }

        .flag-malicious {
            background-color: rgba(220, 53, 69, 0.1);
            border-color: #dc3545;
            color: #721c24;
        }

        .section-header {
            font-size: 16px;
            font-weight: 700;
            margin: 20px 0 10px 0;
            color: #495057;
        }

        .detail-analysis {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }

        .attack-types {
            margin-top: 15px;
        }

        .attack-types ul {
            margin: 10px 0;
            padding-left: 25px;
        }

        .attack-types li {
            margin-bottom: 8px;
            line-height: 1.5;
        }

        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 10px;
            color: #6c757d;
            padding: 15px 0;
            border-top: 1px solid #dee2e6;
            background-color: #f8f9fa;
        }

        /* No defects message */
        .no-defects {
            text-align: center;
            padding: 50px 40px;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-radius: 10px;
            color: #155724;
            border: 2px solid #28a745;
        }

        .no-defects h3 {
            font-size: 24px;
            margin-bottom: 15px;
            font-weight: 700;
        }

        .no-defects p {
            font-size: 16px;
            font-weight: 500;
        }

        /* SVG Icon Styles */
        .icon {
            display: inline-block;
            width: 16px;
            height: 16px;
            vertical-align: middle;
            margin-right: 6px;
        }

        .icon-large {
            width: 24px;
            height: 24px;
        }

        .icon-status {
            width: 18px;
            height: 18px;
            margin-right: 8px;
        }

        /* Status Icons using CSS background images with SVG */
        .status-icon-clean {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%2328a745'%3e%3cpath fill-rule='evenodd' d='M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        .status-icon-suspicious {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%23ffc107'%3e%3cpath fill-rule='evenodd' d='M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        .status-icon-malicious {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%23dc3545'%3e%3cpath fill-rule='evenodd' d='M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        .icon-camera {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%236c757d'%3e%3cpath fill-rule='evenodd' d='M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        .icon-search {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%236c757d'%3e%3cpath fill-rule='evenodd' d='M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        .icon-chart {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%23007bff'%3e%3cpath d='M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z'/%3e%3c/svg%3e");
        }

        .icon-warning {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%23ffc107'%3e%3cpath fill-rule='evenodd' d='M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        .icon-clock {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%2328a745'%3e%3cpath fill-rule='evenodd' d='M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        .icon-flag {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor'%3e%3cpath fill-rule='evenodd' d='M3 6a3 3 0 013-3h10a1 1 0 01.8 1.6L14.25 8l2.55 3.4A1 1 0 0116 13H6a1 1 0 00-1 1v3a1 1 0 11-2 0V6z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        .icon-shield {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor'%3e%3cpath fill-rule='evenodd' d='M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        .icon-lightning {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor'%3e%3cpath fill-rule='evenodd' d='M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z' clip-rule='evenodd'/%3e%3c/svg%3e");
        }

        /* Icon base styles */
        .icon,
        .icon-large,
        .icon-status,
        [class^="icon-"],
        [class*=" icon-"] {
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
        }
    </style>
</head>

<body>
    <!-- Header -->
    <div class="header">
        <h1>{{ $reportTitle }}</h1>
        <div class="subtitle">Generated on {{ $generatedAt->format('F j, Y \a\t g:i A') }}</div>
    </div>

    <!-- Scan Information -->
    <div class="section">
        <h2 class="section-title">Scan Information</h2>
        <div class="meta-info">
            <div class="meta-row">
                <div class="meta-label">Scan ID:</div>
                <div class="meta-value">#{{ $scan->id }}</div>
            </div>
            <div class="meta-row">
                <div class="meta-label">Filename:</div>
                <div class="meta-value">{{ $scan->filename }}</div>
            </div>
            <div class="meta-row">
                <div class="meta-label">Scanned By:</div>
                <div class="meta-value">
                    @if ($scan->user)
                        {{ $scan->user->name }} ({{ $scan->user->email }})
                    @else
                        <em>Unknown User (User Deleted)</em>
                    @endif
                </div>
            </div>
            <div class="meta-row">
                <div class="meta-label">Scan Date:</div>
                <div class="meta-value">{{ $scan->created_at->format('F j, Y \a\t g:i A') }}</div>
            </div>
            <div class="meta-row">
                <div class="meta-label">Overall Status:</div>
                <div class="meta-value">
                    <span
                        class="status-badge {{ $summary['overall_status'] === 'GOOD' ? 'status-good' : 'status-defective' }}">
                        {{ $summary['overall_status'] }}
                    </span>
                </div>
            </div>
        </div>
    </div>

    <!-- Summary Dashboard -->
    <div class="section">
        <h2 class="section-title">Analysis Summary</h2>
        <div class="summary-grid">
            <div class="summary-row">
                <div class="summary-cell header">Anomaly Score</div>
                <div class="summary-cell header">Confidence Level</div>
                <div class="summary-cell header">Defects Found</div>
                <div class="summary-cell header">Processing Time</div>
            </div>
            <div class="summary-row">
                <div class="summary-cell">
                    <div class="value">{{ number_format($summary['anomaly_score'], 3) }}</div>
                    <div class="label">Score</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ $summary['confidence_level'] ?? 'N/A' }}</div>
                    <div class="label">Level</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ $summary['defects_found'] }}</div>
                    <div class="label">Total</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ number_format($summary['processing_performance']['total_time'], 1) }}ms
                    </div>
                    <div class="label">Total</div>
                </div>
            </div>
        </div>

        <div class="summary-grid">
            <div class="summary-row">
                <div class="summary-cell header">Unique Defect Types</div>
                <div class="summary-cell header">Total Affected Area</div>
                <div class="summary-cell header">Avg Defect Confidence</div>
                <div class="summary-cell header">Threshold Used</div>
            </div>
            <div class="summary-row">
                <div class="summary-cell">
                    <div class="value">{{ $summary['unique_defect_types'] }}</div>
                    <div class="label">Types</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ $summary['total_affected_area'] }}%</div>
                    <div class="label">Percentage</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ number_format($summary['avg_defect_confidence'], 3) }}</div>
                    <div class="label">Score</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ number_format($summary['threshold_used'], 3) }}</div>
                    <div class="label">Threshold</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Images Section -->
    <div class="section">
        <h2 class="section-title">Scan Images</h2>
        <div class="two-column">
            <div class="column">
                <div class="image-container">
                    @if ($imagePaths['original'])
                        <img src="{{ $imagePaths['original'] }}" alt="Original Image" class="scan-image">
                        <div class="image-caption">Original Image</div>
                    @else
                        <div class="image-placeholder">
                            <div>
                                <div class="icon-camera icon-large"
                                    style="width: 48px; height: 48px; margin-bottom: 10px;"></div>
                                <div style="font-weight: bold;">Original Image Not Available</div>
                                <div style="font-size: 11px;">Image file could not be found or has been deleted</div>
                            </div>
                        </div>
                        <div class="image-caption">Original Image (Not Available)</div>
                    @endif
                </div>
            </div>

            <div class="column">
                <div class="image-container">
                    @if ($imagePaths['annotated'])
                        <img src="{{ $imagePaths['annotated'] }}" alt="Annotated Image" class="scan-image">
                        <div class="image-caption">Annotated Image with Detected Defects</div>
                    @else
                        <div class="image-placeholder">
                            <div>
                                <div class="icon-search icon-large"
                                    style="width: 48px; height: 48px; margin-bottom: 10px;"></div>
                                <div style="font-weight: bold;">Annotated Image Not Available</div>
                                <div style="font-size: 11px;">Processed image file could not be found or has been
                                    deleted</div>
                            </div>
                        </div>
                        <div class="image-caption">Annotated Image (Not Available)</div>
                    @endif
                </div>
            </div>
        </div>
    </div>

    <div class="page-break"></div>

    <!-- Defects Analysis -->
    @if ($defects->count() > 0)
        <div class="section">
            <h2 class="section-title">Detected Defects Analysis</h2>

            <!-- Charts Section -->
            <div class="two-column">
                <div class="column">
                    <h3 style="font-size: 16px; margin-bottom: 15px; font-weight: 600;">Defect Distribution</h3>
                    @if (isset($chartImages['defect_distribution']) && $chartImages['defect_distribution'])
                        <img src="{{ $chartImages['defect_distribution'] }}" alt="Defect Distribution Chart"
                            class="chart-image">
                    @else
                        <div class="image-placeholder">
                            <div>
                                <div class="icon-chart icon-large"
                                    style="width: 24px; height: 24px; margin-bottom: 8px;"></div>
                                <div style="font-weight: bold;">Defect Distribution Chart</div>
                                <div style="font-size: 11px;">{{ count($chartsData['defect_distribution']) }}
                                    different
                                    types detected</div>
                            </div>
                        </div>
                    @endif
                </div>
                <div class="column">
                    <h3 style="font-size: 16px; margin-bottom: 15px; font-weight: 600;">Severity Levels</h3>
                    @if (isset($chartImages['severity_distribution']) && $chartImages['severity_distribution'])
                        <img src="{{ $chartImages['severity_distribution'] }}" alt="Severity Distribution Chart"
                            class="chart-image">
                    @else
                        <div class="image-placeholder">
                            <div>
                                <div class="icon-warning icon-large"
                                    style="width: 24px; height: 24px; margin-bottom: 8px;"></div>
                                <div style="font-weight: bold;">Severity Distribution Chart</div>
                                <div style="font-size: 11px;">{{ count($chartsData['severity_distribution']) }}
                                    severity
                                    levels</div>
                            </div>
                        </div>
                    @endif
                </div>
            </div>

            <!-- Defect Details -->
            <div class="defect-list">
                @foreach ($defects as $defect)
                    <div class="defect-item">
                        <div class="defect-header">
                            <div class="defect-info">
                                <div class="defect-name">{{ $defect['label'] }}</div>
                                <div class="defect-explanation">
                                    {{ $defect['explanation'] }}
                                </div>
                            </div>
                            <div class="defect-metrics">
                                <div class="metric">
                                    <span class="metric-label">Confidence:</span>
                                    <span
                                        class="metric-value">{{ number_format($defect['confidence_score'], 3) }}</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Severity:</span>
                                    <span
                                        class="metric-value severity-{{ strtolower($defect['severity_level'] ?? 'unknown') }}">
                                        {{ $defect['severity_level'] ?? 'Unknown' }}
                                    </span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Area Affected:</span>
                                    <span
                                        class="metric-value">{{ number_format($defect['area_percentage'], 2) }}%</span>
                                </div>
                                @if ($defect['box_location'])
                                    <div class="metric">
                                        <span class="metric-label">Location:</span>
                                        <span class="metric-value">
                                            ({{ $defect['box_location']['x'] ?? 0 }},
                                            {{ $defect['box_location']['y'] ?? 0 }})
                                        </span>
                                    </div>
                                @endif
                            </div>
                        </div>
                    </div>
                @endforeach
            </div>
        </div>
    @else
        <div class="section">
            <h2 class="section-title">Defects Analysis</h2>
            <div class="no-defects">
                <h3>No Defects Detected</h3>
                <p>This scan passed all quality checks successfully.</p>
            </div>
        </div>
    @endif

    <!-- Processing Performance -->
    <div class="section">
        <h2 class="section-title">Processing Performance</h2>
        @if (isset($chartImages['processing_time']) && $chartImages['processing_time'])
            <img src="{{ $chartImages['processing_time'] }}" alt="Processing Time Chart" class="chart-image"
                style="width: 100%;">
        @else
            <div class="image-placeholder" style="width: 100%; height: 200px;">
                <div>
                    <div class="icon-clock icon-large" style="width: 24px; height: 24px; margin-bottom: 8px;"></div>
                    <div style="font-weight: bold;">Processing Time Breakdown Chart</div>
                    <div style="font-size: 11px;">Total:
                        {{ number_format($chartsData['total_processing_time'], 1) }}ms
                    </div>
                </div>
            </div>
        @endif

        <table class="data-table">
            <thead>
                <tr>
                    <th>Processing Stage</th>
                    <th>Time (ms)</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                @foreach ($chartsData['processing_time_breakdown'] as $stage)
                    <tr>
                        <td>{{ $stage['stage'] }}</td>
                        <td>{{ number_format($stage['time_ms'], 1) }}</td>
                        <td>{{ $stage['percentage'] }}%</td>
                    </tr>
                @endforeach
            </tbody>
        </table>
    </div>

    <!-- Threat Analysis -->
    @if ($scan->scanThreat)
        <div class="section">
            <h2 class="section-title">Security Threat Analysis</h2>
            <div class="threat-info threat-{{ strtolower($scan->scanThreat->status) }}">
                <div class="meta-info">
                    <div class="meta-row">
                        <div class="meta-label">Status:</div>
                        <div class="meta-value">
                            <span class="icon-status status-icon-{{ strtolower($scan->scanThreat->status) }}"></span>
                            <span class="threat-status {{ strtolower($scan->scanThreat->status) }}">
                                {{ $scan->scanThreat->status }}
                            </span>
                        </div>
                    </div>
                    <div class="meta-row">
                        <div class="meta-label">Risk Level:</div>
                        <div class="meta-value">
                            <span class="risk-level risk-{{ strtolower($scan->scanThreat->risk_level) }}">
                                {{ $scan->scanThreat->risk_level }}
                            </span>
                        </div>
                    </div>
                    <div class="meta-row">
                        <div class="meta-label">File Hash:</div>
                        <div class="meta-value" style="font-family: monospace; font-size: 10px;">
                            {{ $scan->scanThreat->hash }}</div>
                    </div>
                    @if ($scan->scanThreat->processing_time_ms)
                        <div class="meta-row">
                            <div class="meta-label">Scan Time:</div>
                            <div class="meta-value">{{ number_format($scan->scanThreat->processing_time_ms, 1) }}ms
                            </div>
                        </div>
                    @endif
                </div>

                @if ($scan->scanThreat->flags && count($scan->scanThreat->flags) > 0)
                    <div style="margin-top: 25px;">
                        <h4 class="section-header">
                            <span class="icon-flag icon" style="margin-right: 8px;"></span>
                            Security Flags Detected:
                        </h4>
                        <div class="security-flags">
                            @foreach ($scan->scanThreat->flags as $flag)
                                <div class="security-flag flag-{{ strtolower($scan->scanThreat->status) }}">
                                    {{ ucwords(str_replace('_', ' ', $flag)) }}
                                </div>
                            @endforeach
                        </div>
                    </div>
                @endif

                @if ($scan->scanThreat->details && count($scan->scanThreat->details) > 0)
                    <div style="margin-top: 25px;">
                        <h4 class="section-header">
                            <span class="icon-shield icon" style="margin-right: 8px;"></span>
                            Detailed Analysis:
                        </h4>
                        <div class="detail-analysis">
                            @foreach ($scan->scanThreat->details as $key => $value)
                                <div style="margin-bottom: 12px;">
                                    <strong>{{ ucwords(str_replace('_', ' ', $key)) }}:</strong>
                                    @if (is_array($value))
                                        {{ implode(', ', $value) }}
                                    @else
                                        @if (str_contains(strtolower($key), 'url') || str_contains(strtolower($key), 'content'))
                                            <code
                                                style="background: #f8f9fa; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 10px;">
                                                {{ $value }}
                                            </code>
                                        @else
                                            {{ $value }}
                                        @endif
                                    @endif
                                </div>
                            @endforeach
                        </div>
                    </div>
                @endif

                @if ($scan->scanThreat->possible_attack && count($scan->scanThreat->possible_attack) > 0)
                    <div class="attack-types">
                        <h4 class="section-header">
                            <span class="icon-lightning icon" style="margin-right: 8px;"></span>
                            Possible Attack Types:
                        </h4>
                        <ul>
                            @foreach ($scan->scanThreat->possible_attack as $attack)
                                <li>{{ $attack }}</li>
                            @endforeach
                        </ul>
                    </div>
                @endif
            </div>
        </div>
    @endif

    <!-- Technical Details -->
    <div class="section">
        <h2 class="section-title">Technical Details</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Anomaly Threshold</td>
                    <td>{{ number_format($scan->anomaly_threshold, 5) }}</td>
                    <td>Threshold used for anomaly detection</td>
                </tr>
                <tr>
                    <td>Anomaly Score</td>
                    <td>{{ number_format($scan->anomaly_score, 5) }}</td>
                    <td>Calculated anomaly score for this scan</td>
                </tr>
                <tr>
                    <td>Confidence Level</td>
                    <td>{{ $scan->anomaly_confidence_level ?? 'N/A' }}</td>
                    <td>Model confidence in the prediction</td>
                </tr>
                <tr>
                    <td>Classification Result</td>
                    <td>{{ $scan->is_defect ? 'DEFECTIVE' : 'GOOD' }}</td>
                    <td>Final classification result</td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- Data Tables -->
    @if (count($chartsData['defect_distribution']) > 0)
        <div class="section">
            <h2 class="section-title">Defect Distribution Data</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Defect Type</th>
                        <th>Count</th>
                        <th>Avg Confidence</th>
                        <th>Total Area (%)</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach ($chartsData['defect_distribution'] as $defect)
                        <tr>
                            <td>{{ $defect['label'] }}</td>
                            <td>{{ $defect['count'] }}</td>
                            <td>{{ number_format($defect['avg_confidence'], 3) }}</td>
                            <td>{{ $defect['total_area'] }}%</td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    @endif

    <!-- Footer -->
    <div class="footer">
        <div>Report generated on {{ $generatedAt->format('Y-m-d H:i:s') }} | Scan ID: #{{ $scan->id }}</div>
    </div>
</body>

</html>
