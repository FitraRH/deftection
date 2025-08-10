<script setup>
import StatsCard from "./Components/StatsCard.vue";
import PerformanceTrendChart from "./Components/PerformanceTrendChart.vue";
import DefectTypeChart from "./Components/DefectTypeChart.vue";
import DailyAnalysisTrend from "./Components/DailyAnalysisTrend.vue";
import RecentAnalysisTable from "./Components/RecentAnalysisTable.vue";
import AnalysisOverview from "./Components/AnalysisOverview.vue";
</script>

<template>
    <!-- Status Grid -->
    <div class="gap-6 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4">
        <!--
            Total Defects
            Expected props:
                value = totalDefectsDetected
            Used as:
                <StatsCard :value="totalDefectsDetected" ... />
        -->
        <StatsCard
            title="Total Defects"
            value="5"
            description="Total number of defects detected."
            badgeColor="primary"
            changeRate="+10%"
        >
            <template #icon>
                <font-awesome-icon icon="fa-solid fa-bug" />
            </template>
        </StatsCard>

        <!--
            Images Processed
            Expected props:
                value = totalImagesProcessed
            Used as:
                <StatsCard :value="totalImagesProcessed" ... />
        -->
        <StatsCard
            title="Images Processed"
            value="10"
            description="Number of images processed."
            badgeColor="success"
            changeRate="-5%"
        >
            <template #icon>
                <font-awesome-icon icon="fa-solid fa-images" />
            </template>
        </StatsCard>

        <!--
            Accuracy
            Expected props:
                value = totalAccuracy (both Models ig)
            Used as:
                <StatsCard :value="totalAccuracy" ... />
        -->
        <StatsCard
            title="Accuracy"
            value="75%"
            description="Overall accuracy of the analysis."
            badgeColor="secondary"
            changeRate="+0.2%"
        >
            <template #icon>
                <font-awesome-icon icon="fa-solid fa-bullseye" />
            </template>
        </StatsCard>

        <!--
            Throughput (How many images are processed per second)
            Expected props:
                value = throughputTotal
            Used as:
                <StatsCard :value="throughputTotal" />
        -->
        <StatsCard
            title="Throughput"
            value="2 images/s"
            description="Average time taken per image."
            badgeColor="error"
            changeRate="+10%"
        >
            <template #icon>
                <font-awesome-icon icon="fa-solid fa-gauge" />
            </template>
        </StatsCard>
    </div>

    <!-- Daily Analysis Trend & Defect Type Distribution -->
    <div class="gap-6 grid grid-cols-1 md:grid-cols-2 py-6">
        <div
            class="flex flex-col justify-start items-start p-6 border border-gray-200 dark:border-dark-700 rounded-lg"
        >
            <h2
                class="font-medium text-gray-800 dark:text-dark-100 text-sm-plus tracking-wide"
            >
                Daily Analysis Trend (7 Days)
            </h2>

            <!--
                📊 Daily Analysis Trend Chart
                Expected backend prop: `daily_analysis`

                Structure:
                {
                    daily_analysis: {
                        labels: ["2025-07-01", "2025-07-02", ..., "2025-07-07"], // Last 7 days (or any date range)
                        total_defects: [5, 8, 4, 10, 6, 3, 7], // Total defects detected per day
                        total_processed: [50, 60, 55, 70, 65, 48, 52] // Total images analyzed per day
                    }
                }

                Notes:
                - All arrays must align by index (labels[i] → total_defects[i] & total_processed[i])
                - Supports grouped bar chart: GOOD (processed - defects) vs DEFECT
                - Used to monitor daily analysis volume and quality trends
            -->
            <DailyAnalysisTrend />
        </div>
        <div
            class="flex flex-col justify-start p-6 border border-gray-200 dark:border-dark-700 rounded-lg"
        >
            <h2
                class="font-medium text-gray-800 dark:text-dark-100 text-sm-plus tracking-wide"
            >
                Defect Type Distribution
            </h2>

            <!--
                📊 Defect Type Distribution Chart
                Expected backend prop: `defect_type`

                Structure:
                {
                    defect_type: {
                        labels: ["scratched", "opened", ...], // All detectable defect types
                        data: [10, 20, ...] // Total count per defect type, same index as labels
                    }
                }

                Notes:
                - `labels[i]` corresponds to `data[i]`
                - Used for PolarArea or Doughnut chart to visualize distribution of defect types
            -->
            <DefectTypeChart />
        </div>
    </div>

    <div class="grid grid-cols-1 py-6">
        <div
            class="flex flex-col justify-start p-6 border border-gray-200 dark:border-dark-700 rounded-lg"
        >
            <h2
                class="font-medium text-gray-800 dark:text-dark-100 text-sm-plus tracking-wide"
            >
                Performance Trend (30 Days)
            </h2>

            <!--
               Performance Trend Chart
               Expected data:
                 {
                    "performance_trend": {
                        "labels": ["2025-07-01", "2025-07-02", "2025-07-03", ...], // <-  30 tanggal terakhir(?)
                        "data": [10, 20, 15, ...] // ← total defects detected on each corresponding date
                    }
                }

                Notes:
                - `labels[i]` corresponds to `data[i]`
                - Used for Line chart to visualize performance trend
            -->
            <PerformanceTrendChart />
        </div>
    </div>

    <!-- Recent Analysis Table & Analysis Widget -->
    <div class="gap-6 grid grid-cols-1 md:grid-cols-2 py-6">
        <div
            class="flex flex-col justify-start gap-4 p-6 border border-gray-200 dark:border-dark-700 rounded-lg"
        >
            <div class="flex justify-between items-center">
                <h2
                    class="font-medium text-gray-800 dark:text-dark-100 text-sm-plus tracking-wide"
                >
                    Recent Analysis
                </h2>
                <a
                    href="/history"
                    class="pb-0.5 border-current border-b border-dotted outline-hidden font-medium text-primary-600 hover:text-primary-600/70 focus:text-primary-600/70 dark:hover:text-primary-400/70 dark:focus:text-primary-400/70 dark:text-primary-400 text-xs-plus transition-colors duration-300"
                >
                    <span>View all</span>
                </a>
            </div>

            <!--
                🧾 Recent Analysis Table
                Expected backend prop: `analyses`

                Maximum 5 data aja

                Structure:
                {
                    analyses: [
                        {
                            id: 1, // Unique identifier for detail route
                            imageName: "product_001.jpg", // Original image file name
                            result: "GOOD" | "BAD", // Classification result
                            score: 0.95 // Confidence score (0.0 - 1.0)
                        },
                        ...
                    ]
                }

                Notes:
                - `score` can be formatted as a percentage (e.g., 95%)
                - `result` can be used for conditional styling (green = GOOD, red = BAD)
                - Detail view can be routed using `/view-detail-analysis/{id}`
            -->
            <RecentAnalysisTable
                :analyses="[
                    {
                        id: 1,
                        imageName: 'product_001.jpg',
                        result: 'GOOD',
                        score: 0.95,
                    },
                    {
                        id: 2,
                        imageName: 'product_002.jpg',
                        result: 'BAD',
                        score: 0.8,
                    },
                    {
                        id: 3,
                        imageName: 'product_003.jpg',
                        result: 'GOOD',
                        score: 0.98,
                    },
                    {
                        id: 4,
                        imageName: 'product_004.jpg',
                        result: 'BAD',
                        score: 0.72,
                    },
                    {
                        id: 5,
                        imageName: 'product_005.jpg',
                        result: 'GOOD',
                        score: 0.92,
                    },
                ]"
            />
        </div>
        <div
            class="flex flex-col justify-start gap-4 p-6 border border-gray-200 dark:border-dark-700 rounded-lg"
        >
            <h2
                class="font-medium text-gray-800 dark:text-dark-100 text-sm-plus tracking-wide"
            >
                Analysis Overview
            </h2>

            <!--
                📊 Real-Time Analysis Overview
                Expected backend props:
                    - averageTime: string | number (e.g., "0.5") → Average processing time per image (in seconds)
                    - successRate: number (e.g., 86) → Percentage of successfully processed images
                    - defectRate: number (e.g., 65) → Percentage of defect images among all processed
                    - AIConfidence: number (e.g., 90) → Average model confidence across recent detections

                Structure:
                {
                    averageTime: "0.5",       // in seconds
                    successRate: 86,          // in percent (%)
                    defectRate: 65,           // in percent (%)
                    AIConfidence: 90          // in percent (%)
                }

                Notes:
                - Values are usually derived from today's or recent activity window
                - Format `averageTime` with units (e.g., "0.5s") in UI if not already
                - Can be visualized using cards or icon-metric pairs for quick-glance metrics
            -->
            <AnalysisOverview
                averageTime="0.5"
                successRate="86"
                defectRate="65"
                AIConfidence="90"
            />
        </div>
    </div>
</template>
