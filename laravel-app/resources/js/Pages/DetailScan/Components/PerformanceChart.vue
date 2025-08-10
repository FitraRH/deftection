<script setup>
import { computed } from "vue";
import VueApexCharts from "vue3-apexcharts";

// --- Props ---
// This component will receive the performance data from the parent.
const props = defineProps({
    performanceData: {
        type: Object,
        required: true,
        // Mock data for visualization
        default: () => ({
            timeBreakdown: {
                preprocessing: 0.12,
                aiInference: 0.75,
                postprocessing: 0.08,
            },
            modelPerformance: {
                accuracy: 99,
                precision: 85,
                reliability: 95,
                speed: 70,
            },
        }),
    },
});

// --- ApexCharts Configuration ---

// --- Pie Chart for Time Breakdown ---
const pieChartSeries = computed(() => {
    const data = props.performanceData.timeBreakdown;
    return [data.preprocessing, data.aiInference, data.postprocessing];
});

const pieChartOptions = computed(() => ({
    chart: {
        type: "pie",
    },
    labels: ["Preprocessing", "AI Inference", "Post-processing"],
    colors: ["#3B82F6", "#10B981", "#F59E0B"], // Blue, Green, Amber
    dataLabels: {
        enabled: false,
    },
    legend: {
        position: "bottom",
        labels: {
            colors: "#D1D5DB", // gray-300 for dark mode text
        },
    },
    tooltip: {
        theme: "dark",
        y: {
            formatter: (val) => `${val.toFixed(2)}s`,
            title: {
                formatter: (seriesName) => `${seriesName}:`,
            },
        },
    },
}));

// --- Radar Chart for AI Model Performance ---
const radarChartSeries = computed(() => {
    const data = props.performanceData.modelPerformance;
    return [
        {
            name: "Performance Score",
            data: [data.accuracy, data.precision, data.reliability, data.speed],
        },
    ];
});

const radarChartOptions = computed(() => ({
    chart: {
        type: "radar",
        toolbar: { show: false },
    },
    xaxis: {
        categories: ["Accuracy", "Precision", "Reliability", "Speed"],
        labels: {
            style: {
                colors: ["#D1D5DB", "#D1D5DB", "#D1D5DB", "#D1D5DB"], // gray-300
            },
        },
    },
    yaxis: {
        show: true,
        min: 0,
        max: 100,
        tickAmount: 5,
        labels: {
            style: {
                colors: "#9CA3AF", // gray-400
            },
            formatter: (val) => val.toFixed(0),
        },
    },
    stroke: {
        width: 2,
        colors: ["#3B82F6"], // Blue
    },
    fill: {
        opacity: 0.1,
        colors: ["#3B82F6"],
    },
    markers: {
        size: 4,
        colors: ["#FFF"],
        strokeColors: "#3B82F6",
        strokeWidth: 2,
    },
    tooltip: {
        theme: "dark",
        y: {
            formatter: (val) => val.toFixed(0),
        },
    },
    plotOptions: {
        radar: {
            polygons: {
                strokeColors: "#4B5563", // gray-600
                connectorColors: "#4B5563",
            },
        },
    },
}));
</script>

<template>
    <div
        class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
    >
        <h3
            class="flex items-center mb-6 gap-4 text-lg font-semibold text-gray-900 dark:text-dark-50"
        >
            <font-awesome-icon icon="fa-solid fa-chart-pie" />
            Analysis Performance Metrics
        </h3>

        <div class="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <!-- Left Side: Pie Chart -->
            <div
                class="p-4 border border-gray-200 rounded-lg dark:border-dark-600"
            >
                <h5
                    class="mb-4 font-medium text-center text-gray-700 dark:text-dark-200"
                >
                    Processing Time Breakdown
                </h5>
                <VueApexCharts
                    type="pie"
                    height="420"
                    :options="pieChartOptions"
                    :series="pieChartSeries"
                />
            </div>

            <!-- Right Side: Radar Chart -->
            <div
                class="flex flex-col border border-gray-200 rounded-lg dark:border-dark-600"
            >
                <h5
                    class="pt-4 font-medium text-center text-gray-700 dark:text-dark-200"
                >
                    AI Model Performance
                </h5>
                <div class="">
                    <VueApexCharts
                        type="radar"
                        :options="radarChartOptions"
                        :series="radarChartSeries"
                    />
                </div>
            </div>
        </div>
    </div>
</template>
