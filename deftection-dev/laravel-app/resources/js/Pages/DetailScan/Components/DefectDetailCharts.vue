<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
// 1. Import the ApexCharts component
import VueApexCharts from "vue3-apexcharts";

// --- Props ---
// The component now expects an array of defect objects.
const props = defineProps({
    defects: {
        type: Array,
        required: true,
        default: () => [
            {
                name: "Missing Component",
                confidence: 73,
                coverage: 5.1,
                regions: 2,
            },
            {
                name: "Scratch",
                confidence: 92,
                coverage: 2.3,
                regions: 1,
            },
            {
                name: "Opened",
                confidence: 33,
                coverage: 3.1,
                regions: 1,
            },
        ],
    },
});

// --- NEW: Local Theme Management ---
// A reactive ref to hold the current theme state.
const theme = ref("dark"); // Default to dark

// A computed property that returns true if the theme is dark.
const isDarkMode = computed(() => theme.value === "dark");

// Function to check and update the theme from localStorage.
const checkTheme = () => {
    // Ensure code runs only in the browser where localStorage is available.
    if (typeof window !== "undefined") {
        theme.value = localStorage.getItem("theme") || "dark"; // Default to dark if nothing is set
    }
};

// When the component is mounted, check the theme and listen for changes.
onMounted(() => {
    checkTheme();
    window.addEventListener("storage", checkTheme);
});

// When the component is unmounted, clean up the event listener.
onUnmounted(() => {
    window.removeEventListener("storage", checkTheme);
});

// --- NEW: Helper function to get dynamic styles for defect cards ---
const getDefectCardStyles = (confidence) => {
    if (confidence > 75) {
        // High confidence - Error variant (Red)
        return {
            card: "bg-[var(--color-error-darker)]/[0.07] border-[var(--color-error-darker)]/20 dark:bg-red-500/10 dark:border-red-500/30",
            title: "text-[var(--color-error-darker)] dark:text-red-300",
            text: "text-[var(--color-error-darker)] dark:text-red-400",
            badge: "bg-[var(--color-error)]",
        };
    }
    if (confidence > 50) {
        // Medium confidence - Warning variant (Amber)
        return {
            card: "bg-[var(--color-warning-darker)]/[0.07] border-[var(--color-warning-darker)]/20 dark:bg-yellow-500/10 dark:border-yellow-500/30",
            title: "text-[var(--color-warning-darker)] dark:text-yellow-300",
            text: "text-[var(--color-warning-darker)] dark:text-yellow-400",
            badge: "bg-[var(--color-warning)]",
        };
    }
    // Low confidence - Info variant (Blue)
    return {
        card: "bg-blue-50 border-blue-200 dark:bg-blue-500/10 dark:border-blue-500/30",
        title: "text-blue-800 dark:text-blue-300",
        text: "text-blue-700 dark:text-blue-400",
        badge: "bg-blue-500",
    };
};

// --- ApexCharts Configuration ---

// --- Bar Chart ---
const barChartSeries = computed(() => [
    { name: "Confidence", data: props.defects.map((d) => d.confidence) },
]);

const barChartOptions = computed(() => ({
    chart: { type: "bar", toolbar: { show: false } },
    plotOptions: {
        bar: { horizontal: false, columnWidth: "50%", distributed: true },
    },
    dataLabels: { enabled: false },
    xaxis: {
        categories: props.defects.map((d) => d.name),
        labels: {
            style: {
                colors: isDarkMode.value ? "#9CA3AF" : "#6B7280",
                fontSize: "12px",
            },
        },
    },
    yaxis: {
        max: 100,
        labels: {
            style: {
                colors: isDarkMode.value ? "#9CA3AF" : "#6B7280",
            },
        },
    },
    grid: {
        borderColor: isDarkMode.value ? "#374151" : "#E5E7EB",
        strokeDashArray: 4,
    },
    tooltip: {
        theme: isDarkMode.value ? "dark" : "light",
        y: { formatter: (val) => `${val.toFixed(2)}%` },
    },
    legend: { show: false },
    theme: {
        monochrome: {
            enabled: true,
            color: "#ff4f1a",
            shadeTo: isDarkMode.value ? "dark" : "light",
        },
    },
}));

// --- Donut Chart ---
const doughnutChartSeries = computed(() =>
    props.defects.map((d) => d.coverage)
);

// Computed property for the Donut Chart's options
const doughnutChartOptions = computed(() => ({
    chart: { type: "donut" },
    labels: props.defects.map((d) => d.name),
    plotOptions: { pie: { donut: { size: "70%" } } },
    dataLabels: {
        enabled: false,
    },
    legend: {
        position: "bottom",
        labels: {
            colors: isDarkMode.value ? "#D1D5DB" : "#374151",
        },
    },
    tooltip: {
        theme: isDarkMode.value ? "dark" : "light",
        y: { formatter: (val) => `${val.toFixed(1)}%` },
    },
    states: { hover: { filter: { type: "none" } } },
    theme: {
        monochrome: {
            enabled: true,
            color: "#ff4f1a",
            shadeTo: isDarkMode.value ? "dark" : "light",
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
            <font-awesome-icon icon="fa-solid fa-triangle-exclamation" />
            Detected Defects Analysis
        </h3>

        <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <!-- Left Side: Defect Info  -->
            <div
                class="flex flex-col justify-between gap-6"
                :class="{
                    'justify-between': defects.length > 2,
                    'justify-start': defects.length <= 2,
                }"
            >
                <!-- Defect Summary Cards: Now loops through each defect -->
                <div
                    v-for="defect in defects"
                    :key="defect.name"
                    class="p-4 border rounded-lg"
                    :class="getDefectCardStyles(defect.confidence).card"
                >
                    <div class="flex items-center justify-between">
                        <h4
                            class="font-bold"
                            :class="
                                getDefectCardStyles(defect.confidence).title
                            "
                        >
                            {{ defect.name }}
                        </h4>
                        <span
                            class="px-2 py-1 text-xs font-semibold text-white rounded-md"
                            :class="
                                getDefectCardStyles(defect.confidence).badge
                            "
                        >
                            {{
                                defect.confidence > 75
                                    ? "High"
                                    : defect.confidence > 50
                                    ? "Medium"
                                    : "Low"
                            }}
                        </span>
                    </div>
                    <div
                        class="mt-2 text-sm space-y-1"
                        :class="getDefectCardStyles(defect.confidence).text"
                    >
                        <div class="flex justify-between">
                            <span>Confidence:</span>
                            <span>{{ defect.confidence }}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Coverage:</span>
                            <span>{{ defect.coverage }}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Regions:</span>
                            <span>{{ defect.regions }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Bar Chart -->
            <div
                class="p-4 border border-gray-200 rounded-lg dark:border-dark-600"
            >
                <h5
                    class="mb-2 font-medium text-center text-gray-700 dark:text-dark-200"
                >
                    Defect Confidence Levels
                </h5>
                <div class="h-full">
                    <VueApexCharts
                        type="bar"
                        height="400"
                        :options="barChartOptions"
                        :series="barChartSeries"
                    />
                </div>
            </div>

            <!-- Right Side: Donut Chart -->
            <div
                class="p-4 border border-gray-200 rounded-lg dark:border-dark-600"
            >
                <h5
                    class="mb-2 font-medium text-center text-gray-700 dark:text-dark-200"
                >
                    Area Coverage Distribution
                </h5>
                <div class="h-full">
                    <VueApexCharts
                        type="donut"
                        height="400"
                        :options="doughnutChartOptions"
                        :series="doughnutChartSeries"
                    />
                </div>
            </div>
        </div>
    </div>
</template>
