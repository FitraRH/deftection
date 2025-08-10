<script setup>
import { computed } from "vue";

// --- Props ---
const props = defineProps({
    details: {
        type: Object,
        required: true,
        default: () => ({
            parameters: {
                "Total Processing Time": "0.018s",
                "Image Preprocessing": "0.12s",
                "AI Inference": "-0.170s",
                "Result Processing": "0.05s",
                "Model Used": "HRNet + Anomalib",
            },
            metrics: {
                "Anomaly Detection": 76,
                "Classification Accuracy": 79,
                "Overall Confidence": 79,
            },
            rawData: {
                analysis_date: "2025-07-09 09:40:47",
                anomaly_score: 0.7638,
                confidence_level: "High",
                defect_statistics: {
                    area_percentage: 1.2,
                    bbox_count: 1,
                    confidence: 0.88,
                    defect_type: "missing_component",
                },
                final_decision: "DEFECT",
                id: 15,
                image_name: "work.png",
            },
        }),
    },
});

// --- Computed Properties ---
// A helper to format the raw JSON data for display
const formattedRawData = computed(() => {
    return JSON.stringify(props.details.rawData, null, 2);
});

// Helper to get progress bar color based on percentage
const getMetricColor = (value) => {
    if (value > 75) return "bg-green-500";
    if (value > 50) return "bg-yellow-500";
    return "bg-red-500";
};
</script>

<template>
    <div
        class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
    >
        <h3
            class="flex items-center mb-6 gap-4 text-lg font-semibold text-gray-900 dark:text-dark-50"
        >
            <font-awesome-icon icon="fa-solid fa-code" />
            Technical Analysis Details
        </h3>

        <div class="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <!-- Left Side: Parameters and Metrics -->
            <div class="space-y-6">
                <!-- Analysis Parameters -->
                <div>
                    <h2
                        class="mb-3 truncate text-sm-plus font-medium tracking-wide text-gray-800 dark:text-dark-100"
                    >
                        Analysis Parameters
                    </h2>
                    <dl
                        class="text-sm border-t border-gray-200 dark:border-dark-600"
                    >
                        <div
                            v-for="(value, key) in details.parameters"
                            :key="key"
                            class="grid grid-cols-2 px-0 py-2 border-b border-gray-200 dark:border-dark-600"
                        >
                            <dt class="font-mono truncate">{{ key }}</dt>
                            <dd class="font-mono text-right truncate">
                                {{ value }}
                            </dd>
                        </div>
                    </dl>
                </div>

                <!-- Model Confidence Metrics -->
                <div>
                    <h2
                        class="mb-3 truncate text-sm-plus font-medium tracking-wide text-gray-800 dark:text-dark-100"
                    >
                        Model Confidence Metrics
                    </h2>
                    <div class="space-y-4">
                        <div v-for="(value, key) in details.metrics" :key="key">
                            <div class="flex justify-between mb-2 text-sm">
                                <span class="truncate">{{ key }}</span>
                                <span class="truncate">{{ value }}%</span>
                            </div>
                            <div
                                class="w-full bg-gray-200 rounded-full h-2.5 dark:bg-dark-700"
                            >
                                <div
                                    class="h-2.5 rounded-full"
                                    :class="getMetricColor(value)"
                                    :style="{ width: `${value}%` }"
                                ></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Side: Raw JSON Data -->
            <div class="h-full flex flex-col">
                <h2
                    class="mb-3 truncate text-sm-plus font-medium tracking-wide text-gray-800 dark:text-dark-100"
                >
                    Raw Analysis Data
                </h2>
                <div
                    class="flex-1 border border-gray-200 rounded-lg dark:border-dark-600"
                >
                    <pre
                        class="p-4 text-xs text-gray-800 dark:text-dark-100 whitespace-pre-wrap"
                    ><code>{{ formattedRawData }}</code></pre>
                </div>
            </div>
        </div>
    </div>
</template>
