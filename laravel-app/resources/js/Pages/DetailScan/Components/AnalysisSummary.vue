<script setup>
import { ref, computed } from "vue";

// --- Props ---
// This component will receive a single object with all the summary data.
const props = defineProps({
    summary: {
        type: Object,
        // The default function provides the mock data for visualization.
        default: () => ({
            imageName: "work.png",
            originalSize: "16x16",
            analysisDate: "09/07/2025, 09:43:49",
            processingTime: "0.000s",
            finalDecision: "GOOD",
            anomalyScore: 0.6734,
            confidenceLevel: "Medium",
            status: "Completed",
            processingSpeed: 1.0,
            aiConfidence: 67,
            analysisQuality: "Medium",
        }),
    },
});

// --- Computed Properties ---

// Determines the color of the confidence level bar based on the score.
const confidenceTextClass = computed(() => {
    const score = props.summary.anomalyScore;
    if (score > 0.75)
        return {
            score: "high",
            color: "success",
        };
    if (score > 0.5)
        return {
            score: "medium",
            color: "warning",
        };
    return {
        score: "low",
        color: "error",
    };
});

const confidenceAIClass = computed(() => {
    const score = props.summary.aiConfidence;
    if (score > 75)
        return {
            score: "high",
            color: "success",
        };
    if (score > 50)
        return {
            score: "medium",
            color: "warning",
        };
    return {
        score: "low",
        color: "error",
    };
});
</script>

<template>
    <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <!-- Card 1: Image Information -->
        <div
            class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
        >
            <h3
                class="flex items-center mb-6 gap-4 text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                <font-awesome-icon icon="fa-solid fa-image" />
                Image Information
            </h3>
            <div class="space-y-5">
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100">
                        Image Name
                    </span>
                    <span class="text-right">{{ summary.imageName }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Original Size</span
                    >
                    <span class="text-right">{{ summary.originalSize }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Analysis Date</span
                    >
                    <span class="text-right">{{ summary.analysisDate }}</span>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Processing Time</span
                    >
                    <div
                        class="badge-base badge this:info text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20"
                    >
                        {{ summary.processingTime }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Card 2: Detection Results -->
        <div
            class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
        >
            <h3
                class="flex items-center mb-6 gap-4 text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                <font-awesome-icon icon="fa-solid fa-magnifying-glass" />
                Detection Results
            </h3>
            <div class="space-y-4 text-sm">
                <div class="flex items-center justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Final Decision</span
                    >
                    <div
                        v-if="summary.finalDecision === 'GOOD'"
                        class="badge-base badge this:success text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20"
                    >
                        {{ summary.finalDecision }}
                    </div>
                    <div
                        v-else
                        class="badge-base badge this:error text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20"
                    >
                        DEFECT
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Anomaly Score</span
                    >
                    <div
                        class="badge-base badge text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 uppercase"
                        :class="`this:${confidenceTextClass.color}`"
                    >
                        {{ summary.anomalyScore.toFixed(4) }}
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Confidence Level</span
                    >
                    <div
                        class="badge-base badge text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 uppercase"
                        :class="`this:${confidenceTextClass.color}`"
                    >
                        {{ confidenceTextClass.score }}
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Status</span
                    >
                    <div
                        class="badge-base badge this:success text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 uppercase"
                    >
                        {{ summary.status }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Card 3: Performance Metrics -->
        <div
            class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
        >
            <h3
                class="flex items-center mb-6 gap-4 text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                <font-awesome-icon icon="fa-solid fa-chart-column" />
                Performance Metrics
            </h3>
            <div class="space-y-4 text-sm">
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Processing Speed</span
                    >
                    <div
                        class="badge-base badge this:info text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 uppercase"
                    >
                        {{ summary.processingSpeed.toFixed(1) }} FPS
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >AI Confidence</span
                    >
                    <div
                        class="badge-base badge text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20"
                        :class="`this:${confidenceAIClass.color}`"
                    >
                        {{ summary.aiConfidence }}%
                    </div>
                </div>
                <div class="flex justify-between">
                    <span class="font-medium text-gray-800 dark:text-dark-100"
                        >Analysis Quality</span
                    >
                    <div
                        class="badge-base badge text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 border border-this-darker/20 dark:border-this-lighter/20 uppercase"
                        :class="`this:${confidenceAIClass.color}`"
                    >
                        {{ confidenceAIClass.score }}
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
