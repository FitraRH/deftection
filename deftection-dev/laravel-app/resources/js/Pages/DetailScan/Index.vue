<script setup>
import { ref, computed } from "vue";

import AnalysisSummary from "./Components/AnalysisSummary.vue";
import ImageComparison from "./Components/ImageComparison.vue";
import DefectDetailCharts from "./Components/DefectDetailCharts.vue";
import PerformanceChart from "./Components/PerformanceChart.vue";
import ProductQuality from "./Components/ProductQuality.vue";
import TechnicalDetails from "./Components/TechnicalDetails.vue";

// --- Props Definition ---
// Receive data from Laravel controller via Inertia
const props = defineProps({
    analysis: {
        type: Object,
        required: true,
    },
    title: {
        type: String,
        default: "Scan Analysis Details",
    },
    scan: {
        type: Object,
        default: () => ({}),
    },
});

// --- Mock Data (for development/fallback) ---
// Keep the mock data as fallback or for development purposes
const mockGoodAnalysisData = {
    id: 16,
    status: "good",
    summary: {
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
    },
    visuals: {
        status: "good",
        originalImageUrl:
            "https://placehold.co/600x400/FFFFFF/000000?text=Original+Image",
        analyzedImageUrl: "", // Not needed for 'good' status
    },
    defects: [], // No defects for a good analysis
    performance: {
        timeBreakdown: {
            preprocessing: 0.1,
            aiInference: 0.5,
            postprocessing: 0.05,
        },
        modelPerformance: {
            accuracy: 99,
            precision: 95,
            reliability: 98,
            speed: 90,
        },
    },
    technical: {
        parameters: {
            "Total Processing Time": "0.000s",
            "Image Preprocessing": "0.12s",
            "AI Inference": "-0.170s",
            "Result Processing": "0.05s",
            "Model Used": "HRNet + Anomalib",
        },
        metrics: {
            "Anomaly Detection": 67,
            "Classification Accuracy": 94,
            "Overall Confidence": 72,
        },
        rawData: {
            analysis_date: "2025-07-09 09:43:49",
            anomaly_score: 0.67339,
            final_decision: "GOOD",
            id: 16,
            image_name: "work.png",
        },
    },
};

const mockDefectAnalysisData = {
    id: 15,
    status: "defect",
    summary: {
        imageName: "work.png",
        originalSize: "16x16",
        analysisDate: "09/07/2025, 09:40:47",
        processingTime: "0.018s",
        finalDecision: "DEFECT",
        anomalyScore: 0.7638,
        confidenceLevel: "High",
        status: "Completed",
        processingSpeed: 56.4,
        aiConfidence: 76,
        analysisQuality: "High",
    },
    visuals: {
        status: "defect",
        originalImageUrl:
            "https://placehold.co/600x400/FFFFFF/000000?text=Original+Image",
        analyzedImageUrl: "https://i.imgur.com/pUq3f7A.png",
    },
    defects: [
        {
            name: "Missing Component",
            confidence: 79,
            coverage: 5.1,
            regions: 2,
        },
        { name: "Scratch", confidence: 92, coverage: 2.3, regions: 1 },
    ],
    performance: {
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
    },
    technical: {
        parameters: {
            "Total Processing Time": "0.018s",
            "Image Preprocessing": "0.12s",
            "AI Inference": "-0.152s",
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
            final_decision: "DEFECT",
            id: 15,
            defects: ["Missing Component", "Scratch"],
        },
    },
};

// --- Reactive State ---
// Use the passed analysis data from the controller, with mock as fallback
const analysis = ref(props.analysis || mockDefectAnalysisData);

// --- Computed Properties ---
const isDefectStatus = computed(() => analysis.value.status === "defect");
const isGoodStatus = computed(() => analysis.value.status === "good");

// --- Development Helper (can be removed in production) ---
const isDevelopment = computed(() => import.meta.env.DEV);

const toggleLayout = () => {
    if (analysis.value.status === "defect") {
        analysis.value = mockGoodAnalysisData;
    } else {
        analysis.value = mockDefectAnalysisData;
    }
};

// --- Error Handling ---
const hasValidAnalysis = computed(() => {
    return (
        analysis.value &&
        analysis.value.summary &&
        analysis.value.visuals &&
        analysis.value.performance &&
        analysis.value.technical
    );
});

// --- Debug Info (development only) ---
const debugInfo = computed(() => ({
    hasAnalysis: !!props.analysis,
    analysisId: analysis.value?.id,
    status: analysis.value?.status,
    propsTitle: props.title,
    scanInfo: props.scan,
}));
</script>

<template>
    <!-- Development Debug Panel (only shows in development) -->
    <div v-if="isDevelopment" class="bg-gray-100 mb-4 p-4 rounded-lg text-sm">
        <details class="cursor-pointer">
            <summary class="mb-2 font-semibold text-gray-700">
                🛠 Development Debug Info
            </summary>
            <div class="space-y-2 text-xs">
                <div>
                    <strong>Has Analysis:</strong> {{ debugInfo.hasAnalysis }}
                </div>
                <div>
                    <strong>Analysis ID:</strong> {{ debugInfo.analysisId }}
                </div>
                <div><strong>Status:</strong> {{ debugInfo.status }}</div>
                <div><strong>Title:</strong> {{ debugInfo.propsTitle }}</div>
                <div>
                    <strong>Scan Info:</strong>
                    {{ JSON.stringify(debugInfo.scanInfo) }}
                </div>
            </div>
            <button
                @click="toggleLayout"
                class="bg-indigo-600 hover:bg-indigo-700 mt-2 px-3 py-1 rounded text-white text-xs"
            >
                Toggle Mock Layout (Currently:
                {{ analysis.status?.toUpperCase() }})
            </button>
        </details>
    </div>

    <!-- Error State -->
    <div
        v-if="!hasValidAnalysis"
        class="bg-red-50 p-6 border border-red-200 rounded-lg"
    >
        <div class="flex items-center">
            <div class="flex-shrink-0">
                <svg
                    class="w-5 h-5 text-red-400"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                >
                    <path
                        fill-rule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clip-rule="evenodd"
                    />
                </svg>
            </div>
            <div class="ml-3">
                <h3 class="font-medium text-red-800 text-sm">
                    Analysis Data Missing
                </h3>
                <div class="mt-2 text-red-700 text-sm">
                    <p>
                        Unable to load scan analysis data. Please try refreshing
                        the page or contact support.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div v-else class="flex flex-col gap-6">
        <!-- Page Title -->
        <div class="pb-4 border-gray-200 border-b">
            <h1 class="font-bold text-gray-900 text-2xl">{{ title }}</h1>
            <p class="mt-1 text-gray-600 text-sm">
                Analysis for {{ analysis.summary?.imageName }} • Scan ID:
                {{ analysis.id }} •
                {{ analysis.summary?.analysisDate }}
            </p>
        </div>

        <!-- Analysis Summary -->
        <AnalysisSummary :summary="analysis.summary" />

        <!-- Image Comparison -->
        <ImageComparison :comparison-data="analysis.visuals" />

        <!-- Defect Details (only for defect status) -->
        <DefectDetailCharts v-if="isDefectStatus" :defects="analysis.defects" />

        <!-- Performance Chart -->
        <PerformanceChart :performance-data="analysis.performance" />

        <!-- Product Quality (only for good status) -->
        <ProductQuality v-if="isGoodStatus" />

        <!-- Technical Details -->
        <TechnicalDetails :details="analysis.technical" />
    </div>
</template>
