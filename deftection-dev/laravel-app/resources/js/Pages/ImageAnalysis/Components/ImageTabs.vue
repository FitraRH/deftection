<script setup>
import { ref } from "vue";
import DetectionResults from "./DetectionResults.vue";

const props = defineProps({
    originalUrl: {
        type: String,
    },
    detectionUrl: {
        type: String,
    },
    showSteps: Boolean,
    hasRunAnalysis: Boolean,
    detectionResult: Object,
});
</script>

<template>
    <div
        class="h-full p-6 border border-gray-200 dark:border-dark-500 rounded-lg"
    >
        <!-- Tab Headers -->
        <h2
            class="text-sm-plus font-medium tracking-wide text-gray-800 dark:text-dark-100"
        >
            Analysis Results
        </h2>

        <!-- Tab Content -->
        <div class="h-full flex flex-col items-center justify-center p-4">
            <template v-if="!hasRunAnalysis">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke-width="0.5"
                    stroke="currentColor"
                    class="size-30"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Zm10.5-11.25h.008v.008h-.008V8.25Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z"
                    />
                </svg>

                <p class="text-sm text-gray-500 dark:text-dark-300 text-center">
                    Upload an image and click <strong>"Run Analysis"</strong> to
                    see the results.
                </p>
            </template>

            <template v-else>
                <div class="flex flex-col gap-5">
                    <div class="flex gap-5">
                        <!-- Original Image -->
                        <div class="flex flex-col items-center">
                            <p
                                class="mb-2 text-sm font-medium text-gray-700 dark:text-dark-100"
                            >
                                Original
                            </p>
                            <img
                                :src="originalUrl"
                                class="w-full max-h-[300px] object-contain rounded"
                            />
                        </div>

                        <div
                            class="mx-4 mt-6 w-px bg-gray-200 dark:bg-dark-500"
                        ></div>

                        <!-- Detection Image -->
                        <div class="flex flex-col items-center">
                            <p
                                class="mb-2 text-sm font-medium text-gray-700 dark:text-dark-100"
                            >
                                Detection Result
                            </p>
                            <img
                                :src="detectionUrl"
                                class="w-full max-h-[300px] object-contain rounded"
                            />
                        </div>
                    </div>
                </div>

                <!-- Detection Result -->
                <DetectionResults :result="detectionResult" />
            </template>
        </div>
    </div>
</template>
