<script setup>
import { useForm } from "@inertiajs/vue3";

// --- Props ---
// This component will receive the current detection settings from the parent page.
const props = defineProps({
    settings: {
        type: Object,
        // Mock data for visualization
        default: () => ({
            anomalyThreshold: 0.7,
            defectThreshold: 0.85,
            autoSave: true,
            generateVisualizations: true,
            exportFormat: "pdf",
        }),
    },
});

// --- Form Management ---
// Initialize the form with the settings data.
const form = useForm({
    anomalyThreshold: props.settings.anomalyThreshold,
    defectThreshold: props.settings.defectThreshold,
    autoSave: props.settings.autoSave,
    generateVisualizations: props.settings.generateVisualizations,
    exportFormat: props.settings.exportFormat,
});

// The main "Save Settings" button is in the parent Index.vue.
// In a real app, the parent would read this form's data when saving.
</script>

<template>
    <div
        class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
    >
        <!-- Section 1: Detection Configuration -->
        <div class="flex flex-col">
            <h2
                class="mb-1 flex items-center text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                Detection Configuration
            </h2>
            <p class="mb-6 text-xs text-gray-400 dark:text-dark-300">
                Configure AI detection parameters and thresholds.
            </p>

            <div class="grid grid-cols-1 gap-8 sm:grid-cols-2">
                <!-- Anomaly Detection Threshold Slider -->
                <div>
                    <label
                        for="anomalyThreshold"
                        class="mb-1 text-sm font-medium text-gray-800 dark:text-dark-100"
                        >Anomaly Detection Threshold</label
                    >
                    <div class="flex items-center my-2 space-x-4">
                        <span class="text-xs">0.1</span>
                        <input
                            v-model.number="form.anomalyThreshold"
                            type="range"
                            id="anomalyThreshold"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-dark-700"
                        />
                        <span class="text-xs">1.0</span>
                    </div>
                    <div class="flex justify-between">
                        <p class="text-xs">Sensitive</p>
                        <p
                            class="text-xs font-bold text-blue-600 dark:text-blue-400"
                        >
                            {{ form.anomalyThreshold.toFixed(2) }}
                        </p>
                        <p class="text-xs">Conservative</p>
                    </div>
                </div>
                <!-- Defect Classification Threshold Slider -->
                <div>
                    <label
                        for="defectThreshold"
                        class="mb-1 text-sm font-medium text-gray-800 dark:text-dark-100"
                        >Defect Classification Threshold</label
                    >
                    <div class="flex items-center my-2 space-x-4">
                        <span class="text-xs">0.1</span>
                        <input
                            v-model.number="form.defectThreshold"
                            type="range"
                            id="defectThreshold"
                            min="0.1"
                            max="1.0"
                            step="0.05"
                            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-dark-700"
                        />
                        <span class="text-xs">1.0</span>
                    </div>
                    <div class="flex justify-between">
                        <p class="text-xs">Sensitive</p>
                        <p
                            class="text-xs font-bold text-blue-600 dark:text-blue-400"
                        >
                            {{ form.defectThreshold.toFixed(2) }}
                        </p>
                        <p class="text-xs">Conservative</p>
                    </div>
                </div>
            </div>
        </div>

        <hr class="my-6 border-gray-200 dark:border-dark-600" />

        <!-- Section 2: Export Settings -->
        <div>
            <h2
                class="flex items-center text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                Export Settings
            </h2>
            <div class="mt-6">
                <label
                    for="exportFormat"
                    class="text-sm font-medium text-gray-800 dark:text-dark-100"
                    >Preferred Export Format</label
                >
                <div class="mt-1 max-w-xs">
                    <div class="input-root undefined">
                        <div class="input-wrapper relative">
                            <select
                                v-model="form.exportFormat"
                                id="exportFormat"
                                class="form-select-base form-select block w-full max-w-xs mt-1 text-sm ltr:pr-9 rtl:pl-9 peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                            >
                                <option value="pdf">PDF Report</option>
                                <option value="csv">CSV Data</option>
                                <option value="json">JSON Raw Data</option>
                            </select>
                            <div
                                class="suffix ltr:right-0 rtl:left-0 pointer-events-none absolute top-0 flex h-full w-9 items-center justify-center transition-colors text-gray-400 peer-focus:text-primary-600 dark:text-dark-300 dark:peer-focus:text-primary-500"
                            >
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    viewBox="0 0 20 20"
                                    fill="currentColor"
                                    aria-hidden="true"
                                    data-slot="icon"
                                    class="w-2/3"
                                >
                                    <path
                                        fill-rule="evenodd"
                                        d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                                        clip-rule="evenodd"
                                    ></path>
                                </svg>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<style scoped>
.dark #exportFormat {
    --bg-color: rgb(21, 22, 26);
}

#exportFormat {
    --bg-color: rgb(255, 255, 255);
}
</style>
