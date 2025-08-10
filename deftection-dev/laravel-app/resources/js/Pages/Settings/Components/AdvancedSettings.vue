<script setup>
import { useForm } from "@inertiajs/vue3";

// --- Props ---
const props = defineProps({
    settings: {
        type: Object,
        // Mock data for visualization
        default: () => ({
            systemInfo: {
                "System Version": "v2.0.0",
                "API Status": "Unknown",
                "Models Loaded": "Yes",
                Database: "Connected",
            },
            performance: {
                maxConcurrentAnalyses: 2,
                cacheDuration: 24,
            },
            dataManagement: {
                autoCleanup: true,
                backupToCloud: false,
            },
        }),
    },
});

// --- Emits ---
// Defines events for the dangerous actions. The parent component will handle these.
const emit = defineEmits(["clear-data", "reset-settings"]);

// --- Form Management ---
const form = useForm({
    maxConcurrentAnalyses: props.settings.performance.maxConcurrentAnalyses,
    cacheDuration: props.settings.performance.cacheDuration,
    autoCleanup: props.settings.dataManagement.autoCleanup,
    backupToCloud: props.settings.dataManagement.backupToCloud,
});

// Helper to get status text color
const getStatusColor = (status) => {
    if (status === "Connected" || status === "Yes") return "text-green-500";
    if (status === "Unknown") return "text-yellow-500";
    return "text-red-500";
};
</script>

<template>
    <div
        class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
    >
        <h2
            class="mb-1 flex items-center text-lg font-semibold text-gray-900 dark:text-dark-50"
        >
            Advanced Configuration
        </h2>
        <p class="mb-6 text-xs text-gray-400 dark:text-dark-300">
            Advanced settings for system behavior and performance.
        </p>

        <!-- Section 1: System Information -->
        <div class="p-4 border border-gray-200 rounded-lg dark:border-dark-600">
            <h4 class="font-medium text-gray-900 dark:text-dark-50">
                System Information
            </h4>
            <dl class="mt-2 text-sm">
                <div
                    v-for="(value, key) in settings.systemInfo"
                    :key="key"
                    class="grid grid-cols-2 py-1"
                >
                    <dt class="text-gray-400 dark:text-dark-300">{{ key }}</dt>
                    <dd
                        class="font-medium text-right"
                        :class="getStatusColor(value)"
                    >
                        {{ value }}
                    </dd>
                </div>
            </dl>
        </div>

        <hr class="my-6 border-gray-200 dark:border-dark-600" />

        <!-- Section 2: Performance Settings -->
        <div>
            <h2
                class="flex items-center text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                Performance Settings
            </h2>
            <div class="mt-4 space-y-4">
                <div>
                    <label
                        for="maxConcurrent"
                        class="text-sm font-medium text-gray-800 dark:text-dark-100"
                        >Max Concurrent Analyses</label
                    >
                    <div class="mt-1 max-w-xs">
                        <div class="input-root undefined">
                            <div class="input-wrapper relative">
                                <select
                                    v-model="form.maxConcurrentAnalyses"
                                    id="maxConcurrent"
                                    class="form-select-base form-select block w-full max-w-xs mt-1 text-sm ltr:pr-9 rtl:pl-9 peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                                >
                                    <option :value="1">1 (Slowest)</option>
                                    <option :value="2">2 (Balanced)</option>
                                    <option :value="4">4 (Fastest)</option>
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
                    <p class="mt-1 text-xs text-gray-400 dark:text-dark-300">
                        Number of images that can be processed simultaneously.
                    </p>
                </div>
                <div>
                    <label
                        for="cacheDuration"
                        class="text-sm font-medium text-gray-800 dark:text-dark-100"
                        >Cache Duration (hours)</label
                    >
                    <input
                        v-model.number="form.cacheDuration"
                        type="number"
                        id="cacheDuration"
                        class="mt-1 max-w-xs form-input-base form-input peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                    />
                    <p class="mt-1 text-xs text-gray-400 dark:text-dark-300">
                        How long to keep processed results in cache.
                    </p>
                </div>
            </div>
        </div>

        <hr class="my-6 border-gray-200 dark:border-dark-600" />

        <!-- Section 3: Danger Zone -->
        <div
            class="p-4 mt-6 border-2 border-dashed rounded-lg border-red-300 dark:border-red-500/50"
        >
            <h2
                class="flex items-center font-bold text-red-700 dark:text-red-400"
            >
                Danger Zone
            </h2>
            <div class="mt-4 space-y-4">
                <div
                    class="flex flex-col sm:flex-row sm:items-center sm:justify-between"
                >
                    <div>
                        <p
                            class="mb-1 text-sm font-medium text-gray-800 dark:text-dark-100"
                        >
                            Clear All Analysis Data
                        </p>
                        <p class="text-xs text-gray-400 dark:text-dark-300">
                            Permanently delete all analysis history and uploaded
                            images.
                        </p>
                    </div>
                    <button
                        @click="emit('clear-data')"
                        class="btn btn-base px-4 py-2 mt-2 gap-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-md shadow-sm sm:mt-0 sm:ml-4 hover:bg-red-700"
                    >
                        <font-awesome-icon icon="fa-solid fa-trash-can" />
                        Clear Data
                    </button>
                </div>
                <div
                    class="flex flex-col pt-4 mt-4 border-t border-red-200 sm:flex-row sm:items-center sm:justify-between dark:border-red-500/30"
                >
                    <div>
                        <p
                            class="mb-1 text-sm font-medium text-gray-800 dark:text-dark-100"
                        >
                            Reset All Settings
                        </p>
                        <p class="text-xs text-gray-400 dark:text-dark-300">
                            Reset all configuration to factory defaults.
                        </p>
                    </div>
                    <button
                        @click="emit('reset-settings')"
                        class="btn btn-base px-4 py-2 mt-2 gap-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-md shadow-sm sm:mt-0 sm:ml-4 hover:bg-red-700"
                    >
                        <font-awesome-icon
                            icon="fa-solid fa-clock-rotate-left"
                        />
                        Reset All
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>
