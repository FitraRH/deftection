<script setup>
// This component receives the analysis ID as a prop to display it.
const props = defineProps({
    analysisId: {
        type: [String, Number],
        required: true,
    },
});

// Defines the events this component can send up to the parent page.
const emit = defineEmits(["download-report", "delete-analysis"]);

const handleDownload = () => {
    emit("download-report", props.analysisId);
};

const handleDelete = () => {
    emit("delete-analysis", props.analysisId);
};
</script>

<template>
    <div
        class="app-header transition-content sticky top-0 z-20 flex h-[90px] items-center gap-1 border-b border-gray-200 bg-white/80 px-(--margin-x) backdrop-blur-sm backdrop-saturate-150 dark:border-dark-600 max-sm:justify-between dark:bg-dark-900/80"
    >
        <div class="flex items-center gap-2 sm:flex-1">
            <!-- Left Side: Title -->
            <div class="flex items-center gap-4 flex-1">
                <div class="space-y-2">
                    <h1
                        class="text-xl font-bold text-gray-900 dark:text-dark-50"
                    >
                        Enhanced Analysis Details
                    </h1>
                    <div
                        class="badge-base badge bg-gray-200 text-gray-900 dark:bg-surface-2 dark:text-dark-50"
                    >
                        Report ID: {{ analysisId }}
                    </div>
                </div>
            </div>

            <!-- Right Side: Action Buttons -->
            <div class="flex items-center gap-3">
                <button
                    @click="handleDownload"
                    class="btn-base btn gap-2 this:primary bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
                >
                    <font-awesome-icon icon="fa-solid fa-download" />
                    <span>Download</span>
                </button>
                <button
                    @click="handleDelete"
                    class="btn-base btn gap-2 text-white transition-colors bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                >
                    <font-awesome-icon icon="fa-solid fa-trash-can" />
                    <span>Delete</span>
                </button>
            </div>
        </div>
    </div>
</template>
