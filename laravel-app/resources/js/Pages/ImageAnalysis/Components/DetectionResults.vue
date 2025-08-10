<script setup>
import { computed } from "vue";

const props = defineProps({
    result: {
        type: Object,
        required: true,
    },
});

const viewDetailedAnalysis = () => {
    if (!props.result?.id) return;

    // Navigate to a new Inertia page
    // router.visit(route('', { id: props.result.id }))
};

// Optional formatting
const formattedScore = computed(() => props.result.score?.toFixed(4) || "N/A");
const formattedTime = computed(() => `${(props.result.time ?? 0).toFixed(2)}s`);
</script>

<template>
    <div class="w-full max-w-xl p-6 pb-0">
        <div class="space-y-2 text-sm text-gray-700 dark:text-dark-100">
            <div class="flex justify-between">
                <span>Decision:</span>
                <span
                    :class="
                        props.result.decision === 'GOOD'
                            ? 'text-green-600'
                            : 'text-red-500'
                    "
                >
                    {{ props.result.decision }}
                </span>
            </div>
            <div class="flex justify-between">
                <span>Anomaly Score:</span>
                <strong>{{ formattedScore }}</strong>
            </div>
            <div class="flex justify-between">
                <span>Processing Time:</span>
                <strong>{{ formattedTime }}</strong>
            </div>
            <div class="flex justify-between">
                <span>Defects Found:</span>
                <strong>{{ props.result.defects }}</strong>
            </div>
        </div>

        <div class="mt-4 space-y-2">
            <button
                @click="viewDetailedAnalysis"
                class="btn btn-base w-full rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 transition flex items-center justify-center space-x-2"
            >
                <font-awesome-icon icon="fa-solid fa-file-image" />
                <span>View Analysis Detail</span>
            </button>
        </div>
    </div>
</template>
