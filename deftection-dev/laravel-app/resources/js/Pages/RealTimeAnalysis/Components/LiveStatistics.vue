<script setup>
import { ref, defineExpose } from "vue";

// --- Props ---
// This component receives all its data from the parent.
const props = defineProps({
    totalFrames: { type: Number, default: 0 },
    goodProducts: { type: Number, default: 0 },
    defectiveProducts: { type: Number, default: 0 },
    detectionRate: { type: Number, default: 0 }, // Should be a value between 0 and 1
    screenshots: { type: Number, default: 0 },
});

// --- Expose Method ---
// The parent component can call this method on the component's ref to reset the stats.
// This is an alternative to passing a prop down to trigger a reset.
const reset = () => {
    // This component doesn't own the state, so it doesn't reset anything itself.
    // The parent will handle the state reset. This method is here to complete the pattern
    // if we were to handle state internally. For now, it's a placeholder.
    console.log("LiveStatistics component reset method called.");
};

defineExpose({
    reset,
});
</script>

<template>
    <div
        class="flex flex-col justify-start border border-gray-200 dark:border-dark-700 rounded-lg p-6 gap-4"
    >
        <h2
            class="truncate text-base font-medium tracking-wide text-gray-800 dark:text-dark-100"
        >
            Live Statistics
        </h2>

        <div class="space-y-3">
            <div class="flex items-center justify-between">
                <span>Total Frames</span>
                <span>
                    {{ totalFrames }}
                </span>
            </div>
            <div class="flex items-center justify-between">
                <span>Good Products</span>
                <span class="text-success dark:text-success-lighter">
                    {{ goodProducts }}
                </span>
            </div>
            <div class="flex items-center justify-between">
                <span>Defective Products</span>
                <span class="text-error dark:text-error-lighter">
                    {{ defectiveProducts }}
                </span>
            </div>
            <div class="flex items-center justify-between">
                <span>Detection Rate</span>
                <span class="text-warning dark:text-warning-lighter">
                    {{ (detectionRate * 100).toFixed(0) }}%
                </span>
            </div>
            <div class="flex items-center justify-between">
                <span>Screenshots</span>
                <span class="text-info dark:text-info-lighter">
                    {{ screenshots }}
                </span>
            </div>
        </div>
    </div>
</template>
