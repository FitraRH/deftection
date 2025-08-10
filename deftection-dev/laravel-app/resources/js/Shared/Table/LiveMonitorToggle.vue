<script setup>
import { defineProps, defineEmits } from "vue";
defineProps({
    isPolling: {
        type: Boolean,
        default: false,
    },
    isPollingLoading: {
        type: Boolean,
        default: false,
    },
    isManuallyPaused: {
        type: Boolean,
        default: false,
    },
    isOnline: {
        type: Boolean,
        default: false,
    },
    visibility: {
        type: String,
        default: "visible",
    },
});

const emit = defineEmits(["click"]);
</script>
<template>
    <button
        @click="isOnline && visibility !== 'hidden' && emit('click')"
        :disabled="!isOnline || visibility === 'hidden'"
        :class="[
            'polling-toggle cursor-pointer flex items-center space-x-2',
            !isManuallyPaused && isPolling
                ? 'active text-green-600 dark:text-green-500'
                : 'inactive text-red-500 dark:text-red-400',
        ]"
        title="Toggle live monitoring"
    >
        <!-- SVG Circle -->
        <svg width="12" height="12" viewBox="0 0 12 12" class="flex-shrink-0">
            <circle
                cx="6"
                cy="6"
                r="5"
                fill="currentColor"
                :class="isPollingLoading ? 'animate-pulse' : ''"
            />
        </svg>

        <!-- Text -->
        <span class="font-medium text-sm">
            Live Monitoring
            {{
                !isOnline
                    ? "Offline"
                    : visibility === "hidden"
                    ? "Hidden"
                    : isManuallyPaused
                    ? "Paused"
                    : !isManuallyPaused && isPolling
                    ? "Active"
                    : "Inactive"
            }}
        </span>
    </button>
</template>
