<script setup>
import ToolbarButton from "../TableToolbar/ToolbarButton.vue";
import { defineProps, defineEmits, ref } from "vue";

defineProps({
    isOnline: {
        type: Boolean,
        default: false,
    },
    isPollingLoading: {
        type: Boolean,
        default: false,
    },
});

const emit = defineEmits(["click"]);
const isRotating = ref(false);

const handleClick = () => {
    if (!isRotating.value) {
        isRotating.value = true;
        emit("click");

        // Reset rotation after animation completes
        setTimeout(() => {
            isRotating.value = false;
        }, 500); // Match this with the CSS animation duration
    }
};
</script>

<template>
    <ToolbarButton
        @click="isOnline && !isPollingLoading && handleClick()"
        :disabled="!isOnline || isPollingLoading"
        title="Refresh data"
        label=""
    >
        <template #icon>
            <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                :class="{
                    'animate-spin': isPollingLoading,
                    'rotate-on-click': isRotating && !isPollingLoading,
                }"
            >
                <polyline points="23 4 23 10 17 10" />
                <polyline points="1 20 1 14 7 14" />
                <path
                    d="m3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"
                />
            </svg>
        </template>
    </ToolbarButton>
</template>

<style scoped>
.rotate-on-click {
    animation: rotateOnce 0.5s ease-in-out;
}

@keyframes rotateOnce {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
</style>
