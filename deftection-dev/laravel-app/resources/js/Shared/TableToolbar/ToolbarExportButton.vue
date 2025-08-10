<script setup>
import ToolbarButton from "./ToolbarButton.vue";
import { defineProps, defineEmits } from "vue";

const emit = defineEmits(["export"]);

defineProps({
    label: {
        type: String,
        default: "Export",
    },
    disabled: {
        type: Boolean,
        default: false,
    },
    title: {
        type: String,
        default: "",
    },
});
</script>
<template>
    <ToolbarButton
        :label="label"
        @click="!disabled && emit('export')"
        :disabled="disabled"
        :title="title"
    >
        <template #icon>
            <svg
                stroke="currentColor"
                fill="none"
                stroke-width="2"
                viewBox="0 0 24 24"
                stroke-linecap="round"
                stroke-linejoin="round"
                class="w-4 h-4"
                height="1em"
                width="1em"
                xmlns="http://www.w3.org/2000/svg"
            >
                <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2"></path>
                <path d="M7 9l5 -5l5 5"></path>
                <path d="M12 4l0 12"></path>
            </svg>
        </template>
    </ToolbarButton>
</template>
