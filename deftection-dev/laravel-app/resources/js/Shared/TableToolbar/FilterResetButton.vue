<script setup>
import { defineProps, defineEmits } from "vue";

defineProps({
    label: {
        type: String,
        default: "Reset",
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
const emit = defineEmits(["click"]);
</script>
<template>
    <button
        @click="!disabled && emit('click')"
        :disabled="disabled"
        :title="title"
        class="bg-gray-150 hover:bg-gray-200 focus:bg-gray-200 active:bg-gray-200/80 dark:active:bg-surface-1/90 dark:bg-surface-2 dark:hover:bg-surface-1 dark:focus:bg-surface-1 px-2.5 h-8 text-gray-900 dark:text-dark-50 text-xs whitespace-nowrap btn-base btn"
        type="button"
    >
        {{ label }}
    </button>
</template>
