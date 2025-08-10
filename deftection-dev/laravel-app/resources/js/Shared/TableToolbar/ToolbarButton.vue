<script setup>
import { defineProps, defineEmits } from "vue";

const emit = defineEmits(["click"]);
defineProps({
    label: {
        type: String,
        default: "",
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
    <button
        @click="!disabled && emit('click')"
        class="hover:bg-gray-300/20 focus:bg-gray-300/20 active:bg-gray-300/25 dark:hover:bg-dark-300/20 dark:focus:bg-dark-300/20 dark:active:bg-dark-300/25 px-3 border border-gray-300 dark:border-dark-450 rounded-md h-8 text-gray-900 dark:text-dark-50 text-xs btn-base btn"
        :class="label ? 'space-x-2' : ''"
        type="button"
        :disabled="disabled"
        :title="title"
    >
        <slot name="icon"></slot>
        <span>{{ label }}</span>
    </button>
</template>
