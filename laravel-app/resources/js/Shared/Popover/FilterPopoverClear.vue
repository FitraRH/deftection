<script setup>
defineProps({
    label: {
        type: String,
        default: "Clear",
    },
});

const emit = defineEmits(["clicked"]);
</script>
<template>
    <button
        @click="emit('clicked')"
        class="bg-gray-150 hover:bg-gray-200 focus:bg-gray-200 active:bg-gray-200/80 dark:active:bg-surface-1/90 dark:bg-surface-2 dark:hover:bg-surface-1 dark:focus:bg-surface-1 rounded-t-none w-full text-gray-900 dark:text-dark-50 btn-base btn shrink-0"
    >
        {{ label }}
    </button>
</template>
