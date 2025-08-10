<script setup>
defineProps({ for: String });
</script>

<template>
    <label
        :for="for"
        class="block mb-2 font-medium text-gray-700 dark:text-gray-300 text-sm"
    >
        <slot />
    </label>
</template>
