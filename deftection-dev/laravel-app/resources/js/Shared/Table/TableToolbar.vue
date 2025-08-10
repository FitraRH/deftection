<script setup>
import { defineProps } from "vue";

defineProps({
    title: {
        type: String,
        default: "Table Titles",
    },
    description: {
        type: String,
        default: "",
    },
});
</script>
<template>
    <div class="table-toolbar">
        <div class="flex justify-between items-center gap-4 transition-content">
            <div class="min-w-0">
                <h2
                    class="font-medium text-gray-800 dark:text-dark-50 text-xl truncate tracking-wide"
                >
                    {{ title }}
                </h2>
                <p>{{ description }}</p>
            </div>
        </div>
        <div
            class="flex flex-wrap justify-between items-center gap-x-4 gap-y-2 pt-4 pb-1 transition-content"
        >
            <!-- Left Section -->
            <div class="flex space-x-2 shrink-0">
                <slot name="left"></slot>
            </div>

            <!-- Right Section -->
            <slot name="right"></slot>
        </div>
    </div>
</template>
