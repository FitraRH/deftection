<script setup>
import { PopoverButton } from "@headlessui/vue";
import { defineProps } from "vue";

defineProps({
    label: {
        type: String,
        default: "Filter",
    },
    selectedOptions: {
        type: Number,
        default: 0,
    },
});
</script>
<template>
    <PopoverButton
        class="flex items-center gap-2 hover:bg-gray-300/20 focus:bg-gray-300/20 active:bg-gray-300/25 dark:hover:bg-dark-300/20 dark:focus:bg-dark-300/20 dark:active:bg-dark-300/25 px-2.5 border border-gray-300 dark:border-dark-450 border-dashed h-8 text-gray-900 dark:text-dark-50 text-xs whitespace-nowrap btn-base btn"
    >
        <slot name="icon">
            <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke-width="1.5"
                stroke="currentColor"
                aria-hidden="true"
                class="w-4 h-4"
            >
                <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75"
                ></path>
            </svg>
        </slot>

        <span>{{ label }}</span>
        <span
            v-if="selectedOptions > 0"
            class="bg-primary-500 ml-1 px-2 rounded-full text-white text-xs"
        >
            {{ selectedOptions }}
        </span>
    </PopoverButton>
</template>
