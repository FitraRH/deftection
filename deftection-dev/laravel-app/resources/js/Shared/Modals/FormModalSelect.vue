<script setup>
import { defineModel } from "vue";

defineProps({
    modelValue: String,
    options: {
        type: Array,
        required: true,
        default: () => [],
    },
    placeholder: {
        type: String,
        default: "Select an option",
    },
    disabled: {
        type: Boolean,
        default: false,
    },
});

const model = defineModel("modelValue");
</script>

<template>
    <div class="relative">
        <select
            v-model="model"
            :disabled="disabled"
            class="dark:bg-dark-700 disabled:opacity-50 py-2 pr-10 pl-4 border dark:border-dark-600 focus:border-transparent rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 w-full dark:text-dark-100 transition-all duration-200 appearance-none cursor-pointer disabled:cursor-not-allowed placeholder-dark-400"
        >
            <option value="" disabled>{{ placeholder }}</option>
            <option
                v-for="option in options"
                :key="option.value"
                :value="option.value"
            >
                {{ option.label }}
            </option>
        </select>

        <!-- Custom dropdown arrow -->
        <div
            class="right-0 absolute inset-y-0 flex items-center pr-3 pointer-events-none"
            :class="{
                'opacity-50': disabled,
            }"
        >
            <svg
                class="w-5 h-5 text-gray-400 dark:text-dark-300"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                aria-hidden="true"
            >
                <path
                    fill-rule="evenodd"
                    d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                    clip-rule="evenodd"
                />
            </svg>
        </div>
    </div>
</template>
