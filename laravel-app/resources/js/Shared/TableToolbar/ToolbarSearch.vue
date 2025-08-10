<script setup>
import { defineProps, defineModel } from "vue";

const searchTerm = defineModel();

defineProps({
    placeholder: {
        type: String,
        default: "Search...",
    },
});
</script>
<template>
    <div class="input-root shrink-0">
        <div class="relative input-wrapper">
            <input
                v-model="searchTerm"
                class="peer rtl:pr-9 ltr:pl-9 border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:hover:border-dark-400 dark:focus:border-primary-500 dark:border-dark-450 ring-primary-500/50 focus:ring-3 h-8 text-xs form-input-base form-input"
                :placeholder="placeholder"
                type="text"
            />
            <div
                class="top-0 rtl:right-0 ltr:left-0 absolute flex justify-center items-center w-9 h-full text-gray-400 dark:text-dark-300 dark:peer-focus:text-primary-500 peer-focus:text-primary-600 transition-colors prefix"
            >
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
                        d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
                    ></path>
                </svg>
            </div>
        </div>
    </div>
</template>
