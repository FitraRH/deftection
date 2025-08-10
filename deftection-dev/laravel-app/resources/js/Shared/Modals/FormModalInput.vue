<script setup>
import { defineModel } from "vue";
defineProps({
    modelValue: String,
});

const model = defineModel("modelValue");
</script>

<template>
    <input
        v-model="model"
        class="dark:bg-dark-700 py-2 pl-4 border dark:border-dark-600 focus:border-transparent rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 w-full dark:text-dark-100 transition-all duration-200 placeholder-dark-400"
    />
</template>
