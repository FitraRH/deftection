<script setup>
import { ref } from "vue";

defineProps({
    loading: Boolean,
});

// State
const sensitivity = ref(0.5);
const options = ref({
    threatChecker: false,
});

// Emits to parent (optional if you want to lift state up later)
const emit = defineEmits(["runAnalysis"]);
</script>

<template>
    <div class="pt-3">
        <!-- Run Analysis Button -->
        <button
            @click="emit('runAnalysis', { sensitivity, ...options })"
            class="btn-base btn w-full gap-2 mt-2 this:primary bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
        >
            <template v-if="loading">
                <div
                    class="spinner spinner-base rounded-full animate-spin ghost-spinner border-white/30 border-r-white size-4 border-2"
                ></div>
                <span>Analyzing...</span>
            </template>
            <template v-else>
                <font-awesome-icon icon="fa-solid fa-magnifying-glass" />
                Run Analysis
            </template>
        </button>
    </div>
</template>
