<script setup>
import { ref, watch } from "vue";
import AnalysisControls from "./AnalysisControls.vue";
import vueFilePond from "vue-filepond";

import FilePondPluginImagePreview from "filepond-plugin-image-preview";
import FilePondPluginFileValidateType from "filepond-plugin-file-validate-type";

import "filepond/dist/filepond.min.css";
import "filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css";

const FilePond = vueFilePond(
    FilePondPluginImagePreview,
    FilePondPluginFileValidateType
);

const emit = defineEmits(["file-selected", "run-analysis", "switch-mode"]);

const files = ref([]);
watch(files, (newFiles) => {
    const file = newFiles[0]?.file || null;
    emit("file-selected", file);
});

defineProps({
    loading: Boolean,
});
</script>

<template>
    <div
        class="flex flex-col p-6 border border-gray-200 dark:border-dark-500 rounded-lg gap-1"
    >
        <div class="flex flex-col">
            <h2
                class="text-sm-plus font-medium tracking-wide text-gray-800 dark:text-dark-100"
            >
                Image Selection
            </h2>
            <p class="mt-1 mb-3 text-sm text-gray-600 dark:text-dark-300">
                Accepted formats: PNG, JPG, JPEG
            </p>

            <FilePond
                :name="image"
                accepted-file-types="image/png, image/jpeg, image/jpg"
                :allow-multiple="false"
                :files="files"
                :required="true"
                @updatefiles="files = $event"
                label-idle='Drag & Drop or <span class="filepond--label-action">Browse</span>'
            />

            <p class="mt-1 text-sm text-gray-600 dark:text-dark-300">
                Want to analyze multiple images at once?
                <button
                    @click="emit('switch-mode', 'batch')"
                    class="ml-1 text-primary-600 dark:text-primary-400 underline hover:no-underline transition cursor-pointer"
                >
                    Switch to batch mode
                </button>
            </p>

            <AnalysisControls
                :loading="loading"
                @runAnalysis="(settings) => emit('run-analysis', settings)"
            />
        </div>
    </div>
</template>
