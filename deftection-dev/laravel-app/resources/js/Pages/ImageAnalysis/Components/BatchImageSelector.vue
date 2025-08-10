<script setup>
import { ref } from "vue";
import vueFilePond from "vue-filepond";

import FilePondPluginImagePreview from "filepond-plugin-image-preview";
import FilePondPluginFileValidateType from "filepond-plugin-file-validate-type";

import "filepond/dist/filepond.min.css";
import "filepond-plugin-image-preview/dist/filepond-plugin-image-preview.css";

const FilePond = vueFilePond(
    FilePondPluginImagePreview,
    FilePondPluginFileValidateType
);

const emit = defineEmits(["run-batch-analysis", "switch-mode"]);

const files = ref([]);

const pond = ref(null);
function removeAllFiles() {
    if (pond.value) {
        pond.value.removeFiles();
    }

    files.value = [];
}

defineProps({
    loading: Boolean,
});

defineExpose({
    clearFiles: () => {
        pond.value?.removeFiles();
        files.value = [];
    },
});
</script>

<template>
    <div
        class="flex flex-col p-6 border border-gray-200 dark:border-dark-500 rounded-lg"
    >
        <div class="flex flex-col gap-2">
            <div class="flex justify-between items-center mb-4">
                <div class="flex flex-col gap-1">
                    <h2
                        class="text-sm-plus font-medium tracking-wide text-gray-800 dark:text-dark-100"
                    >
                        Batch Image Selection
                        <span
                            class="ml-1 text-xs font-normal text-gray-500 dark:text-dark-300"
                        >
                            (Max 10 images)
                        </span>
                    </h2>
                    <p
                        class="text-xs font-normal text-gray-500 dark:text-dark-300"
                    >
                        Accepted formats: PNG, JPG, JPEG
                    </p>
                </div>
                <div :class="files.length > 0 ? '' : 'hidden'">
                    <button
                        @click="removeAllFiles"
                        class="btn-base btn gap-2 this:error bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
                        type="button"
                    >
                        <font-awesome-icon icon="fa-solid fa-trash-can" />
                        Remove
                    </button>
                </div>
            </div>

            <FilePond
                ref="pond"
                :name="image"
                accepted-file-types="image/png, image/jpeg, image/jpg"
                :allow-multiple="true"
                :max-files="10"
                :files="files"
                :required="true"
                @updatefiles="files = $event"
                label-idle='Drag & Drop or <span class="filepond--label-action">Browse</span>'
            />

            <div class="flex justify-between items-center mb-3">
                <p class="text-sm text-gray-600 dark:text-dark-300">
                    Want to analyze just a single image?
                    <button
                        @click="emit('switch-mode', 'single')"
                        class="ml-1 text-primary-600 dark:text-primary-400 underline hover:no-underline transition cursor-pointer"
                    >
                        Switch to single mode
                    </button>
                </p>

                <p class="text-sm text-gray-600 dark:text-dark-300 text-right">
                    {{ files.length }} / 10 files selected
                </p>
            </div>

            <button
                :disabled="files.length === 0 || loading"
                @click="emit('run-batch-analysis', files)"
                class="btn-base btn gap-2 this:primary bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
                type="button"
            >
                <template v-if="loading">
                    <div
                        class="spinner spinner-base rounded-full animate-spin ghost-spinner border-white/30 border-r-white size-4 border-2"
                    ></div>
                    <span>Analyzing...</span>
                </template>
                <template v-else>
                    <font-awesome-icon icon="fa-solid fa-magnifying-glass" />
                    Run Batch Analysis
                </template>
            </button>
        </div>
    </div>
</template>

<style>
.dark .filepond--panel-root {
    background-color: transparent;
    border: 1px solid #2a2c32;
}

.filepond--panel-root {
    background-color: transparent;
    border: 1px solid #e2e8f0;
}

.filepond--item {
    width: calc(33.3333% - 0.5em);
}

.filepond--credits {
    display: none;
}
</style>
