<script setup>
// Imports
import { ref, computed } from "vue";
import { useForm } from "@inertiajs/vue3";

import ImageSelector from "./Components/ImageSelector.vue";
import ImageTabs from "./Components/ImageTabs.vue";
import BatchImageSelector from "./Components/BatchImageSelector.vue";
import ProcessingModal from "./Components/Modals/ProcessingModal.vue";
import SuccessModal from "./Components/Modals/SuccessModal.vue";

import { successToast } from "@/utils/swal.js"; // Make sure this path is correct

// View Mode State (Single vs Batch)
const isBatchMode = ref(false);
const handleSwitchMode = (mode) => {
    isBatchMode.value = mode === "batch";
    detectionResult.value = null;
};

// Single Image Analysis State
const selectedFile = ref(null);
const detectionResult = ref(null);
const loading = ref(false);

// Inertia form for single image
const form = useForm({
    image: null,
    sensitivity: 50,
    threatChecker: false,
});

// Handle file selection for single image mode
const handleFileSelected = (file) => {
    selectedFile.value = file;
    form.image = file;

    // Reset result if file is cleared
    if (!file) {
        detectionResult.value = null;
    }
    console.log("File selected:", file);
};

// Original image preview URL for ImageTabs.vue
const originalUrl = computed(() => {
    return selectedFile.value ? URL.createObjectURL(selectedFile.value) : null;
});

// Run single image analysis (mocked async)
async function handleRunAnalysis(settings) {
    if (!selectedFile.value) return alert("Please select a file first.");

    loading.value = true;
    detectionResult.value = null;

    form.image = selectedFile.value;
    form.sensitivity = settings.sensitivity;
    form.threatChecker = settings.threatChecker;

    // form.post(route(""), {
    //     forceFormData: true,
    //     preserveScroll: true,
    //     onSuccess: () => {
    //         console.log("Analysis completed successfully.");
    //     },
    //     onError: (errors) => {
    //         console.error("Analysis failed:", errors);
    //         alert("Analysis failed. Please try again.");
    //     },
    // });

    // Mock simulation (3 seconds)
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Simulate if success
    successToast("Analysis completed successfully!");
    // toast success here

    // Mock data
    detectionResult.value = {
        decision: "GOOD",
        score: 0.8764,
        time: 0.39,
        defects: 0,
        resultImageUrl: originalUrl.value,
    };

    loading.value = false;

    console.log(detectionResult.value);
}

// Batch Analysis State
const isProcessing = ref(false);
const isSuccess = ref(false);
const processedCount = ref(0);
const totalCount = ref(0);

// Run batch analysis (simulated per file)
async function handleRunBatchAnalysis(files) {
    isProcessing.value = true;
    isSuccess.value = false;
    processedCount.value = 0;
    totalCount.value = files.length;

    // Simulate async image-by-image processing
    const processEach = async () => {
        for (let i = 0; i < files.length; i++) {
            form.image = files[i];
            form.sensitivity = 50;
            form.threatChecker = true;

            // await form.post(route(''), {
            //     forceFormData: true,
            //     preserveScroll: true,
            //     onError: (errors) => {
            //         console.error(`Error on image ${i + 1}:`, errors);
            //     },
            // })

            await new Promise((resolve) => setTimeout(resolve, 500)); // simulate 500ms/image
            processedCount.value = i + 1;
        }

        // Done
        // Set Delay 500ms to show the success modal
        setTimeout(() => {
            isProcessing.value = false;
            isSuccess.value = true;
        }, 500);
    };

    processEach();
}

// Retry batch analysis (reset everything)
const batchRef = ref(null);
function resetBatch() {
    isSuccess.value = false;
    processedCount.value = 0;
    totalCount.value = 0;

    // Call the exposed method in BatchImageSelector.vue
    batchRef.value?.clearFiles();
}

// URL of processed detection image for ImageTabs.vue
const detectionUrl = computed(() => {
    return detectionResult.value ? detectionResult.value.resultImageUrl : null;
});
</script>

<template>
    <div
        v-if="!isBatchMode"
        class="flex lg:flex-row flex-col gap-6 w-full h-fit"
    >
        <div class="flex flex-col gap-6 w-full lg:w-xl">
            <ImageSelector
                :loading="loading"
                @file-selected="handleFileSelected"
                @run-analysis="handleRunAnalysis"
                @switch-mode="handleSwitchMode"
            />
        </div>
        <div class="flex flex-col flex-1">
            <ImageTabs
                :original-url="originalUrl"
                :detection-url="detectionUrl"
                :show-steps="true"
                :has-run-analysis="!!detectionResult"
                :detectionResult="detectionResult"
            />
        </div>
    </div>

    <div v-else>
        <BatchImageSelector
            ref="batchRef"
            :loading="loading"
            @run-batch-analysis="handleRunBatchAnalysis"
            @switch-mode="handleSwitchMode"
        />
    </div>

    <!-- Modal for batch -->
    <ProcessingModal
        :visible="isProcessing"
        :current="processedCount"
        :total="totalCount"
    />

    <SuccessModal
        :visible="isSuccess"
        :total="totalCount"
        @retry="resetBatch"
    />
</template>
