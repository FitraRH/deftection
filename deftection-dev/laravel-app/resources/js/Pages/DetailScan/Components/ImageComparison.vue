<script setup>
// --- Props ---
// This component receives the necessary data to display the comparison.
const props = defineProps({
    comparisonData: {
        type: Object,
        required: true,
        // The default function provides mock data for the "Defect" layout.
        default: () => ({
            status: "defect",
            originalImageUrl:
                "https://placehold.co/600x400/FFFFFF/000000?text=Original+Image",
            analyzedImageUrl: "https://i.imgur.com/pUq3f7A.png", // Using a real example for the bounding box
        }),
    },
});
</script>

<template>
    <div
        class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
    >
        <h3
            class="flex items-center mb-6 gap-4 text-lg font-semibold text-gray-900 dark:text-dark-50"
        >
            <font-awesome-icon icon="fa-solid fa-images" />
            Image Comparison
        </h3>

        <div class="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <!-- Left Card: Original Image -->
            <div class="p-4">
                <h4
                    class="mb-4 font-medium tracking-wide text-gray-800 dark:text-dark-100"
                >
                    Original Image
                </h4>
                <div
                    class="flex items-center justify-center p-2 bg-gray-100 rounded-md dark:bg-dark-700 aspect-video"
                >
                    <img
                        :src="comparisonData.originalImageUrl"
                        alt="Original analysis image"
                        class="object-contain max-w-full max-h-full"
                        onerror="this.onerror=null;this.src='https://placehold.co/600x400/CCCCCC/FFFFFF?text=Image+Not+Found';"
                    />
                </div>
            </div>

            <!-- Right Card: Detection Result -->
            <div class="p-4">
                <h4
                    class="mb-4 font-medium tracking-wide text-gray-800 dark:text-dark-100"
                >
                    Detection Result with Bounding Boxes
                </h4>

                <!-- State for Defected Products -->
                <div
                    v-if="comparisonData.status === 'defect'"
                    class="flex items-center justify-center p-2 bg-gray-100 rounded-md dark:bg-dark-700 aspect-video"
                >
                    <img
                        :src="comparisonData.analyzedImageUrl"
                        alt="Analyzed image with bounding boxes"
                        class="object-contain max-w-full max-h-full"
                        onerror="this.onerror=null;this.src='https://placehold.co/600x400/CCCCCC/FFFFFF?text=Image+With+Bounding+Box';"
                    />
                </div>

                <!-- State for Good Products -->
                <div
                    v-else
                    class="flex flex-col items-center justify-center text-center text-success dark:text-success-lighter bg-gray-100 rounded-md dark:bg-dark-700 aspect-video"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 154 154"
                        fill="none"
                        data-animated-tick="true"
                        class="stroke-current mx-auto size-24 shrink-0 text-success"
                    >
                        <path
                            d="M77 141C112.346 141 141 112.346 141 77C141 41.6538 112.346 13 77 13C41.6538 13 13 41.6538 13 77C13 112.346 41.6538 141 77 141Z"
                            stroke-width="10"
                            class="checkmark-circle"
                            style="
                                stroke-dasharray: 480px, 480px;
                                stroke-dashoffset: 960px;
                            "
                        ></path>
                        <path
                            d="M46 80.2444L63.9556 98.1111L107.067 55"
                            stroke-width="10"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            class="checkmark-tick"
                            style="
                                stroke-dasharray: 100px, 100px;
                                stroke-dashoffset: 200px;
                            "
                        ></path>
                    </svg>
                    <h5 class="mt-4 text-lg font-semibold">
                        No Defects Detected
                    </h5>
                    <p class="mt-1 text-sm">Product meets quality standards.</p>
                </div>
            </div>
        </div>
    </div>
</template>
