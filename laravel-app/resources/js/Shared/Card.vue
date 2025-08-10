<template>
    <section
        class="relative bg-gradient-to-br from-slate-800 to-slate-900 shadow-indigo-900/30 shadow-lg border border-slate-700/50 rounded-xl overflow-hidden"
    >
        <header
            class="relative bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-700 p-6 rounded-t-xl"
        >
            <div class="z-10 relative">
                <h2 class="font-bold text-white text-xl tracking-wide">
                    Upload New Image
                </h2>
                <p class="mt-1 text-indigo-100/80 text-sm">
                    Upload a product image to detect defects using our
                    AI-powered analysis
                </p>
            </div>
        </header>

        <!-- Upload area -->
        <div class="p-6">
            <div
                class="group relative hover:bg-slate-700/30 p-8 border-2 border-slate-600 hover:border-indigo-400 border-dashed rounded-lg text-center hover:scale-[1.02] transition-all duration-300 ease-out cursor-pointer"
                :class="{
                    'border-emerald-400 bg-emerald-500/10 scale-[1.02]':
                        isDragging,
                    'border-emerald-500 bg-emerald-500/10': uploadSuccess,
                }"
                @dragover.prevent="handleDragOver"
                @dragleave="handleDragLeave"
                @drop.prevent="handleDrop"
            >
                <div
                    class="absolute inset-0 bg-gradient-to-r from-indigo-500/5 via-purple-500/5 to-indigo-500/5 opacity-0 group-hover:opacity-100 rounded-lg transition-opacity duration-300"
                ></div>

                <!-- Icon & gradient background -->
                <div
                    class="z-10 relative flex justify-center items-center bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg mx-auto mb-4 rounded-full w-16 h-16 group-hover:scale-110 transition-transform duration-300"
                >
                    <svg
                        class="w-8 h-8 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="2"
                            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                        ></path>
                    </svg>
                </div>

                <!-- Dynamic text -->
                <div class="z-10 relative space-y-2 mb-6">
                    <p class="font-medium text-white text-lg">
                        <span v-if="isDragging" class="text-emerald-400"
                            >Drop your image here!</span
                        >
                        <span v-else>Drag and drop your image</span>
                    </p>
                    <p class="text-slate-400 text-sm">~ OR ~</p>
                </div>

                <!-- Browse button -->
                <div class="z-10 relative">
                    <button
                        type="button"
                        @click="fileInput.click()"
                        class="inline-flex items-center bg-gradient-to-r from-indigo-600 hover:from-indigo-700 to-purple-600 hover:to-purple-700 shadow-lg hover:shadow-xl px-6 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-800 font-medium text-white transition-all hover:-translate-y-0.5 duration-300 transform"
                    >
                        <svg
                            class="mr-2 w-5 h-5"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z"
                            ></path>
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="m9 10 2 2 4-4"
                            ></path>
                        </svg>
                        Browse Image
                    </button>
                </div>
            </div>

            <!-- File info -->
            <div
                class="bg-slate-700/30 mt-4 p-4 border border-slate-600/50 rounded-lg"
            >
                <div class="flex justify-between items-center">
                    <div class="flex items-center space-x-4 text-sm">
                        <span class="flex items-center text-slate-300">
                            <svg
                                class="mr-1.5 w-4 h-4 text-indigo-400"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    stroke-width="2"
                                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                                ></path>
                            </svg>
                            JPEG, PNG, WebP
                        </span>
                        <span class="flex items-center text-slate-300">
                            <svg
                                class="mr-1.5 w-4 h-4 text-purple-400"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    stroke-width="2"
                                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                ></path>
                            </svg>
                            Max 5MB
                        </span>
                    </div>
                    <div class="font-medium text-indigo-400 text-xs">
                        Drag & Drop Ready
                    </div>
                </div>
            </div>
        </div>

        <!-- Hidden file input -->
        <input
            type="file"
            ref="fileInput"
            @change="handleFileSelect"
            class="hidden"
            accept="image/jpeg,image/png,image/webp"
        />
    </section>
</template>

<script setup>
import { ref } from "vue";

const isDragging = ref(false);
const fileInput = ref(null);

function handleDragOver() {
    isDragging.value = true;
}

function handleDragLeave() {
    isDragging.value = false;
}

function handleDrop(e) {
    isDragging.value = false;
    const files = e.dataTransfer.files;
    if (files.length) {
        handleFileSelect({ target: { files } });
    }
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (!files.length) return;
    const file = files[0];
    // You can add validation and preview logic here
    console.log("File selected:", file);
}
</script>
