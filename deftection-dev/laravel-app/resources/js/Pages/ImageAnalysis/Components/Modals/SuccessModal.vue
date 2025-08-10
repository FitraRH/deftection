<script setup>
import {
    Dialog,
    DialogPanel,
    DialogTitle,
    TransitionRoot,
    TransitionChild,
} from "@headlessui/vue";
import { route } from "ziggy-js";

const props = defineProps({
    visible: Boolean,
    total: Number,
});

const emit = defineEmits(["close", "retry"]);
</script>

<template>
    <TransitionRoot appear :show="visible" as="template">
        <Dialog
            as="div"
            class="fixed inset-0 z-[100] flex items-center justify-center overflow-hidden px-4 py-6 sm:px-5"
            @close="$emit('close')"
        >
            <!-- Overlay -->
            <TransitionChild
                as="template"
                enter="ease-out duration-300"
                enter-from="opacity-0"
                enter-to="opacity-100"
                leave="ease-in duration-200"
                leave-from="opacity-100"
                leave-to="opacity-0"
            >
                <div
                    class="absolute inset-0 bg-gray-900/50 backdrop-blur dark:bg-black/30"
                />
            </TransitionChild>

            <!-- Modal Panel -->
            <TransitionChild
                as="template"
                enter="ease-out duration-300"
                enter-from="opacity-0 scale-95"
                enter-to="opacity-100 scale-100"
                leave="ease-in duration-200"
                leave-from="opacity-100 scale-100"
                leave-to="opacity-0 scale-95"
            >
                <DialogPanel
                    class="relative w-full max-w-lg rounded-lg bg-white transition-all duration-300 dark:bg-dark-800 px-6 py-9 shadow-xl"
                >
                    <DialogTitle
                        class="text-base font-medium text-green-600 dark:text-green-400"
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
                    </DialogTitle>

                    <div class="flex flex-col justify-center items-center mt-4">
                        <h3
                            class="text-xl font-medium text-green-600 dark:text-green-400"
                        >
                            Batch Analysis Completed
                        </h3>
                        <p class="mx-auto mt-2 max-w-xs">
                            All <b>{{ total }}</b> images have been processed
                            successfully.
                        </p>

                        <!-- Actions -->
                        <div class="mt-6 flex flex-col gap-3">
                            <button
                                @click="$emit('retry')"
                                class="btn-base btn w-full gap-2 this:primary bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
                            >
                                <font-awesome-icon icon="fa-solid fa-recycle" />
                                Analyze Another Batch
                            </button>
                            <Link :href="route('scans.myscans')">
                                <button
                                    class="btn-base btn w-full gap-2 bg-gray-150 text-gray-900 hover:bg-gray-200 focus:bg-gray-200 active:bg-gray-200/80 dark:bg-surface-2 dark:text-dark-50 dark:hover:bg-surface-1 dark:focus:bg-surface-1 dark:active:bg-surface-1/90"
                                >
                                    <font-awesome-icon
                                        icon="fa-solid fa-folder-open"
                                    />
                                    View History Analysis
                                </button>
                            </Link>
                        </div>
                    </div>
                </DialogPanel>
            </TransitionChild>
        </Dialog>
    </TransitionRoot>
</template>

<style scoped>
.checkmark-circle {
    stroke-dasharray: 480px;
    stroke-dashoffset: 960px;
    animation: draw-circle 1s ease-out forwards;
}

.checkmark-tick {
    stroke-dasharray: 100px;
    stroke-dashoffset: 200px;
    animation: draw-check 0.8s ease-out 0.5s forwards;
}

@keyframes draw-circle {
    to {
        stroke-dashoffset: 0;
    }
}

@keyframes draw-check {
    to {
        stroke-dashoffset: 0;
    }
}
</style>
