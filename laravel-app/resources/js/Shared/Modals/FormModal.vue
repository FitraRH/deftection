<script setup>
import { ref } from "vue";
import {
    Dialog,
    DialogPanel,
    DialogTitle,
    DialogDescription,
    TransitionRoot,
    TransitionChild,
} from "@headlessui/vue";

const emit = defineEmits(["close", "submit", "refresh"]);

const props = defineProps({
    isOpen: {
        type: Boolean,
        default: false,
    },
    isLoading: {
        type: Boolean,
        default: false,
    },
    title: {
        type: String,
        default: "Title of the Modal",
    },
    description: {
        type: String,
        default: "Fill in Modal Description.",
    },
    submitText: {
        type: String,
        default: "Submit",
    },
    loadingText: {
        type: String,
        default: "Loading...",
    },
    hasErrors: {
        type: Boolean,
        default: false,
    },
    focus: {
        type: String,
        default: null,
    },
    preventCloseOnOutsideClick: {
        type: Boolean,
        default: false,
    },
    showCloseButton: {
        type: Boolean,
        default: true,
    },
    showRefreshButton: {
        type: Boolean,
        default: false,
    },
    isRefreshing: {
        type: Boolean,
        default: false,
    },
});

// Template ref for focus management
const firstInputRef = ref(null);

// Handle modal close
const handleClose = () => {
    // Prevent closing if outside click is disabled and it's a backdrop click
    if (props.preventCloseOnOutsideClick) {
        return;
    }
    emit("close");
};

// Handle modal submit
const handleSubmit = () => {
    emit("submit");
};

// Handle close button click (always allow this)
const handleCloseButton = () => {
    emit("close");
};

// Handle refresh button click
const handleRefresh = () => {
    emit("refresh");
};
</script>

<template>
    <TransitionRoot appear :show="isOpen" as="div">
        <Dialog
            as="div"
            class="z-[100] fixed inset-0 flex flex-col justify-center items-center px-4 sm:px-5 py-6 overflow-hidden"
            @close="handleClose"
            :initial-focus="firstInputRef"
        >
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
                    class="absolute inset-0 bg-gray-900/50 dark:bg-black/30 backdrop-blur transition-opacity"
                />
            </TransitionChild>

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
                    class="relative flex flex-col bg-white dark:bg-dark-800 rounded-lg w-full max-w-lg overflow-hidden origin-top transition-all duration-300"
                >
                    <!-- Header -->
                    <div
                        class="space-y-4 bg-gray-200 dark:bg-dark-900 px-4 sm:px-5 py-3 rounded-t-lg"
                    >
                        <div class="flex justify-between items-center">
                            <div class="flex items-center space-x-3">
                                <DialogTitle
                                    as="h3"
                                    class="font-medium text-gray-800 dark:text-gray-100 text-base"
                                >
                                    {{ title }}
                                </DialogTitle>

                                <!-- Refresh Button -->
                                <button
                                    v-if="showRefreshButton"
                                    @click="handleRefresh"
                                    :disabled="isLoading || isRefreshing"
                                    class="flex justify-center items-center hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 rounded-md w-6 h-6 text-gray-500 hover:text-gray-700 dark:hover:text-gray-200 dark:text-gray-400 transition-colors disabled:cursor-not-allowed"
                                    title="Refresh data"
                                >
                                    <svg
                                        class="w-4 h-4 transition-transform duration-200"
                                        :class="{
                                            'animate-spin': isRefreshing,
                                        }"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                    >
                                        <path
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            stroke-width="2"
                                            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                                        />
                                    </svg>
                                </button>
                            </div>

                            <button
                                v-if="showCloseButton"
                                @click="handleCloseButton"
                                class="px-2 border-0 rounded-full btn btn-base btn-style"
                            >
                                <svg
                                    class="w-4 h-4"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path
                                        stroke-linecap="round"
                                        stroke-linejoin="round"
                                        stroke-width="2"
                                        d="M6 18L18 6M6 6l12 12"
                                    />
                                </svg>
                            </button>
                        </div>
                        <DialogDescription
                            class="text-gray-600 dark:text-gray-400 text-sm"
                        >
                            {{ description }}
                        </DialogDescription>
                    </div>

                    <!-- Form Content -->
                    <form
                        @submit.prevent="handleSubmit"
                        class="flex flex-col px-4 sm:px-5 py-4 overflow-y-auto"
                    >
                        <div class="space-y-5 mt-4">
                            <slot />
                        </div>

                        <!-- Actions -->
                        <div class="flex justify-end space-x-3 mt-6">
                            <button
                                type="button"
                                @click="handleCloseButton"
                                class="px-5 py-4 rounded-xl btn btn-base btn-style"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                :disabled="isLoading || hasErrors"
                                class="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 px-5 py-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-offset-2 font-medium text-white text-sm disabled:cursor-not-allowed btn btn-base btn-style"
                            >
                                <span
                                    v-if="isLoading"
                                    class="flex items-center"
                                >
                                    <svg
                                        class="mr-2 -ml-1 w-4 h-4 text-white animate-spin"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                    >
                                        <circle
                                            class="opacity-25"
                                            cx="12"
                                            cy="12"
                                            r="10"
                                            stroke="currentColor"
                                            stroke-width="4"
                                        ></circle>
                                        <path
                                            class="opacity-75"
                                            fill="currentColor"
                                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                        ></path>
                                    </svg>
                                    {{ loadingText }}
                                </span>
                                <span v-else>{{ submitText }}</span>
                            </button>
                        </div>
                    </form>
                </DialogPanel>
            </TransitionChild>
        </Dialog>
    </TransitionRoot>
</template>
