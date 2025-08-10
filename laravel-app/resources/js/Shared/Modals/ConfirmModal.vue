<script setup>
import {
    Dialog,
    DialogPanel,
    DialogTitle,
    TransitionRoot,
    TransitionChild,
} from "@headlessui/vue";

const props = defineProps({
    isOpen: {
        type: Boolean,
        default: false,
    },
    title: {
        type: String,
        default: "Confirm Action",
    },
    message: {
        type: String,
        default: "Are you sure you want to proceed?",
    },
    type: {
        type: String,
        default: "info", // 'success', 'danger', 'warning', 'info'
        validator: (value) =>
            ["success", "danger", "warning", "info"].includes(value),
    },
    confirmText: {
        type: String,
        default: "Confirm",
    },
    cancelText: {
        type: String,
        default: "Cancel",
    },
    showCancel: {
        type: Boolean,
        default: true,
    },
});

const emit = defineEmits(["confirm", "cancel", "close"]);

const handleConfirm = () => {
    emit("confirm");
};

const handleCancel = () => {
    emit("cancel");
};

const handleClose = () => {
    emit("close");
};

const getConfirmButtonClass = () => {
    const classes = {
        success: "bg-green-600 hover:bg-green-700 focus:ring-green-500",
        danger: "bg-red-600 hover:bg-red-700 focus:ring-red-500",
        warning: "bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-500",
        info: "bg-blue-600 hover:bg-blue-700 focus:ring-blue-500",
    };
    return classes[props.type] || classes.info;
};
</script>

<!-- ConfirmModal.vue -->
<template>
    <TransitionRoot appear :show="isOpen" as="template">
        <Dialog
            as="div"
            class="z-[100] fixed inset-0 flex flex-col justify-center items-center px-4 sm:px-5 py-6 overflow-hidden"
            @close="handleClose"
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
                    class="absolute inset-0 bg-dark-900/50 dark:bg-black/40 transition-opacity"
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
                    class="relative flex flex-col bg-white dark:bg-dark-800 px-4 sm:px-5 py-10 rounded-lg max-w-md overflow-y-auto text-center transition-all duration-300 scrollbar-sm"
                >
                    <!-- Icon -->
                    <div
                        class="flex justify-center items-center mx-auto w-28 h-28"
                    >
                        <!-- <font-awesome-icon
                            v-if="type === 'success'"
                            icon="fa-regular fa-circle-check"
                            class="w-28 h-28 text-green-500"
                        /> -->
                        <svg
                            v-if="type === 'success'"
                            class="w-28 h-28 text-green-500"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                        </svg>

                        <svg
                            v-else-if="type === 'danger'"
                            class="w-28 h-28 text-red-500"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z"
                            />
                        </svg>

                        <svg
                            v-else-if="type === 'warning'"
                            class="w-28 h-28 text-yellow-500"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                        </svg>

                        <svg
                            v-else
                            class="w-28 h-28 text-blue-500"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                stroke-width="2"
                                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                        </svg>
                    </div>

                    <!-- Content -->
                    <div class="mt-4">
                        <DialogTitle
                            as="h3"
                            class="font-semibold text-gray-800 dark:text-gray-100 text-2xl"
                        >
                            {{ title }}
                        </DialogTitle>
                        <p class="mt-2 text-gray-600 dark:text-gray-300">
                            {{ message }}
                        </p>

                        <!-- Actions -->
                        <div
                            class="flex sm:flex-row flex-col sm:justify-center gap-3 mt-6"
                        >
                            <button
                                v-if="showCancel"
                                @click="handleCancel"
                                class="p-5 text-sm btn btn-base btn-style"
                            >
                                {{ cancelText }}
                            </button>

                            <button
                                @click="handleConfirm"
                                :class="[
                                    'btn btn-base btn-style p-5 text-sm text-white border-red-600',
                                    getConfirmButtonClass(),
                                ]"
                            >
                                {{ confirmText }}
                            </button>
                        </div>
                    </div>
                </DialogPanel>
            </TransitionChild>
        </Dialog>
    </TransitionRoot>
</template>
