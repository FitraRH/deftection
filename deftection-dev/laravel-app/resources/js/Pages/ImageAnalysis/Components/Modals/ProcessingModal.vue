<script setup>
import { computed } from "vue";
import {
    Dialog,
    DialogPanel,
    DialogTitle,
    TransitionRoot,
    TransitionChild,
} from "@headlessui/vue";

const props = defineProps({
    visible: Boolean,
    current: Number,
    total: Number,
});

defineEmits(["close"]);

const percent = computed(() => {
    if (!props.total) return 0;
    return Math.round((props.current / props.total) * 100);
});
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
                    class="relative w-full max-w-md rounded-lg bg-white transition-all duration-300 dark:bg-dark-800 p-6 shadow-xl"
                >
                    <DialogTitle
                        class="text-base font-medium text-gray-800 dark:text-white"
                    >
                        Running Batch Analysis
                    </DialogTitle>

                    <div class="mt-4 text-gray-700 dark:text-gray-300">
                        <p class="text-base">
                            Processing image <b>{{ current }}</b> of
                            <b>{{ total }}</b
                            >...
                        </p>

                        <!-- Progress Bar -->
                        <div
                            class="w-full h-2 mt-4 bg-gray-300 rounded-full overflow-hidden dark:bg-gray-700"
                        >
                            <div
                                class="h-full bg-green-500 transition-all duration-300"
                                :style="{ width: percent + '%' }"
                            ></div>
                        </div>
                    </div>
                </DialogPanel>
            </TransitionChild>
        </Dialog>
    </TransitionRoot>
</template>
