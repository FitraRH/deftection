<script setup>
import { defineProps, defineEmits } from "vue";
import { PopoverPanel } from "@headlessui/vue";
import FilterPopoverClear from "./FilterPopoverClear.vue";

defineProps({
    showClearButton: {
        type: Boolean,
        default: false,
    },
    selectedOptions: {
        type: Number,
        default: 0,
    },
});

const emit = defineEmits(["click-clear"]);
</script>
<template>
    <transition
        enter-active-class="transition duration-200 ease-out"
        enter-from-class="translate-y-1 opacity-0"
        enter-to-class="translate-y-0 opacity-100"
        leave-active-class="transition duration-150 ease-in"
        leave-from-class="translate-y-0 opacity-100"
        leave-to-class="translate-y-1 opacity-0"
    >
        <PopoverPanel
            class="z-10 absolute bg-white dark:bg-dark-700 ring-opacity-5 shadow-lg mt-2 rounded-lg ring-1 ring-black w-58"
        >
            <div class="relative space-y-4 p-4">
                <slot></slot>
            </div>
            <div
                v-if="showClearButton && selectedOptions > 0"
                class="border-gray-200 dark:border-dark-500 border-t"
            >
                <FilterPopoverClear
                    @clicked="emit('click-clear')"
                    label="Clear Filters"
                >
                </FilterPopoverClear>
            </div>
        </PopoverPanel>
    </transition>
</template>
