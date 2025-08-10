<script setup>
import DropdownItem from "./DropdownItem.vue";

import { defineProps, defineEmits } from "vue";

defineProps({
    label: {
        type: String,
        default: "Edit",
    },
    disabled: {
        type: Boolean,
        default: false,
    },
    variant: {
        type: String,
        default: "default",
        // 'default' | 'danger' | 'success' | 'warning' | 'info'
    },
    customClasses: {
        type: String,
        default: "",
    },
});

const emit = defineEmits(["click"]);
</script>
<template>
    <DropdownItem
        as="button"
        @click="!disabled && emit('click')"
        :label="label"
        :disabled="disabled"
        :variant="variant"
        :custom-classes="customClasses"
    >
        <template #icon>
            <svg
                class="mr-3 w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
            >
                <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002 2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                />
            </svg>
        </template>
    </DropdownItem>
</template>
