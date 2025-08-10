<script setup>
import { MenuItem } from "@headlessui/vue";
import { computed } from "vue";

const props = defineProps({
    as: {
        type: String,
        default: "button",
    },
    label: {
        type: String,
        required: true,
    },
    disabled: {
        type: Boolean,
        default: false,
    },
    variant: {
        type: String,
        default: "default",
        validator: (value) =>
            ["default", "danger", "success", "warning", "info"].includes(value),
    },
    href: {
        type: String,
        default: null,
    },
    openInNewTab: {
        type: Boolean,
        default: false,
    },
    customClasses: {
        type: String,
        default: "",
    },
});

const emit = defineEmits(["click"]);

const handleClick = () => {
    if (!props.disabled) {
        emit("click");
    }
};

const hasHref = computed(() => {
    return props.href !== null && (props.as == "a" || props.as == "link");
});
// Compute variant-based classes
const variantClasses = computed(() => {
    const variants = {
        default: {
            text: "text-gray-700 dark:text-gray-300",
            activeText: "text-gray-800 dark:text-gray-100",
            activeBg: "bg-gray-100 dark:bg-dark-700",
        },
        danger: {
            text: "text-red-600 dark:text-red-400",
            activeText: "text-red-800 dark:text-red-400",
            activeBg: "bg-red-50 dark:bg-red-900/20",
        },
        success: {
            text: "text-green-600 dark:text-green-400",
            activeText: "text-green-800 dark:text-green-400",
            activeBg: "bg-green-50 dark:bg-green-900/20",
        },
        warning: {
            text: "text-yellow-600 dark:text-yellow-400",
            activeText: "text-yellow-800 dark:text-yellow-400",
            activeBg: "bg-yellow-50 dark:bg-yellow-900/20",
        },
        info: {
            text: "text-blue-600 dark:text-blue-400",
            activeText: "text-blue-800 dark:text-blue-400",
            activeBg: "bg-blue-50 dark:bg-blue-900/20",
        },
    };

    return variants[props.variant] || variants.default;
});
</script>

<template>
    <MenuItem as="div" v-slot="{ active }">
        <component
            :is="as"
            @click="handleClick()"
            :disabled="disabled"
            :href="hasHref ? href : null"
            :target="openInNewTab ? '_blank' : '_self'"
            :class="[
                'flex items-center w-full px-3 py-2 text-left text-sm h-9 tracking-wide outline-none transition-colors',
                {
                    // Active state with variant colors
                    [variantClasses.activeBg]: active && !disabled,
                    [variantClasses.activeText]: active && !disabled,

                    // Normal state with variant colors
                    [variantClasses.text]: !active && !disabled,

                    // Disabled state
                    'text-gray-400 dark:text-dark-400 cursor-not-allowed':
                        disabled,
                    'cursor-pointer': !disabled,
                },
                customClasses,
            ]"
        >
            <slot name="icon" />
            {{ label }}
        </component>
    </MenuItem>
</template>
