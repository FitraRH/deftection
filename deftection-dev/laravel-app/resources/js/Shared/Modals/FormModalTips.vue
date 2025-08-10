<script setup>
import { computed, defineProps } from "vue";
// Props
const props = defineProps({
    tips: {
        type: Array,
        required: true,
        validator: (tips) => {
            // Each tip should be either a string or an object with text and condition
            return tips.every(
                (tip) =>
                    typeof tip === "string" ||
                    (typeof tip === "object" &&
                        tip.text &&
                        typeof tip.text === "string")
            );
        },
    },
    title: {
        type: String,
        default: "Tips:",
    },
    iconColor: {
        type: String,
        default: "text-blue-500",
    },
    bgColor: {
        type: String,
        default: "bg-blue-50 dark:bg-blue-900/20",
    },
    titleColor: {
        type: String,
        default: "text-blue-700 dark:text-blue-300",
    },
    textColor: {
        type: String,
        default: "text-blue-600 dark:text-blue-400",
    },
    secondaryTextColor: {
        type: String,
        default: "text-gray-500 dark:text-gray-400",
    },
});

// Computed property to filter visible tips
const visibleTips = computed(() => {
    return props.tips.filter((tip) => {
        if (typeof tip === "string") {
            return true; // Always show string tips
        }

        // For object tips, check if condition is met
        if (tip.condition !== undefined) {
            return tip.condition;
        }

        return true; // Show by default if no condition
    });
});
</script>

<template>
    <div
        v-if="visibleTips.length > 0"
        class="p-3 rounded-lg text-xs"
        :class="[bgColor, secondaryTextColor]"
    >
        <div class="flex items-start">
            <svg
                class="mt-0.5 mr-2 w-4 h-4"
                :class="iconColor"
                fill="currentColor"
                viewBox="0 0 20 20"
            >
                <path
                    fill-rule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                    clip-rule="evenodd"
                />
            </svg>
            <div>
                <p class="font-medium" :class="titleColor">
                    {{ title }}
                </p>
                <ul class="space-y-1 mt-1" :class="textColor">
                    <li v-for="(tip, index) in visibleTips" :key="index">
                        <span v-if="typeof tip === 'string'">
                            • {{ tip }}
                        </span>
                        <span v-else> • {{ tip.text }} </span>
                    </li>
                </ul>
            </div>
        </div>
    </div>
</template>
