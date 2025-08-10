<script setup>
import { ref, computed, watch } from "vue";
import { Popover } from "@headlessui/vue";
import FilterPopoverButton from "../Popover/FilterPopoverButton.vue";
import FilterPopoverPanel from "../Popover/FilterPopoverPanel.vue";

const props = defineProps({
    dateFrom: {
        type: String,
        default: "",
    },
    dateTo: {
        type: String,
        default: "",
    },
    disabled: {
        type: Boolean,
        default: false,
    },
});

const emit = defineEmits(["update:dateFrom", "update:dateTo", "change"]);

// Local reactive date values
const localDateFrom = ref(props.dateFrom || "");
const localDateTo = ref(props.dateTo || "");

// Watch for prop changes to update local values
watch(
    () => props.dateFrom,
    (newValue) => {
        localDateFrom.value = newValue || "";
    }
);

watch(
    () => props.dateTo,
    (newValue) => {
        localDateTo.value = newValue || "";
    }
);

// Computed property to check if any date filters are active
const hasDateFilters = computed(() => {
    return localDateFrom.value.length > 0 || localDateTo.value.length > 0;
});

// Date presets for quick selection
const datePresets = [
    {
        label: "Today",
        getDates: () => {
            const today = new Date().toISOString().split("T")[0];
            return { from: today, to: today };
        },
    },
    {
        label: "Yesterday",
        getDates: () => {
            const yesterday = new Date();
            yesterday.setDate(yesterday.getDate() - 1);
            const date = yesterday.toISOString().split("T")[0];
            return { from: date, to: date };
        },
    },
    {
        label: "Last 7 Days",
        getDates: () => {
            const today = new Date().toISOString().split("T")[0];
            const weekAgo = new Date();
            weekAgo.setDate(weekAgo.getDate() - 7);
            return { from: weekAgo.toISOString().split("T")[0], to: today };
        },
    },
    {
        label: "Last 30 Days",
        getDates: () => {
            const today = new Date().toISOString().split("T")[0];
            const monthAgo = new Date();
            monthAgo.setDate(monthAgo.getDate() - 30);
            return { from: monthAgo.toISOString().split("T")[0], to: today };
        },
    },
    {
        label: "This Month",
        getDates: () => {
            const now = new Date();
            const firstDay = new Date(now.getFullYear(), now.getMonth(), 1);
            const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0);
            return {
                from: firstDay.toISOString().split("T")[0],
                to: lastDay.toISOString().split("T")[0],
            };
        },
    },
    {
        label: "Last Month",
        getDates: () => {
            const now = new Date();
            const firstDay = new Date(now.getFullYear(), now.getMonth() - 1, 1);
            const lastDay = new Date(now.getFullYear(), now.getMonth(), 0);
            return {
                from: firstDay.toISOString().split("T")[0],
                to: lastDay.toISOString().split("T")[0],
            };
        },
    },
];

// Handle date input changes
const handleDateChange = () => {
    if (props.disabled) return;

    // Emit the updated values
    emit("update:dateFrom", localDateFrom.value);
    emit("update:dateTo", localDateTo.value);

    // Emit change event for parent to handle filtering
    emit("change");
};

// Apply date preset
const applyPreset = (preset) => {
    if (props.disabled) return;

    const dates = preset.getDates();
    localDateFrom.value = dates.from;
    localDateTo.value = dates.to;
    handleDateChange();
};

// Clear date filters
const clearDateFilters = () => {
    if (props.disabled) return;

    localDateFrom.value = "";
    localDateTo.value = "";
    handleDateChange();
};
</script>

<template>
    <Popover class="relative">
        <FilterPopoverButton
            label="Date Range"
            :selected-options="hasDateFilters ? 1 : 0"
            :disabled="disabled"
        >
            <template #icon>
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke-width="1.5"
                    stroke="currentColor"
                    class="w-4 h-4"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5a2.25 2.25 0 0 0 2.25-2.25m-18 0v-7.5A2.25 2.25 0 0 1 5.25 9h13.5a2.25 2.25 0 0 1 21 9v7.5m-9-13.5h.008v.008H12V8.25Z"
                    />
                </svg>
            </template>
        </FilterPopoverButton>

        <FilterPopoverPanel
            :show-clear-button="hasDateFilters"
            :selected-options="hasDateFilters ? 1 : 0"
            @click-clear="clearDateFilters"
        >
            <div class="space-y-4">
                <div>
                    <label
                        for="date-from"
                        class="block mb-1 font-medium text-gray-700 dark:text-dark-200 text-xs"
                    >
                        From Date
                    </label>
                    <input
                        id="date-from"
                        v-model="localDateFrom"
                        type="date"
                        :disabled="disabled"
                        :max="localDateTo || undefined"
                        class="bg-white dark:bg-dark-600 disabled:opacity-50 shadow-sm px-3 py-2 border border-gray-300 focus:border-primary-500 dark:border-dark-500 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 w-full text-gray-900 dark:text-dark-100 text-sm disabled:cursor-not-allowed"
                        @change="handleDateChange"
                    />
                </div>

                <div>
                    <label
                        for="date-to"
                        class="block mb-1 font-medium text-gray-700 dark:text-dark-200 text-xs"
                    >
                        To Date
                    </label>
                    <input
                        id="date-to"
                        v-model="localDateTo"
                        type="date"
                        :disabled="disabled"
                        :min="localDateFrom || undefined"
                        class="bg-white dark:bg-dark-600 disabled:opacity-50 shadow-sm px-3 py-2 border border-gray-300 focus:border-primary-500 dark:border-dark-500 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 w-full text-gray-900 dark:text-dark-100 text-sm disabled:cursor-not-allowed"
                        @change="handleDateChange"
                    />
                </div>

                <!-- Quick date presets -->
                <div class="pt-2 border-gray-200 dark:border-dark-500 border-t">
                    <p
                        class="mb-2 font-medium text-gray-700 dark:text-dark-200 text-xs"
                    >
                        Quick Select
                    </p>
                    <div class="gap-2 grid grid-cols-2">
                        <button
                            v-for="preset in datePresets"
                            :key="preset.label"
                            :disabled="disabled"
                            class="bg-white hover:bg-gray-50 dark:hover:bg-dark-500 dark:bg-dark-600 disabled:opacity-50 px-2 py-1.5 border border-gray-300 dark:border-dark-500 rounded text-gray-700 dark:text-dark-200 text-xs transition-colors disabled:cursor-not-allowed"
                            @click="applyPreset(preset)"
                        >
                            {{ preset.label }}
                        </button>
                    </div>
                </div>
            </div>
        </FilterPopoverPanel>
    </Popover>
</template>
