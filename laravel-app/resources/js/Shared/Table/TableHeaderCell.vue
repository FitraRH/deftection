<script setup>
defineProps({
    label: {
        type: String,
        default: "",
    },
    isSortable: {
        type: Boolean,
        default: false,
    },
    isActive: {
        type: Boolean,
        default: false,
    },
    sortDir: {
        type: String,
        default: "asc",
    },
});

const emit = defineEmits(["click"]);
</script>

<template>
    <th
        class="group/th table-th bg-gray-200 dark:bg-dark-800 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg font-semibold text-gray-800 dark:text-dark-100 uppercase"
        :class="{ 'cursor-pointer': isSortable }"
        @click="emit('click')"
    >
        <div :class="{ 'inline-flex gap-1 items-center': isSortable }">
            <slot>
                {{ label }}
            </slot>

            <!-- Arrow container - only show if sortable -->
            <span v-if="isSortable" class="flex items-center">
                <svg
                    width="24px"
                    height="24px"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    :class="[
                        'transition-all duration-300 ease-in-out transform',
                        {
                            // Active state - show current sort direction
                            'text-current': isActive,
                            // Inactive but sortable - hidden by default, visible on hover (always up arrow)
                            'text-transparent group-hover/th:text-gray-400 dark:group-hover/th:text-dark-400':
                                !isActive,
                            // Rotation based on sort direction (only when active)
                            'rotate-0': isActive && sortDir === 'asc',
                            'rotate-180': isActive && sortDir === 'desc',
                            // Force up arrow when inactive (hover state)
                            'group-hover/th:rotate-0': !isActive,
                        },
                    ]"
                >
                    <path
                        d="M12 6V18M12 6L7 11M12 6L17 11"
                        stroke="currentColor"
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    />
                </svg>
            </span>
        </div>
    </th>
</template>
