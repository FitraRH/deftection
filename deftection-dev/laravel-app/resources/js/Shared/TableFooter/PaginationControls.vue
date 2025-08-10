<script setup>
import { computed } from "vue";

const { meta, currentPage } = defineProps({
    meta: {
        type: Object,
        required: true,
    },
    currentPage: {
        type: [String, Number],
        required: true,
    },
});

const emit = defineEmits(["goToPage"]);

const displayPages = computed(() => {
    // Capture values immediately to prevent reactive changes during computation
    const totalPages = meta.last_page;
    const current = parseInt(currentPage);

    // Convert to numbers to prevent type issues
    const totalPagesNum = parseInt(totalPages);
    const currentNum = parseInt(current);
    const pages = [];

    if (
        totalPagesNum <= 4 ||
        (currentNum == 3 && totalPagesNum == 5) ||
        (currentNum == 4 && totalPagesNum == 7)
    ) {
        // Show all pages
        for (let i = 1; i <= totalPagesNum; i++) {
            pages.push(i);
        }
        return pages;
    }

    // Always show first page
    pages.push(1);

    // Add left gap if needed
    if (currentNum > 4) {
        pages.push("...");
    }

    // Middle range: currentNum -2 to currentNum +2
    const startLoop = currentNum - 2;
    const endLoop = currentNum + 2;
    for (let i = startLoop; i <= endLoop; i++) {
        if (i > 1 && i < totalPagesNum) {
            pages.push(i);
        }
    }

    // Add right gap if needed
    if (currentNum < totalPagesNum - 3) {
        pages.push("...");
    }

    // Always show last page
    if (totalPagesNum > 1) {
        pages.push(totalPagesNum);
    }

    return pages;
});

const handleSelectPage = (page) => {
    if (page === "..." || page < 1 || page > meta.last_page) return;
    emit("goToPage", page);
};
</script>

<template>
    <div class="max-w-full overflow-x-auto pagination hide-scrollbar">
        <!-- Previous Page Button -->
        <button
            type="button"
            @click="handleSelectPage(Number(currentPage) - 1)"
            class="pagination-control"
            :class="{
                'opacity-60 cursor-not-allowed': currentPage == 1,
                'hover:bg-gray-300 dark:hover:bg-surface-1 cursor-pointer':
                    currentPage != 1,
            }"
            :disabled="currentPage == 1"
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
                    d="M15 19l-7-7 7-7"
                />
            </svg>
        </button>

        <template v-for="(label, index) in displayPages" :key="index">
            <span v-if="isNaN(label)">{{ label }}</span>
            <button
                v-else
                type="button"
                @click="handleSelectPage(label)"
                class="pagination-control"
                :class="{
                    'bg-blue-600 text-white': label == currentPage,
                    'hover:bg-gray-300 dark:hover:bg-surface-1 cursor-pointer':
                        label != currentPage,
                }"
            >
                {{ label }}
            </button>
        </template>

        <!-- Next Page Button -->
        <button
            type="button"
            @click="handleSelectPage(Number(currentPage) + 1)"
            class="pagination-control"
            :class="{
                'opacity-60 cursor-not-allowed ': currentPage == meta.last_page,
                'hover:bg-gray-300 dark:hover:bg-surface-1 cursor-pointer':
                    currentPage != meta.last_page,
            }"
            :disabled="currentPage == meta.last_page"
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
                    d="M9 5l7 7-7 7"
                />
            </svg>
        </button>
    </div>
</template>
