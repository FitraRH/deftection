<script setup>
import { computed } from "vue";
import PerPageSelector from "../TableFooter/PerPageSelector.vue";
import PaginationControls from "../TableFooter/PaginationControls.vue";
import PageInfo from "../TableFooter/PageInfo.vue";

// Props - now expects meta object instead of pagination
const {
    meta,
    currentFilters,
    perPageOptions = [5, 10, 25, 50],
} = defineProps({
    meta: {
        type: Object,
        required: true,
    },
    currentFilters: {
        type: Object,
        required: true,
    },
    perPageOptions: {
        type: Array,
        default: () => [5, 10, 25, 50],
    },
});

// Emits for parent communication
const emit = defineEmits(["goToPage", "changePerPage"]);

// Get current values from filters
const currentPage = computed(() => currentFilters.page || 1);
const perPage = computed(() => currentFilters.perPage || 10);

const handleGoToPage = (page) => {
    emit("goToPage", page);
};

const handleChangePerPage = (newPerPage) => {
    emit("changePerPage", newPerPage);
};
</script>

<template>
    <div v-if="meta.total > 0" class="px-4 sm:px-5 pt-4 sm:pt-4 pb-4">
        <div
            class="flex sm:flex-row flex-col justify-between sm:items-center space-y-4 sm:space-y-0"
        >
            <!-- Mobile: First row with PerPageSelector and PageInfo -->
            <div class="sm:hidden flex justify-between items-center w-full">
                <PerPageSelector
                    :perPage="perPage"
                    :perPageOptions="perPageOptions"
                    @changePerPage="handleChangePerPage"
                />
                <PageInfo :meta="meta" />
            </div>

            <!-- Mobile: Second row with PaginationControls centered -->
            <div class="sm:hidden flex justify-center w-full">
                <PaginationControls
                    :meta="meta"
                    :currentPage="currentPage"
                    @goToPage="handleGoToPage"
                />
            </div>

            <!-- Desktop: Original layout (justify-between) -->
            <div class="hidden sm:flex items-center">
                <PerPageSelector
                    :perPage="perPage"
                    :perPageOptions="perPageOptions"
                    @changePerPage="handleChangePerPage"
                />
            </div>

            <div class="hidden sm:flex">
                <PaginationControls
                    :meta="meta"
                    :currentPage="currentPage"
                    @goToPage="handleGoToPage"
                />
            </div>

            <div class="hidden sm:block">
                <PageInfo :meta="meta" />
            </div>
        </div>
    </div>
</template>
