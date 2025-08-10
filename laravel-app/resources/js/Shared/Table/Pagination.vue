<script setup>
import { Link } from "@inertiajs/vue3";
import { computed } from "vue";
import { router } from "@inertiajs/vue3";

// --- Props ---
// This component receives the pagination object directly from Laravel's Paginator.
const props = defineProps({
    pagination: {
        type: Object,
        required: true,
        default: () => ({
            links: [],
            from: 0,
            to: 0,
            total: 0,
            current_page: 1,
            last_page: 1,
        }),
    },
    itemsPerPageValue: {
        type: String,
        default: "5",
    },
});

// --- Emits ---
// Emits an event when the "items per page" dropdown is changed.
const emit = defineEmits(["update:itemsPerPage", "page-change"]);

// --- Computed Properties ---
// A computed property to handle the v-model for the items per page select.
const itemsPerPage = computed({
    get() {
        return props.itemsPerPageValue;
    },
    set(value) {
        // When a new value is selected, emit an event to the parent
        emit("update:itemsPerPage", value);
    },
});

/**
 * A helper function to handle page clicks.
 * For real links from Laravel, it will use Inertia's router.
 * For our mock links, it will emit an event.
 */
const handlePageClick = (link) => {
    if (!link.url) return; // Do nothing if the link is disabled

    // For the mock setup, the URL is just a number. We emit an event.
    if (typeof link.url === "number") {
        emit("page-change", link.url);
        return;
    }

    // For a real setup, if the URL is a string, use Inertia's router.
    // This part is for when your backend is ready.
    router.visit(link.url, { preserveState: true });
};
</script>

<template>
    <div
        v-if="pagination && pagination.total > 0"
        class="px-4 pb-4 sm:px-5 sm:pt-4 pt-4"
    >
        <div
            class="flex flex-col justify-between space-y-4 sm:flex-row sm:items-center sm:space-y-0"
        >
            <!-- "Show X entries" dropdown -->
            <div class="flex items-center space-x-2 text-xs+">
                <span>Show</span>
                <div class="input-root w-fit">
                    <div class="input-wrapper relative">
                        <select
                            id="items-per-page"
                            v-model="itemsPerPage"
                            class="form-select-base form-select ltr:pr-9 rtl:pl-9 peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500 h-7 rounded-full py-1 text-xs ltr:pr-7! rtl:pl-7!"
                        >
                            <option value="5">5</option>
                            <option value="10">10</option>
                            <option value="20">20</option>
                        </select>
                        <div
                            class="suffix ltr:right-0 rtl:left-0 pointer-events-none absolute top-0 flex h-full w-9 items-center justify-center transition-colors text-gray-400 peer-focus:text-primary-600 dark:text-dark-300 dark:peer-focus:text-primary-500"
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 20 20"
                                fill="currentColor"
                                aria-hidden="true"
                                data-slot="icon"
                                class="w-2/3"
                            >
                                <path
                                    fill-rule="evenodd"
                                    d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                                    clip-rule="evenodd"
                                ></path>
                            </svg>
                        </div>
                    </div>
                </div>
                <span>entries</span>
            </div>

            <!-- Page links -->
            <div>
                <div
                    class="pagination hide-scrollbar max-w-full overflow-x-auto"
                >
                    <button
                        v-for="(link, index) in pagination.links"
                        @click="handlePageClick(link)"
                        :key="index"
                        :href="link.url"
                        :disabled="!link.url"
                        class="pagination-control focus-visible:bg-gray-300 active:bg-gray-300/80 dark:focus-visible:bg-surface-1 dark:active:bg-surface-1/90"
                        :class="{
                            'cursor-pointer': link.url,
                            'cursor-not-allowed opacity-60': !link.url,
                            'this:primary bg-this text-white dark:bg-this dark:text-white':
                                link.active,
                            'dark:text-white hover:bg-gray-300 dark:hover:bg-surface-1':
                                !link.active,
                        }"
                        v-html="link.label"
                    />
                </div>
            </div>

            <!-- "Showing X to Y of Z" text -->
            <div class="truncate text-xs-plus">
                {{ pagination.from }} - {{ pagination.to }} of
                {{ pagination.total }} entries
            </div>
        </div>
    </div>
</template>

<style scoped>
.dark #items-per-page {
    --bg-color: rgb(14, 15, 17);
}

#items-per-page {
    --bg-color: rgb(255, 255, 255);
}
</style>
