<script setup>
import { ref, computed } from "vue";

// --- Props ---
// Receives the array of screenshot objects from the parent page.
const props = defineProps({
    screenshots: {
        type: Array,
        default: () => [],
    },
});

// --- Mock Data ---
// An array of 10 screenshot objects to populate the gallery for visualization.
// const screenshots = ref([
//     {
//         id: 1,
//         url: "https://placehold.co/300x200/DC2626/FFFFFF?text=Defect+1",
//         timestamp: new Date(new Date().getTime() - 15 * 1000),
//         type: "Auto",
//     },
//     {
//         id: 2,
//         url: "https://placehold.co/300x200/4F46E5/FFFFFF?text=Capture+2",
//         timestamp: new Date(new Date().getTime() - 45 * 1000),
//         type: "Manual",
//     },
//     {
//         id: 3,
//         url: "https://placehold.co/300x200/DC2626/FFFFFF?text=Defect+3",
//         timestamp: new Date(new Date().getTime() - 80 * 1000),
//         type: "Auto",
//     },
//     {
//         id: 4,
//         url: "https://placehold.co/300x200/DC2626/FFFFFF?text=Defect+4",
//         timestamp: new Date(new Date().getTime() - 110 * 1000),
//         type: "Auto",
//     },
//     {
//         id: 5,
//         url: "https://placehold.co/300x200/DC2626/FFFFFF?text=Defect+5",
//         timestamp: new Date(new Date().getTime() - 150 * 1000),
//         type: "Auto",
//     },
//     {
//         id: 6,
//         url: "https://placehold.co/300x200/DC2626/FFFFFF?text=Defect+6",
//         timestamp: new Date(new Date().getTime() - 190 * 1000),
//         type: "Auto",
//     },
//     {
//         id: 7,
//         url: "https://placehold.co/300x200/4F46E5/FFFFFF?text=Capture+7",
//         timestamp: new Date(new Date().getTime() - 220 * 1000),
//         type: "Manual",
//     },
//     {
//         id: 8,
//         url: "https://placehold.co/300x200/DC2626/FFFFFF?text=Defect+8",
//         timestamp: new Date(new Date().getTime() - 250 * 1000),
//         type: "Auto",
//     },
//     {
//         id: 9,
//         url: "https://placehold.co/300x200/DC2626/FFFFFF?text=Defect+9",
//         timestamp: new Date(new Date().getTime() - 290 * 1000),
//         type: "Auto",
//     },
//     {
//         id: 10,
//         url: "https://placehold.co/300x200/DC2626/FFFFFF?text=Defect+10",
//         timestamp: new Date(new Date().getTime() - 330 * 1000),
//         type: "Auto",
//     },
// ]);

// --- Emits ---
// Defines the 'clear-all' event to be sent to the parent.
const emit = defineEmits(["clear-all"]);

// --- Computed Properties ---
// A simple computed property to get the number of screenshots.
const screenshotCount = computed(() => props.screenshots.length);

// --- Methods ---
// Emits the event when the user clicks the "Clear All" button.
const handleClearAll = () => {
    // screenshots.value = [];
    emit("clear-all");
};
</script>

<template>
    <div
        class="h-full flex flex-col justify-start border border-gray-200 dark:border-dark-700 rounded-lg p-6 gap-3"
    >
        <!-- Header Section -->
        <div
            class="flex flex-col items-start justify-between sm:flex-row sm:items-center"
        >
            <h2
                class="truncate text-base font-medium tracking-wide text-gray-800 dark:text-dark-100"
            >
                Captured Screenshots ({{ screenshotCount }})
            </h2>
            <button
                v-if="screenshotCount > 0"
                @click="handleClearAll"
                type="button"
                class="btn-base btn gap-2 this:error bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
            >
                <font-awesome-icon icon="fa-solid fa-trash" />
                Clear All
            </button>
        </div>

        <!-- Gallery Grid -->
        <div class="mt-6">
            <!-- Empty State -->
            <div
                v-if="screenshotCount === 0"
                class="py-16 text-center rounded-lg"
            >
                <font-awesome-icon icon="fa-solid fa-file-image" size="2xl" />
                <p
                    class="mt-4 mb-2 text-lg font-medium text-gray-700 hover:text-primary-600 focus:text-primary-600 dark:text-dark-100 dark:hover:text-primary-400 dark:focus:text-primary-400"
                >
                    No screenshots captured yet
                </p>
                <p>Screenshots will appear here when defects are detected.</p>
            </div>

            <!-- Image Grid -->
            <div v-else class="h-64 pr-2 overflow-y-auto custom-scrollbar">
                <div
                    class="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8"
                >
                    <div
                        v-for="img in screenshots"
                        :key="img.id"
                        class="relative overflow-hidden rounded-lg group"
                    >
                        <img
                            :src="img.url"
                            alt="Captured screenshot"
                            class="object-cover w-full h-32 transition-transform duration-300 ease-in-out bg-black group-hover:scale-105"
                        />
                        <!-- Overlay with info -->
                        <div
                            class="absolute inset-0 flex items-end p-2 bg-gradient-to-t from-black/70 to-transparent"
                        >
                            <div class="text-xs text-white">
                                <p class="font-bold">{{ img.type }}</p>
                                <p>{{ img.timestamp.toLocaleTimeString() }}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="bg-gray-300"></div>
</template>

<style scoped>
/* Custom scrollbar for a cleaner look in dark mode */
.custom-scrollbar::-webkit-scrollbar {
    width: 6px;
}
.custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
    border-radius: 20px;
    border: 3px solid;
    border-color: var(--color-gray-300);
}
.dark.custom-scrollbar::-webkit-scrollbar-thumb {
    border-radius: 20px;
    border: 3px solid #2a2c32;
}
</style>
