<script setup>
import { ref } from "vue";
// --- Props ---
// This component receives the list of captured screenshots from its parent.
// The parent is responsible for managing this list.
const props = defineProps({
    detections: {
        type: Array,
        default: () => [], // Default to an empty array
    },
});

// --- Mock Data for Visualization ---
// This data is used to preview the component's appearance.
// const detections = ref([
//     {
//         id: 1,
//         url: "https://placehold.co/64x48/DC2626/FFFFFF?text=Scratched",
//         timestamp: new Date(new Date().getTime() - 10 * 1000), // 10 seconds ago
//         type: "Scratched",
//     },
//     {
//         id: 2,
//         url: "https://placehold.co/64x48/DC2626/FFFFFF?text=Opened",
//         timestamp: new Date(new Date().getTime() - 35 * 1000), // 35 seconds ago
//         type: "Opened",
//     },
//     {
//         id: 3,
//         url: "https://placehold.co/64x48/DC2626/FFFFFF?text=Opened",
//         timestamp: new Date(new Date().getTime() - 72 * 1000), // 72 seconds ago
//         type: "Opened",
//     },
//     {
//         id: 4,
//         url: "https://placehold.co/64x48/DC2626/FFFFFF?text=TauAh",
//         timestamp: new Date(new Date().getTime() - 120 * 1000), // 2 minutes ago
//         type: "Tau Ah",
//     },
//     {
//         id: 5,
//         url: "https://placehold.co/64x48/DC2626/FFFFFF?text=Auto",
//         timestamp: new Date(new Date().getTime() - 155 * 1000), // ~2.5 minutes ago
//         type: "Auto",
//     },
//     {
//         id: 6,
//         url: "https://placehold.co/64x48/4F46E5/FFFFFF?text=Manual",
//         timestamp: new Date(new Date().getTime() - 180 * 1000), // 3 minutes ago
//         type: "Manual",
//     },
//     {
//         id: 7,
//         url: "https://placehold.co/64x48/DC2626/FFFFFF?text=Auto",
//         timestamp: new Date(new Date().getTime() - 210 * 1000), // ~3.5 minutes ago
//         type: "Auto",
//     },
//     {
//         id: 8,
//         url: "https://placehold.co/64x48/DC2626/FFFFFF?text=Auto",
//         timestamp: new Date(new Date().getTime() - 245 * 1000), // ~4 minutes ago
//         type: "Auto",
//     },
//     {
//         id: 9,
//         url: "https://placehold.co/64x48/DC2626/FFFFFF?text=Auto",
//         timestamp: new Date(new Date().getTime() - 280 * 1000), // ~4.5 minutes ago
//         type: "Auto",
//     },
// ]);
// The template below will now use this local 'detections' ref.
</script>

<template>
    <div
        class="flex flex-col justify-start border border-gray-200 dark:border-dark-700 rounded-lg p-6 gap-3"
    >
        <h2
            class="truncate text-base font-medium tracking-wide text-gray-800 dark:text-dark-100"
        >
            Recent Detections
        </h2>

        <!-- Container for the detection list -->
        <div
            class="h-72 pr-2 space-y-3 custom-scrollbar"
            :class="{ 'overflow-y-auto': detections.length > 0 }"
        >
            <!-- Empty State: Shown when there are no detections -->
            <div
                v-if="detections.length === 0"
                class="flex flex-col items-center justify-center h-full text-center gap-2"
            >
                <font-awesome-icon icon="fa-solid fa-layer-group" size="2xl" />
                <span>No detections yet</span>
            </div>

            <!-- Detection List: Shown when there are detections -->
            <transition-group name="list" tag="div">
                <div
                    v-for="detection in detections"
                    :key="detection.id"
                    class="flex items-center p-2 space-x-3 transition-all duration-300 rounded-md"
                >
                    <!-- Thumbnail Image -->
                    <img
                        :src="detection.url"
                        alt="Detection screenshot"
                        class="object-cover w-16 h-12 bg-black rounded-md"
                    />
                    <!-- Detection Info -->
                    <div class="flex-grow">
                        <p class="text-sm font-semibold text-red-400">
                            {{ detection.type }}
                        </p>
                        <p class="text-xs text-gray-400">
                            {{ detection.timestamp.toLocaleTimeString() }}
                        </p>
                    </div>
                </div>
            </transition-group>
        </div>
    </div>
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

/* Transition for new items entering the list */
.list-enter-active,
.list-leave-active {
    transition: all 0.5s ease;
}
.list-enter-from {
    opacity: 0;
    transform: translateX(-20px);
}
.list-leave-to {
    opacity: 0;
    transform: scale(0.9);
}
</style>
