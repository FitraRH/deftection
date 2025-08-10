<script setup>
import { ref, onMounted, computed } from "vue";
import { usePage } from "@inertiajs/vue3";
import { useUiStore } from "../stores/useUiStore";
import { useThemeStore } from "../stores/useThemeStore";
import { storeToRefs } from "pinia";
import DetailHeader from "../Pages/DetailScan/Components/DetailHeader.vue";

const page = usePage();
const isDetailPage = computed(() => page.component === "DetailScan/Index");
const analysisId = computed(() => page.props?.analysis?.id || null);

const uiStore = useUiStore();

const handleSidebarToggle = () => {
    uiStore.toggleSidebar();
};

const themeStore = useThemeStore();
const { isDark } = storeToRefs(themeStore);

const toggleTheme = () => themeStore.toggleTheme();

// Check for saved theme preference on mount
onMounted(() => {
    themeStore.initializeTheme();
});
</script>

<template>
    <div>
        <div v-if="isDetailPage">
            <DetailHeader :analysis-id="analysisId" />
        </div>
        <header
            v-else
            class="app-header transition-content sticky top-0 z-20 flex h-[65px] items-center gap-2 border-b border-gray-200 bg-white/80 px-(--margin-x) backdrop-blur-sm backdrop-saturate-150 dark:border-dark-600 max-sm:justify-between dark:bg-dark-900/80"
        >
            <div class="xl:hidden contents">
                <button
                    @click="handleSidebarToggle"
                    class="flex flex-col justify-center space-y-1.5 rtl:mr-0.5 ltr:ml-0.5 outline-hidden focus:outline-hidden size-7 text-primary-600 dark:text-primary-400 cursor-pointer sidebar-toggle-btn"
                >
                    <span></span><span></span><span></span>
                </button>
            </div>
            <div class="flex flex-1 items-center gap-2">
                <h2
                    class="flex-1 font-medium text-gray-800 dark:text-dark-50 text-xl lg:text-2xl tracking-wide"
                >
                    <slot />
                </h2>

                <!-- Dark/Light Toggle Button -->
                <button
                    @click="toggleTheme"
                    class="p-2 rounded-lg text-dark-200 hover:text-primary-600 transition-all duration-200 cursor-pointer"
                >
                    <font-awesome-icon
                        :icon="isDark ? 'fa-solid fa-sun' : 'fa-solid fa-moon'"
                        class="text-lg"
                    />
                </button>
            </div>
        </header>
    </div>
</template>
