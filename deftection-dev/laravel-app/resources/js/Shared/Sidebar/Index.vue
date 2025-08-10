<script setup>
import { ref, onMounted, onUnmounted, watch, computed } from "vue";
import Nav from "./Components/Nav.vue";
import { route } from "ziggy-js";

// 1. Import the UI store and storeToRefs
import { useUiStore } from "@/stores/useUiStore";
import { storeToRefs } from "pinia";

// Import Logo
import darkLogo from "@/../images/dark-logo.png";
import lightLogo from "@/../images/light-logo.png";
import { useThemeStore } from "../../stores/useThemeStore";

// Initialize the theme store
const themeStore = useThemeStore();
const { isDark } = storeToRefs(themeStore);

const logoSrc = computed(() => {
    return isDark.value ? darkLogo : lightLogo;
});

// 2. Initialize the store
const uiStore = useUiStore();

// 3. Get the state and actions from the store
// `storeToRefs` makes `isSidebarOpen` reactive
const { isSidebarOpen } = storeToRefs(uiStore);
const { toggleSidebar } = uiStore; // Get the action to toggle the state

// All the previous local state (sidebarOpen, windowWidth, isMobile)
// and lifecycle hooks (onMounted, onUnmounted, watch) have been removed
// because the Pinia store now handles all of that logic globally.

// User data from props
const props = defineProps({
    user: {
        type: Object,
        required: true,
    },
});

const userInitials = computed(() => {
    if (!props.user || !props.user.name) {
        return "?";
    }

    const nameParts = props.user.name.split(" ");
    if (nameParts.length > 1) {
        // Example : "John Doe" => "JD"
        return (nameParts[0][0] + nameParts[1][0]).toUpperCase();
    }
    return nameParts[0][0].toUpperCase();
});
</script>

<template>
    <!-- The main aside element now uses `isSidebarOpen` from the store -->
    <aside
        class="sidebar-panel dark:border-dark-600/80 ltr:border-r rtl:border-l border-gray-200 transition-transform duration-300 ease-in-out"
    >
        <div class="flex h-screen grow flex-col bg-white dark:bg-dark-900">
            <header
                class="relative flex h-[61px] shrink-0 items-center justify-between pt-3 ltr:pl-6 ltr:pr-3 rtl:pl-3 rtl:pr-6"
            >
                <a
                    href="/"
                    class="font-semibold text-2xl text-start tracking-wide"
                >
                    <div class="flex items-center">
                        <img :src="logoSrc" alt="Logo" class="size-18" />

                        <span class="text-primary-600 dark:text-primary-400"
                            >Defte</span
                        >
                        <span class="text-dark-900 dark:text-dark-200"
                            >ction</span
                        >
                    </div>
                </a>
                <div class="pt-3 xl:hidden">
                    <!-- 4. The close button now calls the `toggleSidebar` action from the store -->
                    <button
                        @click="toggleSidebar"
                        class="btn-base btn shrink-0 p-0 hover:bg-gray-300/20 focus:bg-gray-300/20 text-gray-700 active:bg-gray-300/25 dark:text-dark-200 dark:hover:bg-dark-300/10 dark:focus:bg-dark-300/10 dark:active:bg-dark-300/20 size-6 rounded-full"
                        type="button"
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            aria-hidden="true"
                            data-slot="icon"
                            class="size-5 rtl:rotate-180"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="M15.75 19.5 8.25 12l7.5-7.5"
                            ></path>
                        </svg>
                    </button>
                </div>
            </header>
            <div
                data-simplebar="init"
                class="flex flex-col justify-between h-full overflow-x-hidden pb-6 simplebar-scrollable-y"
            >
                <!-- Navigation -->
                <Nav />

                <!-- Account Section -->
                <Link :href="route('settings')">
                    <div
                        class="flex items-center p-4 pb-0 gap-3 cursor-pointer transition-all duration-300 group"
                    >
                        <div
                            class="avatar relative inline-flex shrink-0"
                            color="auto"
                            style="height: 2.5rem; width: 2.5rem"
                        >
                            <div
                                class="avatar-initial avatar-display flex h-full w-full select-none items-center justify-center font-medium uppercase bg-gray-200 text-gray-700 dark:bg-surface-2 dark:text-dark-100 text-sm"
                            >
                                {{ userInitials }}
                            </div>
                        </div>
                        <div class="flex flex-col gap-1 flex-1">
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100"
                            >
                                Account
                            </p>
                            <p class="text-xs text-gray-400 dark:text-dark-300">
                                Welcome, {{ user.name }}!
                            </p>
                        </div>
                        <font-awesome-icon
                            icon="fa-solid fa-chevron-down"
                            class="text-sm transition-transform duration-300 group-hover:-rotate-90"
                        />
                    </div>
                </Link>
            </div>
        </div>
    </aside>

    <Transition
        enter-active-class="transition-opacity ease-linear duration-300"
        enter-from-class="opacity-0"
        enter-to-class="opacity-100"
        leave-active-class="transition-opacity ease-linear duration-300"
        leave-from-class="opacity-100"
        leave-to-class="opacity-0"
    >
        <div
            v-if="isSidebarOpen"
            @click="toggleSidebar"
            class="fixed inset-0 z-20 bg-gray-900/50 backdrop-blur-sm dark:bg-black/40 xl:hidden"
        ></div>
    </Transition>
</template>
