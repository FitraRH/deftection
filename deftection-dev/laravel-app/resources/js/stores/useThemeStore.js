// stores/theme.js
import { defineStore } from "pinia";
import { ref, computed } from "vue";

export const useThemeStore = defineStore("theme", () => {
    const isDark = ref(false);
    const theme = computed(() => (isDark.value ? "dark" : "light"));

    const getTheme = () => {
        return isDark.value;
    };

    const toggleTheme = () => {
        isDark.value = !isDark.value;
        applyTheme();
    };

    const setTheme = (newTheme) => {
        isDark.value = newTheme === "dark";
        applyTheme();
    };

    const applyTheme = () => {
        if (typeof document === "undefined") return;

        if (isDark.value) {
            document.documentElement.classList.add("dark");
            document.documentElement.setAttribute("data-theme-dark", "cinder");
            document.body.className = "bg-dark-900 text-dark-200";
        } else {
            document.documentElement.classList.remove("dark");
            document.documentElement.removeAttribute("data-theme-dark");
            document.body.className = "bg-gray-50 text-gray-900";
        }

        localStorage.setItem("theme", isDark.value ? "dark" : "light");
    };

    const initializeTheme = () => {
        if (typeof window === "undefined") return;

        const savedTheme = localStorage.getItem("theme");
        if (savedTheme) {
            isDark.value = savedTheme === "dark";
        } else {
            const prefersDark = window.matchMedia(
                "(prefers-color-scheme: dark)"
            ).matches;
            isDark.value = prefersDark;
        }

        applyTheme();
    };

    return {
        isDark,
        theme,
        toggleTheme,
        setTheme,
        applyTheme,
        initializeTheme,
    };
});
