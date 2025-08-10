import { defineStore } from "pinia";
import { ref, watch } from "vue";

export const useUiStore = defineStore("ui", () => {
    // --- STATE ---
    // This holds the open/closed state of the sidebar.
    const isSidebarOpen = ref(false);

    // --- ACTIONS ---
    // A function that any component can call to toggle the sidebar.
    function toggleSidebar() {
        isSidebarOpen.value = !isSidebarOpen.value;
    }

    // --- SIDE EFFECTS ---
    // This `watch` effect adds or removes the class from the <body> tag
    // whenever the sidebar's state changes.
    watch(isSidebarOpen, (newValue) => {
        const documentBody = document.body;
        if (documentBody) {
            if (newValue) {
                documentBody.classList.add("is-sidebar-open");
            } else {
                documentBody.classList.remove("is-sidebar-open");
            }
        }
    });

    // --- RETURN ---
    // We must return everything we want to use in our components.
    return {
        isSidebarOpen,
        toggleSidebar,
    };
});
