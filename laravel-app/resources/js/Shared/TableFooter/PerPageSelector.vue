<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";

const { perPage, perPageOptions = [5, 10, 25, 50] } = defineProps({
    perPage: {
        type: [String, Number],
        required: true,
    },
    perPageOptions: {
        type: Array,
        default: () => [5, 10, 25, 50],
    },
});

const emit = defineEmits(["changePerPage"]);

const isOpen = ref(false);
const selectRef = ref(null);

const toggleDropdown = () => {
    isOpen.value = !isOpen.value;
};

const selectOption = (option) => {
    emit("changePerPage", parseInt(option));
    isOpen.value = false;
};

const closeDropdown = (event) => {
    if (selectRef.value && !selectRef.value.contains(event.target)) {
        isOpen.value = false;
    }
};

onMounted(() => {
    document.addEventListener("click", closeDropdown);
});

onUnmounted(() => {
    document.removeEventListener("click", closeDropdown);
});
</script>

<template>
    <div class="flex items-center space-x-2 text-xs-plus">
        <span>Show</span>
        <div class="flex w-fit">
            <div class="relative flex" ref="selectRef">
                <!-- Custom Select Button -->
                <button
                    @click="toggleDropdown"
                    class="bg-white dark:bg-dark-700 focus:ring-opacity-50 py-1 pr-8 pl-3 border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:hover:border-dark-400 dark:focus:border-primary-500 dark:border-dark-450 rounded-full focus:outline-none focus:ring-2 focus:ring-primary-500 h-7 text-gray-900 dark:text-white text-xs transition-colors"
                    :class="{
                        'border-primary-600 dark:border-primary-500': isOpen,
                    }"
                >
                    {{ perPage }}
                </button>

                <!-- Dropdown Arrow -->
                <div
                    class="top-0 right-1 absolute flex justify-center items-center pl-5 w-9 h-full text-gray-400 dark:text-dark-300 transition-colors pointer-events-none"
                    :class="{
                        'text-primary-600 dark:text-primary-500': isOpen,
                    }"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                        aria-hidden="true"
                        data-slot="icon"
                        class="w-2/3 transition-transform"
                        :class="{ 'rotate-180': isOpen }"
                    >
                        <path
                            fill-rule="evenodd"
                            d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                            clip-rule="evenodd"
                        ></path>
                    </svg>
                </div>

                <!-- Custom Dropdown Options -->
                <transition
                    enter-active-class="transition ease-out duration-100"
                    enter-from-class="transform opacity-0 scale-95"
                    enter-to-class="transform opacity-100 scale-100"
                    leave-active-class="transition ease-in duration-75"
                    leave-from-class="transform opacity-100 scale-100"
                    leave-to-class="transform opacity-0 scale-95"
                >
                    <div
                        v-if="isOpen"
                        class="top-full left-0 z-50 absolute bg-white dark:bg-dark-700 shadow-lg mt-1 py-1 border border-gray-300 dark:border-dark-450 rounded-lg w-full"
                    >
                        <!-- Show current value if not in options -->
                        <button
                            v-if="!perPageOptions.includes(parseInt(perPage))"
                            @click="selectOption(perPage)"
                            class="bg-primary-50 hover:bg-gray-100 focus:bg-gray-100 dark:bg-primary-900/20 dark:hover:bg-dark-600 dark:focus:bg-dark-600 px-3 py-1 focus:outline-none w-full text-primary-600 dark:text-primary-400 text-xs text-left transition-colors"
                        >
                            {{ perPage }}
                        </button>

                        <!-- Regular options -->
                        <button
                            v-for="option in perPageOptions"
                            :key="option"
                            @click="selectOption(option)"
                            class="hover:bg-gray-100 focus:bg-gray-100 dark:hover:bg-dark-600 dark:focus:bg-dark-600 px-3 py-1 focus:outline-none w-full text-xs text-left transition-colors"
                            :class="{
                                'bg-primary-50 dark:bg-primary-900/20 text-primary-600 dark:text-primary-400':
                                    parseInt(perPage) === option,
                                'text-gray-900 dark:text-white':
                                    parseInt(perPage) !== option,
                            }"
                        >
                            {{ option }}
                        </button>
                    </div>
                </transition>
            </div>
        </div>
        <span>entries</span>
    </div>
</template>
