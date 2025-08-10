<script setup>
import { computed, ref, onMounted, watch } from "vue";
import { Link, useForm } from "@inertiajs/vue3";
import { route } from "ziggy-js";
import { useThemeStore } from "@/stores/useThemeStore.js";
import { storeToRefs } from "pinia";

// Import Logo
import lightLogo from "@/../images/light-logo.png";

const isLoading = ref(false);

const themeStore = useThemeStore();
const { isDark } = storeToRefs(themeStore);
const toggleTheme = () => themeStore.toggleTheme();

// Form data
const form = useForm({
    username: "",
    email: "",
    password: "",
    password_confirmation: "",
});

// Add watchers for form
watch(
    () => form.username,
    () => {
        if (form.errors.username) {
            form.clearErrors("username");
        }
    }
);

watch(
    () => form.email,
    () => {
        if (form.errors.email) {
            form.clearErrors("email");
        }
    }
);

// Password mismatch state
const passwordMismatch = computed(() => {
    return (
        form.password &&
        form.password_confirmation &&
        form.password !== form.password_confirmation
    );
});

// Handle login submission
const handleRegister = async () => {
    if (passwordMismatch.value || form.hasErrors) {
        console.log("Form has errors or passwords do not match");
        return;
    }

    isLoading.value = true;

    // Tinggal set routenya
    form.post(
        route("register.store"),
        {
            onFinish: () => {
                isLoading.value = false;
            },
            onError: () => {
                console.log("Login failed");
            },
        },
        {
            replace: true,
        }
    );
};

onMounted(() => {
    themeStore.initializeTheme();
});

defineOptions({
    layout: false,
});
</script>

<template>
    <main
        class="place-items-center grid grid-cols-1 dark:bg-dark-900 w-full min-h-100vh grow"
    >
        <div class="p-4 sm:px-5 w-full max-w-[26rem]">
            <!-- Header Section -->
            <div class="text-center">
                <div class="flex items-center justify-center mr-10">
                    <!-- Logo (you can replace with your own) -->
                    <img :src="lightLogo" alt="Logo" class="size-32" />
                    <div
                        class="flex justify-center font-semibold text-3xl text-center tracking-wide"
                    >
                        <span class="text-primary-400">Defte</span>
                        <span class="text-dark-900 dark:text-dark-200"
                            >ction</span
                        >
                    </div>
                </div>
            </div>

            <!-- Register Card -->
            <div
                class="dark:bg-dark-800 mt-5 p-5 lg:p-7 border dark:border-dark-600 rounded-lg"
            >
                <form @submit.prevent="handleRegister">
                    <div class="space-y-4">
                        <!-- Username Input -->
                        <div>
                            <label
                                class="block mb-2 font-medium text-dark-400 dark:text-dark-200 text-sm"
                            >
                                Username
                            </label>
                            <div class="relative">
                                <div
                                    class="left-0 absolute inset-y-0 flex items-center pl-3 pointer-events-none"
                                >
                                    <font-awesome-icon
                                        icon="fa-solid fa-user"
                                        class="size-5 text-dark-400 transition-colors duration-200"
                                    />
                                </div>
                                <input
                                    v-model="form.username"
                                    type="text"
                                    placeholder="Enter Username"
                                    autocomplete="off"
                                    class="dark:bg-dark-700 py-2 pr-3 pl-10 border focus:border-transparent rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 w-full dark:text-dark-100 transition-all duration-200 placeholder-dark-400"
                                    :class="{
                                        'border-red-500': form.errors.username,
                                        'border-gray-300 dark:border-dark-600':
                                            !form.errors.username,
                                    }"
                                />
                            </div>
                        </div>
                        <div v-if="form.errors.username">
                            <p class="mt-1 text-red-500 text-sm">
                                {{ form.errors.username }}
                            </p>
                        </div>

                        <!-- Email Input -->
                        <div>
                            <label
                                class="block mb-2 font-medium text-dark-400 dark:text-dark-200 text-sm"
                            >
                                Email
                            </label>
                            <div class="relative">
                                <div
                                    class="left-0 absolute inset-y-0 flex items-center pl-3 pointer-events-none"
                                >
                                    <font-awesome-icon
                                        icon="fa-solid fa-envelope"
                                        class="size-5 text-dark-400 transition-colors duration-200"
                                    />
                                </div>
                                <input
                                    v-model="form.email"
                                    type="text"
                                    placeholder="Enter Email"
                                    class="dark:bg-dark-700 py-2 pr-3 pl-10 border focus:border-transparent rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 w-full dark:text-dark-100 transition-all duration-200 placeholder-dark-400"
                                    :class="{
                                        'border-red-500': form.errors.email,
                                        'border-gray-300 dark:border-dark-600':
                                            !form.errors.email,
                                    }"
                                />
                            </div>
                        </div>
                        <div v-if="form.errors.email">
                            <p class="mt-1 text-red-500 text-sm">
                                {{ form.errors.email }}
                            </p>
                        </div>

                        <!-- Password Input -->
                        <div>
                            <label
                                class="block mb-2 font-medium text-dark-400 dark:text-dark-200 text-sm"
                            >
                                Password
                            </label>
                            <div class="relative">
                                <div
                                    class="left-0 absolute inset-y-0 flex items-center pl-3 pointer-events-none"
                                >
                                    <font-awesome-icon
                                        icon="fa-solid fa-lock"
                                        class="size-5 text-dark-400 transition-colors duration-200"
                                    />
                                </div>
                                <input
                                    v-model.lazy="form.password"
                                    type="password"
                                    placeholder="Enter Password"
                                    class="dark:bg-dark-700 py-2 pr-3 pl-10 border dark:border-dark-600 focus:border-transparent rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 w-full dark:text-dark-100 transition-all duration-200 placeholder-dark-400"
                                />
                            </div>
                        </div>

                        <!-- Repeat Password Input -->
                        <div>
                            <label
                                class="block mb-2 font-medium text-dark-400 dark:text-dark-200 text-sm"
                            >
                                Re-enter Password
                            </label>
                            <div class="relative">
                                <div
                                    class="left-0 absolute inset-y-0 flex items-center pl-3 pointer-events-none"
                                >
                                    <font-awesome-icon
                                        icon="fa-solid fa-lock"
                                        class="size-5 text-dark-400 transition-colors duration-200"
                                    />
                                </div>
                                <input
                                    v-model.lazy="form.password_confirmation"
                                    type="password"
                                    placeholder="Re-enter Password"
                                    class="dark:bg-dark-700 py-2 pr-3 pl-10 border focus:border-transparent rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 w-full dark:text-dark-100 transition-all duration-200 placeholder-dark-400"
                                    :class="{
                                        'border-red-500': passwordMismatch,
                                        'border-gray-300 dark:border-dark-600':
                                            !passwordMismatch,
                                    }"
                                />
                            </div>
                        </div>
                        <div v-if="passwordMismatch">
                            <p class="mt-1 text-red-500 text-sm">
                                Passwords do not match.
                            </p>
                        </div>
                    </div>

                    <!-- Sign Up Button -->
                    <button
                        type="submit"
                        class="bg-primary-600 hover:bg-primary-700 focus:bg-primary-700 disabled:opacity-70 mt-5 py-2.5 w-full font-medium text-white transition-all duration-200 btn-base btn"
                    >
                        <span v-if="!isLoading">Sign Up</span>
                        <span
                            v-else
                            class="flex justify-center items-center gap-2"
                        >
                            <svg
                                class="w-4 h-4 animate-spin"
                                fill="none"
                                viewBox="0 0 24 24"
                            >
                                <circle
                                    class="opacity-25"
                                    cx="12"
                                    cy="12"
                                    r="10"
                                    stroke="currentColor"
                                    stroke-width="4"
                                ></circle>
                                <path
                                    class="opacity-75"
                                    fill="currentColor"
                                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                ></path>
                            </svg>
                            Signing Up...
                        </span>
                    </button>
                </form>

                <!-- Sign In -->
                <div
                    class="mt-4 text-gray-500 dark:text-dark-300 text-sm text-center"
                >
                    Already have an account?
                    <Link
                        :href="route('login')"
                        class="text-primary-500 hover:text-primary-600 transition-colors"
                    >
                        Sign In
                    </Link>
                </div>
            </div>
        </div>

        <button
            @click="toggleTheme"
            class="group right-4 bottom-4 fixed backdrop-blur-sm p-2.5 rounded-lg transition-all duration-200 cursor-pointer"
            :title="isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'"
        >
            <font-awesome-icon
                :icon="isDark ? 'fa-solid fa-sun' : 'fa-solid fa-moon'"
                class="text-dark-300 group-hover:text-primary-400 transition-colors duration-200"
            />
        </button>
    </main>
</template>
