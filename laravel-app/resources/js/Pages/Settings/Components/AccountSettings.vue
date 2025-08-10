<script setup>
import { computed } from "vue";
import { useForm, usePage } from "@inertiajs/vue3";

// --- Get Authenticated User ---
// usePage() gives us access to the shared data from Laravel, including the authenticated user.
const page = usePage();
// A computed property to safely access the user object.
const user = computed(() => page.props.auth.user);

// --- Form Management ---

// The form is now initialized with the real user's name from Inertia's props.
const profileForm = useForm({
    name: user.value?.name || "",
});

// A form for changing the user's password.
const passwordForm = useForm({
    current_password: "",
    new_password: "",
    confirm_password: "",
});

// --- Event Handlers ---

const submitProfile = () => {
    // In a real application, you would post this to your backend.
    // profileForm.post(route('profile.update'), { onSuccess: () => ... });
    console.log("Updating profile with:", profileForm.data());
    // You would typically show a success notification here.
};

const submitPassword = () => {
    // In a real application, you would post this to your backend.
    // passwordForm.post(route('password.update'), { onSuccess: () => passwordForm.reset() });
    console.log("Changing password...");
    // Clear the form fields after submission for security.
    passwordForm.reset();
};
</script>

<template>
    <div
        class="px-6 py-5 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
    >
        <!-- Card 1: User Settings -->
        <div class="flex flex-col">
            <h2
                class="mb-1 flex items-center text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                User Settings
            </h2>
            <p class="mb-6 text-xs text-gray-400 dark:text-dark-300">
                Manage your account settings and preferences.
            </p>

            <form @submit.prevent="submitProfile">
                <div class="grid grid-cols-1 gap-6 sm:grid-cols-2">
                    <!-- Full Name -->
                    <div>
                        <label
                            for="fullName"
                            class="mb-2 block text-sm font-medium text-gray-800 dark:text-dark-100"
                            >Full Name</label
                        >
                        <input
                            v-model="profileForm.name"
                            type="text"
                            id="fullName"
                            class="form-input-base form-input peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                        />
                    </div>
                    <!-- Username (Disabled) -->
                    <div>
                        <label
                            for="username"
                            class="mb-2 block text-sm font-medium text-gray-800 dark:text-dark-100"
                            >Username</label
                        >
                        <input
                            :value="user?.name"
                            type="text"
                            id="username"
                            disabled
                            class="mb-1 form-input-base form-input cursor-not-allowed border-gray-300 bg-gray-150 opacity-60 dark:border-dark-500 dark:bg-dark-600"
                        />
                        <span
                            class="input-text-error text-xs text-error dark:text-error-lighter"
                        >
                            Username cannot be changed.
                        </span>
                    </div>
                </div>
                <!-- The main "Save Settings" button is in the parent Index.vue -->
            </form>
        </div>

        <div class="mt-6 mb-8 h-px bg-gray-200 dark:bg-dark-500"></div>

        <!-- Card 2: Change Password -->
        <div class="flex flex-col gap-4">
            <h2
                class="flex items-center text-lg font-semibold text-gray-900 dark:text-dark-50"
            >
                Change Password
            </h2>

            <form @submit.prevent="submitPassword">
                <div class="grid grid-cols-1 gap-6 sm:grid-cols-3">
                    <!-- Current Password -->
                    <div>
                        <label
                            for="currentPassword"
                            class="mb-2 block text-sm font-medium text-gray-800 dark:text-dark-100"
                            >Current Password</label
                        >
                        <input
                            v-model="passwordForm.current_password"
                            type="password"
                            id="currentPassword"
                            placeholder="Enter your current password"
                            class="form-input-base form-input peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                        />
                    </div>
                    <!-- New Password -->
                    <div>
                        <label
                            for="newPassword"
                            class="mb-2 block text-sm font-medium text-gray-800 dark:text-dark-100"
                            >New Password</label
                        >
                        <input
                            v-model="passwordForm.new_password"
                            type="password"
                            id="newPassword"
                            placeholder="Enter your new password"
                            class="form-input-base form-input peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                        />
                    </div>
                    <!-- Confirm Password -->
                    <div>
                        <label
                            for="confirmPassword"
                            class="mb-2 block text-sm font-medium text-gray-800 dark:text-dark-100"
                            >Confirm Password</label
                        >
                        <input
                            v-model="passwordForm.confirm_password"
                            type="password"
                            id="confirmPassword"
                            placeholder="Confirm your new password"
                            class="form-input-base form-input peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                        />
                    </div>
                </div>
                <div class="mt-6">
                    <button
                        type="submit"
                        class="btn-base btn this:error bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
                        :disabled="passwordForm.processing"
                    >
                        Change Password
                    </button>
                </div>
            </form>
        </div>
    </div>
</template>
