<script setup>
import { ref, watch, nextTick, computed, onMounted } from "vue";
import FormModal from "@/Shared/Modals/FormModal.vue";
import FormModalLabel from "@/Shared/Modals/FormModalLabel.vue";
import FormModalInput from "@/Shared/Modals/FormModalInput.vue";
import FormModalSelect from "@/Shared/Modals/FormModalSelect.vue";
import FormModalTips from "@/Shared/Modals/FormModalTips.vue";
import { useForm } from "@inertiajs/vue3";
import {
    unsavedChangesDialog,
    successToast,
    errorToast,
    editConfirmDialog,
} from "@/utils/swal";

// Props
const props = defineProps({
    isOpen: {
        type: Boolean,
        required: true,
    },
    userId: {
        type: [String, Number],
        required: true,
    },
    redirectAfterUpdate: {
        type: Boolean,
        default: false,
    },
});

// Emit events for parent component
const emit = defineEmits(["operation", "close"]);

// Loading states
const isLoading = ref(false);
const isFetching = ref(false);

// User data
const user = ref(null);
const originalData = ref(null);

// Role options
const roleOptions = [
    { value: "admin", label: "Admin" },
    { value: "user", label: "User" },
];

// Form initialization
const initializeForm = () => ({
    name: "",
    email: "",
    role: "user",
    password: "",
    password_confirmation: "",
});

// Form data
const form = useForm(initializeForm());

// Computed properties
const hasChanges = computed(() => {
    if (!originalData.value) return false;
    return (
        form.name?.trim() !== originalData.value.name?.trim() ||
        form.email?.trim().toLowerCase() !==
            originalData.value.email?.trim().toLowerCase() ||
        form.role !== originalData.value.role ||
        form.password // If password is provided, consider it a change
    );
});

const isFormDisabled = computed(() => isLoading.value || isFetching.value);

// Format date helper
const formatDate = (dateString) => {
    if (!dateString) return "N/A";

    try {
        const date = new Date(dateString);
        return new Intl.DateTimeFormat("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            hour12: true,
        }).format(date);
    } catch (error) {
        return dateString; // Return original if formatting fails
    }
};

// Dynamic tips based on current state
const editTips = computed(() => {
    const tips = [
        "Leave password fields empty to keep current password",
        "Email must be unique in the system",
    ];

    // Add original data tips if we have original data
    if (originalData.value) {
        if (hasChanges.value) {
            tips.push({
                text: `Original email: "${originalData.value.email}"`,
                condition:
                    form.email?.trim().toLowerCase() !==
                    originalData.value.email?.trim().toLowerCase(),
            });
        }
    }

    return tips;
});

// Validation rules
const validateForm = () => {
    const errors = {};

    // Name validation
    if (!form.name?.trim()) {
        errors.name = "Name is required";
    } else if (form.name.trim().length < 2) {
        errors.name = "Name must be at least 2 characters";
    } else if (form.name.trim().length > 255) {
        errors.name = "Name must not exceed 255 characters";
    }

    // Email validation
    if (!form.email?.trim()) {
        errors.email = "Email is required";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) {
        errors.email = "Please enter a valid email address";
    } else if (form.email.trim().length > 255) {
        errors.email = "Email must not exceed 255 characters";
    }

    // Role validation
    if (!form.role) {
        errors.role = "Role is required";
    } else if (!roleOptions.some((option) => option.value === form.role)) {
        errors.role = "Please select a valid role";
    }

    // Password validation (only if password is provided)
    if (form.password) {
        if (form.password.length < 8) {
            errors.password = "Password must be at least 8 characters";
        } else if (form.password.length > 255) {
            errors.password = "Password must not exceed 255 characters";
        }

        // Password confirmation validation (only if password is provided)
        if (!form.password_confirmation) {
            errors.password_confirmation =
                "Password confirmation is required when changing password";
        } else if (form.password !== form.password_confirmation) {
            errors.password_confirmation = "Passwords do not match";
        }
    }

    return errors;
};

// Clear errors when fields change
watch(
    () => form.name,
    (newValue) => {
        if (form.errors.name && newValue?.trim()) {
            form.clearErrors("name");
        }
    }
);

watch(
    () => form.email,
    (newValue) => {
        if (form.errors.email && newValue?.trim()) {
            form.clearErrors("email");
        }
    }
);

watch(
    () => form.role,
    (newValue) => {
        if (form.errors.role && newValue) {
            form.clearErrors("role");
        }
    }
);

watch(
    () => form.password,
    (newValue) => {
        if (form.errors.password && newValue) {
            form.clearErrors("password");
        }
        // Clear confirmation error if passwords now match
        if (
            form.errors.password_confirmation &&
            newValue === form.password_confirmation
        ) {
            form.clearErrors("password_confirmation");
        }
    }
);

watch(
    () => form.password_confirmation,
    (newValue) => {
        if (form.errors.password_confirmation && newValue === form.password) {
            form.clearErrors("password_confirmation");
        }
    }
);

// Fetch user data
const fetchUser = async () => {
    if (isFetching.value) {
        console.warn("Fetch already in progress, skipping...");
        return;
    }

    if (!props.userId) {
        console.error("No user ID provided");
        return;
    }

    isFetching.value = true;

    try {
        const url = route("users.show.api", props.userId);
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: Failed to fetch user`);
        }

        const res = await response.json();

        if (!res.success || !res.data) {
            throw new Error(
                "Invalid response format - missing success or data"
            );
        }

        // Set user data
        user.value = res.data;
        originalData.value = { ...res.data };

        // Pre-fill form with existing data
        await nextTick();
        form.name = res.data.name;
        form.email = res.data.email;
        form.role = res.data.role;
        form.password = "";
        form.password_confirmation = "";
        form.clearErrors();

        console.log("User fetched successfully:", res.data);

        // Emit success event
        emit("operation", {
            type: "success",
            action: "fetch",
            data: res.data,
            message: "User data loaded successfully",
        });
    } catch (error) {
        console.error("Error fetching user:", error);

        // Emit error event
        emit("operation", {
            type: "error",
            action: "fetch",
            error,
            message: `Failed to fetch user data: ${error.message}`,
        });

        // Show error toast
        errorToast(`Failed to fetch user data: ${error.message}`);

        // Close modal on fetch error
        closeModal();
    } finally {
        isFetching.value = false;
    }
};

// Watch for props changes
watch(
    () => props.isOpen,
    (newValue) => {
        if (newValue && props.userId) {
            initializeAndFetch();
        }
    }
);

watch(
    () => props.userId,
    (newValue) => {
        if (props.isOpen && newValue) {
            initializeAndFetch();
        }
    }
);

// Initialize and fetch function
const initializeAndFetch = async () => {
    // Reset everything
    form.reset();
    form.clearErrors();
    user.value = null;
    originalData.value = null;

    // Fetch new data
    await fetchUser();
};

// Refresh function for refresh button
const handleRefresh = async () => {
    if (hasChanges.value) {
        const result = await unsavedChangesDialog();
        if (!result.isConfirmed) {
            return; // User cancelled
        }
    }

    await initializeAndFetch();
    successToast("User data refreshed!");
};

// Mount handler
onMounted(() => {
    // If modal is already open when mounted, fetch immediately
    if (props.isOpen && props.userId) {
        nextTick(() => {
            initializeAndFetch();
        });
    }
});

// Handle form submission
const handleSubmit = async () => {
    // Client-side validation
    const clientErrors = validateForm();
    if (Object.keys(clientErrors).length > 0) {
        Object.keys(clientErrors).forEach((key) => {
            form.setError(key, clientErrors[key]);
        });
        return;
    }

    // Check if there are actual changes
    if (!hasChanges.value) {
        errorToast("No changes detected to save.");
        return;
    }

    // Show edit confirmation dialog
    const result = await editConfirmDialog("this user");
    if (!result.isConfirmed) {
        return;
    }

    isLoading.value = true;

    // Transform data before sending
    const formData = {
        name: form.name?.trim(),
        email: form.email?.trim().toLowerCase(),
        role: form.role,
    };

    // Only include password fields if password is provided
    if (form.password) {
        formData.password = form.password;
        formData.password_confirmation = form.password_confirmation;
    }

    form.transform(() => formData).put(route("users.update", props.userId), {
        preserveState: true,
        preserveScroll: true,
        onSuccess: (response) => {
            handleSuccess(response);
        },
        onError: (errors) => {
            handleError(errors);
        },
        onFinish: () => {
            isLoading.value = false;
            closeModal();
            handleFinish();
        },
    });
};

// Handle success response
const handleSuccess = (response) => {
    emit("operation", {
        type: "success",
        action: "update",
        data: response,
        message: "User updated successfully",
    });

    successToast("User updated successfully!");
};

// Handle error response
const handleError = (errors) => {
    console.error("Form submission errors:", errors);

    emit("operation", {
        type: "error",
        action: "update",
        error: errors,
        message: "Failed to update user",
    });

    errorToast("Failed to update user. Please check your input.");
};

// Handle operation finish
const handleFinish = () => {
    emit("operation", {
        type: "finish",
        action: "update",
        message: "Update operation completed",
    });
};

// Close modal
const closeModal = () => {
    if (!isFormDisabled.value) {
        emit("close");
        setTimeout(() => {
            form.reset();
            form.clearErrors();
            user.value = null;
            originalData.value = null;
        }, 150);
    }
};

// Handle modal close with unsaved changes check
const handleClose = async () => {
    if (isFormDisabled.value) {
        return;
    }

    if (hasChanges.value) {
        const result = await unsavedChangesDialog();
        if (result.isConfirmed) {
            closeModal();
        }
    } else {
        closeModal();
    }
};

// Helper to get loading status
const getLoadingStatus = () => {
    if (isFetching.value) return "Loading user data...";
    if (isLoading.value) return "Updating user...";
    return "";
};
</script>

<template>
    <FormModal
        :is-open="isOpen"
        :is-loading="isFormDisabled"
        title="Edit User"
        description="Update the user information below and click save to apply changes."
        submit-text="Save Changes"
        :loading-text="getLoadingStatus()"
        :has-errors="form.hasErrors"
        focus="edit-name"
        :prevent-close-on-outside-click="isFormDisabled"
        :show-close-button="!isFormDisabled"
        :show-refresh-button="true"
        :is-refreshing="isFetching"
        @close="handleClose"
        @submit="handleSubmit"
        @refresh="handleRefresh"
    >
        <div class="space-y-4">
            <!-- Loading state for fetching data -->
            <div
                v-if="isFetching"
                class="flex justify-center items-center py-8 text-gray-500 dark:text-gray-400"
            >
                <svg
                    class="mr-3 -ml-1 w-5 h-5 animate-spin"
                    xmlns="http://www.w3.org/2000/svg"
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
                Loading user data...
            </div>

            <!-- Form content -->
            <div v-else>
                <!-- Name Field -->
                <div>
                    <FormModalLabel for="edit-name" required>
                        Full Name
                    </FormModalLabel>
                    <FormModalInput
                        id="edit-name"
                        ref="nameInput"
                        v-model="form.name"
                        type="text"
                        placeholder="Enter full name (e.g., John Doe)"
                        maxlength="255"
                        required
                        :disabled="isFormDisabled"
                        :class="{
                            'border-red-500 focus:border-red-500':
                                form.errors.name,
                            'border-gray-300 focus:border-blue-500':
                                !form.errors.name,
                        }"
                        @keydown.enter.prevent="handleSubmit"
                    />

                    <!-- Name Error display -->
                    <div
                        v-if="form.errors.name"
                        class="flex items-center mt-1 text-red-600 dark:text-red-400 text-sm"
                    >
                        <svg
                            class="mr-1 w-4 h-4"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                        >
                            <path
                                fill-rule="evenodd"
                                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                                clip-rule="evenodd"
                            />
                        </svg>
                        {{ form.errors.name }}
                    </div>

                    <!-- Character count and change indicator -->
                    <div class="flex justify-between items-center mt-1">
                        <div class="flex items-center text-xs">
                            <span
                                v-if="
                                    hasChanges &&
                                    originalData &&
                                    form.name?.trim() !==
                                        originalData.name?.trim()
                                "
                                class="flex items-center mr-2 text-orange-600 dark:text-orange-400"
                            >
                                <svg
                                    class="mr-1 w-3 h-3"
                                    fill="currentColor"
                                    viewBox="0 0 20 20"
                                >
                                    <path
                                        fill-rule="evenodd"
                                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                        clip-rule="evenodd"
                                    />
                                </svg>
                                Modified
                            </span>
                        </div>
                        <div class="text-gray-500 dark:text-gray-400 text-xs">
                            {{ (form.name || "").length }}/255
                        </div>
                    </div>
                </div>

                <!-- Email Field -->
                <div>
                    <FormModalLabel for="edit-email" required>
                        Email Address
                    </FormModalLabel>
                    <FormModalInput
                        id="edit-email"
                        v-model="form.email"
                        type="email"
                        placeholder="Enter email address (e.g., john@example.com)"
                        maxlength="255"
                        required
                        :disabled="isFormDisabled"
                        :class="{
                            'border-red-500 focus:border-red-500':
                                form.errors.email,
                            'border-gray-300 focus:border-blue-500':
                                !form.errors.email,
                        }"
                        @keydown.enter.prevent="handleSubmit"
                    />

                    <!-- Email Error display -->
                    <div
                        v-if="form.errors.email"
                        class="flex items-center mt-1 text-red-600 dark:text-red-400 text-sm"
                    >
                        <svg
                            class="mr-1 w-4 h-4"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                        >
                            <path
                                fill-rule="evenodd"
                                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                                clip-rule="evenodd"
                            />
                        </svg>
                        {{ form.errors.email }}
                    </div>

                    <!-- Character count and change indicator -->
                    <div class="flex justify-between items-center mt-1">
                        <div class="flex items-center text-xs">
                            <span
                                v-if="
                                    hasChanges &&
                                    originalData &&
                                    form.email?.trim().toLowerCase() !==
                                        originalData.email?.trim().toLowerCase()
                                "
                                class="flex items-center mr-2 text-orange-600 dark:text-orange-400"
                            >
                                <svg
                                    class="mr-1 w-3 h-3"
                                    fill="currentColor"
                                    viewBox="0 0 20 20"
                                >
                                    <path
                                        fill-rule="evenodd"
                                        d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                        clip-rule="evenodd"
                                    />
                                </svg>
                                Modified
                            </span>
                        </div>
                        <div class="text-gray-500 dark:text-gray-400 text-xs">
                            {{ (form.email || "").length }}/255
                        </div>
                    </div>
                </div>

                <!-- Role Field -->
                <div>
                    <FormModalLabel for="edit-role" required>
                        User Role
                    </FormModalLabel>
                    <FormModalSelect
                        id="edit-role"
                        v-model="form.role"
                        :options="roleOptions"
                        placeholder="Select user role"
                        required
                        :disabled="isFormDisabled"
                        :class="{
                            'border-red-500 focus:border-red-500':
                                form.errors.role,
                            'border-gray-300 focus:border-blue-500':
                                !form.errors.role,
                        }"
                    />

                    <!-- Role Error display -->
                    <div
                        v-if="form.errors.role"
                        class="flex items-center mt-1 text-red-600 dark:text-red-400 text-sm"
                    >
                        <svg
                            class="mr-1 w-4 h-4"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                        >
                            <path
                                fill-rule="evenodd"
                                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                                clip-rule="evenodd"
                            />
                        </svg>
                        {{ form.errors.role }}
                    </div>

                    <!-- Role change indicator -->
                    <div
                        v-if="
                            hasChanges &&
                            originalData &&
                            form.role !== originalData.role
                        "
                        class="flex items-center mt-1 text-orange-600 dark:text-orange-400 text-xs"
                    >
                        <svg
                            class="mr-1 w-3 h-3"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                        >
                            <path
                                fill-rule="evenodd"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                clip-rule="evenodd"
                            />
                        </svg>
                        Role changed from "{{ originalData.role }}" to "{{
                            form.role
                        }}"
                    </div>
                </div>

                <!-- Password Section Divider -->
                <div class="pt-4 border-gray-200 dark:border-gray-700 border-t">
                    <h4
                        class="mb-3 font-medium text-gray-900 dark:text-white text-sm"
                    >
                        Change Password (Optional)
                    </h4>
                    <p class="mb-4 text-gray-500 dark:text-gray-400 text-xs">
                        Leave these fields empty to keep the current password
                    </p>

                    <!-- Password Field -->
                    <div class="mb-4">
                        <FormModalLabel for="edit-password">
                            New Password
                        </FormModalLabel>
                        <FormModalInput
                            id="edit-password"
                            v-model="form.password"
                            type="password"
                            placeholder="Enter new password (optional)"
                            maxlength="255"
                            :disabled="isFormDisabled"
                            :class="{
                                'border-red-500 focus:border-red-500':
                                    form.errors.password,
                                'border-gray-300 focus:border-blue-500':
                                    !form.errors.password,
                            }"
                            @keydown.enter.prevent="handleSubmit"
                        />

                        <!-- Password Error display -->
                        <div
                            v-if="form.errors.password"
                            class="flex items-center mt-1 text-red-600 dark:text-red-400 text-sm"
                        >
                            <svg
                                class="mr-1 w-4 h-4"
                                fill="currentColor"
                                viewBox="0 0 20 20"
                            >
                                <path
                                    fill-rule="evenodd"
                                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                                    clip-rule="evenodd"
                                />
                            </svg>
                            {{ form.errors.password }}
                        </div>

                        <!-- Password strength indicator -->
                        <div
                            v-if="form.password"
                            class="flex justify-between items-center mt-1"
                        >
                            <div
                                class="text-gray-500 dark:text-gray-400 text-xs"
                            >
                                {{ (form.password || "").length }}/255
                            </div>
                        </div>
                    </div>

                    <!-- Password Confirmation Field -->
                    <div v-if="form.password">
                        <FormModalLabel
                            for="edit-password-confirmation"
                            required
                        >
                            Confirm New Password
                        </FormModalLabel>
                        <FormModalInput
                            id="edit-password-confirmation"
                            v-model="form.password_confirmation"
                            type="password"
                            placeholder="Confirm your new password"
                            maxlength="255"
                            required
                            :disabled="isFormDisabled"
                            :class="{
                                'border-red-500 focus:border-red-500':
                                    form.errors.password_confirmation,
                                'border-green-500 focus:border-green-500':
                                    !form.errors.password_confirmation &&
                                    form.password_confirmation &&
                                    form.password ===
                                        form.password_confirmation,
                                'border-gray-300 focus:border-blue-500':
                                    !form.errors.password_confirmation &&
                                    (!form.password_confirmation ||
                                        form.password !==
                                            form.password_confirmation),
                            }"
                            @keydown.enter.prevent="handleSubmit"
                        />

                        <!-- Password Confirmation Error display -->
                        <div
                            v-if="form.errors.password_confirmation"
                            class="flex items-center mt-1 text-red-600 dark:text-red-400 text-sm"
                        >
                            <svg
                                class="mr-1 w-4 h-4"
                                fill="currentColor"
                                viewBox="0 0 20 20"
                            >
                                <path
                                    fill-rule="evenodd"
                                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                                    clip-rule="evenodd"
                                />
                            </svg>
                            {{ form.errors.password_confirmation }}
                        </div>

                        <!-- Password match indicator -->
                        <div
                            v-else-if="form.password_confirmation"
                            class="flex items-center mt-1"
                            :class="{
                                'text-green-600 dark:text-green-400':
                                    form.password ===
                                    form.password_confirmation,
                                'text-red-600 dark:text-red-400':
                                    form.password !==
                                    form.password_confirmation,
                            }"
                        >
                            <svg
                                class="mr-1 w-4 h-4"
                                fill="currentColor"
                                viewBox="0 0 20 20"
                            >
                                <path
                                    v-if="
                                        form.password ===
                                        form.password_confirmation
                                    "
                                    fill-rule="evenodd"
                                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                    clip-rule="evenodd"
                                />
                                <path
                                    v-else
                                    fill-rule="evenodd"
                                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                                    clip-rule="evenodd"
                                />
                            </svg>
                            <span class="text-sm">
                                {{
                                    form.password === form.password_confirmation
                                        ? "Passwords match"
                                        : "Passwords do not match"
                                }}
                            </span>
                        </div>
                    </div>
                </div>
                <br />

                <!-- Tips Component -->
                <FormModalTips :tips="editTips" />

                <!-- User info display with formatted date -->
                <div
                    v-if="user"
                    class="bg-gray-50 dark:bg-gray-900/50 p-3 rounded-lg text-xs"
                >
                    <div class="space-y-2 text-gray-600 dark:text-gray-400">
                        <div class="flex justify-between items-center">
                            <span class="font-medium">User ID:</span>
                            <span class="font-mono">{{ user.id }}</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="font-medium">Current Role:</span>
                            <span class="capitalize">{{ user.role }}</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="font-medium">Created:</span>
                            <span>{{ formatDate(user.created_at) }}</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="font-medium">Last Updated:</span>
                            <span>{{ formatDate(user.updated_at) }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </FormModal>
</template>
