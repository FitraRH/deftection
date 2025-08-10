<script setup>
import { ref, watch, nextTick, computed } from "vue";
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
    createConfirmDialog,
} from "@/utils/swal";

// Props
const props = defineProps({
    isOpen: {
        type: Boolean,
        required: true,
    },
    redirectAfterCreate: {
        type: Boolean,
        default: false,
    },
});

// Emit events for parent component
const emit = defineEmits(["operation", "close"]);

// Loading state
const isLoading = ref(false);

// Role options
const roleOptions = [
    { value: "admin", label: "Admin" },
    { value: "user", label: "User" },
];

// Form initialization
const initializeForm = () => ({
    name: "",
    email: "",
    role: "user", // Default to user role
    password: "",
    password_confirmation: "",
});

// Form data
const form = useForm(initializeForm());

// Computed tips for dynamic content
const createTips = computed(() => [
    "Password must be at least 8 characters long",
    "Email must be unique in the system",
]);

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

    // Password validation
    if (!form.password) {
        errors.password = "Password is required";
    } else if (form.password.length < 8) {
        errors.password = "Password must be at least 8 characters";
    } else if (form.password.length > 255) {
        errors.password = "Password must not exceed 255 characters";
    }

    // Password confirmation validation
    if (!form.password_confirmation) {
        errors.password_confirmation = "Password confirmation is required";
    } else if (form.password !== form.password_confirmation) {
        errors.password_confirmation = "Passwords do not match";
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

// Reset form completely
const resetForm = () => {
    form.reset();
    form.clearErrors();
    // Force reactivity update
    nextTick(() => {
        Object.assign(form, initializeForm());
    });
};

// Watch for modal open/close to reset form
watch(
    () => props.isOpen,
    (isOpen) => {
        if (isOpen) {
            resetForm(); // Reset when modal opens
        }
    }
);

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

    // Show create confirmation dialog
    const result = await createConfirmDialog("this user");
    if (!result.isConfirmed) {
        return; // User cancelled
    }

    isLoading.value = true;

    // Transform data before sending
    const formData = {
        name: form.name?.trim(),
        email: form.email?.trim().toLowerCase(),
        role: form.role,
        password: form.password,
        password_confirmation: form.password_confirmation,
    };

    form.transform(() => formData).post(route("users.store"), {
        preserveScroll: true,
        preserveState: true,
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
    // Emit success event for parent component
    emit("operation", {
        type: "success",
        action: "create",
        data: response,
        message: "User created successfully",
    });

    // Show success toast using utils
    successToast("User created successfully!");
};

// Handle error response
const handleError = (errors) => {
    console.error("Form submission errors:", errors);

    // Emit error event for parent component
    emit("operation", {
        type: "error",
        action: "create",
        error: errors,
        message: "Failed to create user",
    });

    // Show error toast using utils
    errorToast("Failed to create user. Please check your input.");
};

// Handle operation finish
const handleFinish = () => {
    emit("operation", {
        type: "finish",
        action: "create",
        message: "Create operation completed",
    });
};

// Close modal
const closeModal = () => {
    if (!isLoading.value) {
        emit("close");
        // Reset form after a short delay to ensure smooth modal close animation
        setTimeout(resetForm, 150);
    }
};

// Handle modal close with unsaved changes check using utils
const handleClose = async () => {
    if (isLoading.value) return;

    const isDirty = form.name?.trim() || form.email?.trim() || form.password;
    if (isDirty) {
        const result = await unsavedChangesDialog();
        if (result.isConfirmed) {
            closeModal();
        }
    } else {
        closeModal();
    }
};
</script>

<template>
    <FormModal
        :is-open="isOpen"
        :is-loading="isLoading"
        title="Add User"
        description="Fill the user information below and click add to save changes."
        submit-text="Add User"
        loading-text="Creating User..."
        :has-errors="form.hasErrors"
        focus="name"
        :prevent-close-on-outside-click="isLoading"
        :show-close-button="!isLoading"
        @close="handleClose"
        @submit="handleSubmit"
    >
        <div class="space-y-4">
            <!-- Name Field -->
            <div>
                <FormModalLabel for="name" required> Full Name </FormModalLabel>
                <FormModalInput
                    id="name"
                    ref="nameInput"
                    v-model="form.name"
                    type="text"
                    placeholder="Enter full name (e.g., John Doe)"
                    maxlength="255"
                    required
                    :disabled="isLoading"
                    :class="{
                        'border-red-500 focus:border-red-500': form.errors.name,
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

                <!-- Character count -->
                <div class="flex justify-end mt-1">
                    <div class="text-gray-500 dark:text-gray-400 text-xs">
                        {{ (form.name || "").length }}/255
                    </div>
                </div>
            </div>

            <!-- Email Field -->
            <div>
                <FormModalLabel for="email" required>
                    Email Address
                </FormModalLabel>
                <FormModalInput
                    id="email"
                    v-model="form.email"
                    type="email"
                    placeholder="Enter email address (e.g., john@example.com)"
                    maxlength="255"
                    required
                    :disabled="isLoading"
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

                <!-- Character count -->
                <div class="flex justify-end mt-1">
                    <div class="text-gray-500 dark:text-gray-400 text-xs">
                        {{ (form.email || "").length }}/255
                    </div>
                </div>
            </div>

            <!-- Role Field -->
            <div>
                <FormModalLabel for="role" required> User Role </FormModalLabel>
                <FormModalSelect
                    id="role"
                    v-model="form.role"
                    :options="roleOptions"
                    placeholder="Select user role"
                    required
                    :disabled="isLoading"
                    :class="{
                        'border-red-500 focus:border-red-500': form.errors.role,
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
            </div>

            <!-- Password Field -->
            <div>
                <FormModalLabel for="password" required>
                    Password
                </FormModalLabel>
                <FormModalInput
                    id="password"
                    v-model="form.password"
                    type="password"
                    placeholder="Enter a secure password"
                    maxlength="255"
                    required
                    :disabled="isLoading"
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

                <!-- Character count -->
                <div class="flex justify-end mt-1">
                    <div class="text-gray-500 dark:text-gray-400 text-xs">
                        {{ (form.password || "").length }}/255
                    </div>
                </div>
            </div>

            <!-- Password Confirmation Field -->
            <div>
                <FormModalLabel for="password_confirmation" required>
                    Confirm Password
                </FormModalLabel>
                <FormModalInput
                    id="password_confirmation"
                    v-model="form.password_confirmation"
                    type="password"
                    placeholder="Confirm your password"
                    maxlength="255"
                    required
                    :disabled="isLoading"
                    :class="{
                        'border-red-500 focus:border-red-500':
                            form.errors.password_confirmation,
                        'border-green-500 focus:border-green-500':
                            !form.errors.password_confirmation &&
                            form.password_confirmation &&
                            form.password === form.password_confirmation,
                        'border-gray-300 focus:border-blue-500':
                            !form.errors.password_confirmation &&
                            (!form.password_confirmation ||
                                form.password !== form.password_confirmation),
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
                            form.password === form.password_confirmation,
                        'text-red-600 dark:text-red-400':
                            form.password !== form.password_confirmation,
                    }"
                >
                    <svg
                        class="mr-1 w-4 h-4"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                    >
                        <path
                            v-if="form.password === form.password_confirmation"
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

            <!-- Tips Component -->
            <FormModalTips :tips="createTips" />
        </div>
    </FormModal>
</template>
