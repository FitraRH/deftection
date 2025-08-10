<script setup>
import { ref, watch, nextTick, computed } from "vue";
import FormModal from "@/Shared/Modals/FormModal.vue";
import FormModalLabel from "@/Shared/Modals/FormModalLabel.vue";
import FormModalInput from "@/Shared/Modals/FormModalInput.vue";
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

// Form initialization
const initializeForm = () => ({
    name: "",
});

// Form data
const form = useForm(initializeForm());

// Computed tips for dynamic content
const createTips = computed(() => [
    "A URL-friendly slug will be automatically generated",
    "Press Ctrl+Enter to quickly submit",
]);

// Validation rules
const validateForm = () => {
    const errors = {};

    if (!form.name?.trim()) {
        errors.name = "Product name is required";
    } else if (form.name.trim().length < 2) {
        errors.name = "Product name must be at least 2 characters";
    } else if (form.name.trim().length > 255) {
        errors.name = "Product name must not exceed 255 characters";
    }

    return errors;
};

// Clear errors when name changes
watch(
    () => form.name,
    (newValue) => {
        if (form.errors.name && newValue?.trim()) {
            form.clearErrors("name");
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
    const result = await createConfirmDialog("this product");
    if (!result.isConfirmed) {
        return; // User cancelled
    }

    isLoading.value = true;

    // Transform data before sending
    const formData = {
        name: form.name?.trim(),
    };

    form.transform(() => formData).post(route("products.store"), {
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
        message: "Product created successfully",
    });

    // Show success toast using utils
    successToast("Product created successfully!");
};

// Handle error response
const handleError = (errors) => {
    console.error("Form submission errors:", errors);

    // Emit error event for parent component
    emit("operation", {
        type: "error",
        action: "create",
        error: errors,
        message: "Failed to create product",
    });

    // Show error toast using utils
    errorToast("Failed to create product. Please check your input.");
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

    const isDirty = form.name?.trim();
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
        title="Add Product"
        description="Fill the product information below and click add to save changes."
        submit-text="Add Product"
        loading-text="Creating Product..."
        :has-errors="form.hasErrors"
        focus="name"
        :prevent-close-on-outside-click="isLoading"
        :show-close-button="!isLoading"
        @close="handleClose"
        @submit="handleSubmit"
    >
        <div class="space-y-4">
            <div>
                <FormModalLabel for="name" required>
                    Product Name
                </FormModalLabel>
                <FormModalInput
                    id="name"
                    ref="nameInput"
                    v-model="form.name"
                    type="text"
                    placeholder="Enter product name (e.g., Premium Coffee Beans)"
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

                <!-- Error display -->
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

            <!-- Tips Component -->
            <FormModalTips :tips="createTips" />
        </div>
    </FormModal>
</template>
