<script setup>
import { ref, watch, nextTick, computed, onMounted } from "vue";
import FormModal from "@/Shared/Modals/FormModal.vue";
import FormModalLabel from "@/Shared/Modals/FormModalLabel.vue";
import FormModalInput from "@/Shared/Modals/FormModalInput.vue";
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
    slug: {
        type: String,
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

// Product data
const product = ref(null);
const originalData = ref(null);

// Form initialization
const initializeForm = () => ({
    name: "",
});

// Form data
const form = useForm(initializeForm());

// Computed properties
const hasChanges = computed(() => {
    if (!originalData.value) return false;
    return form.name?.trim() !== originalData.value.name?.trim();
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
        "The slug will be automatically updated when you change the name",
        "Press Ctrl+Enter to quickly submit changes",
    ];

    // Add original name tip if we have original data
    if (originalData.value?.name) {
        tips.push({
            text: `Original name: "${originalData.value.name}"`,
            condition: hasChanges.value, // Only show when there are changes
        });
    }

    return tips;
});

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

// Fetch product data
const fetchProduct = async () => {
    if (isFetching.value) {
        console.warn("Fetch already in progress, skipping...");
        return;
    }

    if (!props.slug) {
        console.error("No slug provided");
        return;
    }

    isFetching.value = true;

    try {
        const url = route("products.show.api", props.slug);
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: Failed to fetch product`);
        }

        const res = await response.json();

        if (!res.success || !res.data) {
            throw new Error(
                "Invalid response format - missing success or data"
            );
        }

        // Set product data
        product.value = res.data;
        originalData.value = { ...res.data };

        // Pre-fill form with existing data
        await nextTick();
        form.name = res.data.name;
        form.clearErrors();

        console.log("Product fetched successfully:", res.data);

        // Emit success event
        emit("operation", {
            type: "success",
            action: "fetch",
            data: res.data,
            message: "Product data loaded successfully",
        });
    } catch (error) {
        console.error("Error fetching product:", error);

        // Emit error event
        emit("operation", {
            type: "error",
            action: "fetch",
            error,
            message: `Failed to fetch product data: ${error.message}`,
        });

        // Show error toast
        errorToast(`Failed to fetch product data: ${error.message}`);

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
        if (newValue && props.slug) {
            initializeAndFetch();
        }
    }
);

watch(
    () => props.slug,
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
    product.value = null;
    originalData.value = null;

    // Fetch new data
    await fetchProduct();
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
    successToast("Product data refreshed!");
};

// Mount handler
onMounted(() => {
    // If modal is already open when mounted, fetch immediately
    if (props.isOpen && props.slug) {
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
    const result = await editConfirmDialog("this product");
    if (!result.isConfirmed) {
        return;
    }

    isLoading.value = true;

    // Transform data before sending
    const formData = {
        name: form.name?.trim(),
    };

    form.transform(() => formData).put(route("products.update", props.slug), {
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
        message: "Product updated successfully",
    });

    successToast("Product updated successfully!");
};

// Handle error response
const handleError = (errors) => {
    console.error("Form submission errors:", errors);

    emit("operation", {
        type: "error",
        action: "update",
        error: errors,
        message: "Failed to update product",
    });

    errorToast("Failed to update product. Please check your input.");
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
            product.value = null;
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
    if (isFetching.value) return "Loading product data...";
    if (isLoading.value) return "Updating product...";
    return "";
};
</script>

<template>
    <FormModal
        :is-open="isOpen"
        :is-loading="isFormDisabled"
        title="Edit Product"
        description="Update the product information below and click save to apply changes."
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
                Loading product data...
            </div>

            <!-- Form content -->
            <div v-else>
                <div>
                    <FormModalLabel for="edit-name" required>
                        Product Name
                    </FormModalLabel>
                    <FormModalInput
                        id="edit-name"
                        ref="nameInput"
                        v-model="form.name"
                        type="text"
                        placeholder="Enter product name (e.g., Premium Coffee Beans)"
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

                    <!-- Character count and change indicator -->
                    <div class="flex justify-between items-center mt-1">
                        <div class="flex items-center text-xs">
                            <span
                                v-if="hasChanges"
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

                <!-- Tips Component -->
                <FormModalTips :tips="editTips" />

                <!-- Product info display with formatted date -->
                <div
                    v-if="product"
                    class="bg-gray-50 dark:bg-gray-900/50 p-3 rounded-lg text-xs"
                >
                    <div class="space-y-2 text-gray-600 dark:text-gray-400">
                        <div class="flex justify-between items-center">
                            <span class="font-medium">Current Slug:</span>
                            <span class="font-mono">{{ product.slug }}</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="font-medium">Created:</span>
                            <span>{{ formatDate(product.created_at) }}</span>
                        </div>
                        <div class="flex justify-between items-center">
                            <span class="font-medium">Last Updated:</span>
                            <span>{{ formatDate(product.updated_at) }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </FormModal>
</template>
