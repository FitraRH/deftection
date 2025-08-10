<script setup>
import { ref, computed, onMounted, watch } from "vue";
import { router } from "@inertiajs/vue3";
import { useDebounceFn } from "@vueuse/core";
import { usePolling } from "@/composables/usePolling";
import { route } from "ziggy-js";
import { useForm } from "@inertiajs/vue3";
import { successToast, errorToast, deleteConfirmDialog } from "@/utils/swal";

// Import child components
import Table from "../../../Shared/Table/Table.vue";
import Tablewrapper from "../../../Shared/Table/Tablewrapper.vue";
import TableToolbar from "../../../Shared/Table/TableToolbar.vue";
import TableContainer from "../../../Shared/Table/TableContainer.vue";
import ToolbarSearch from "../../../Shared/TableToolbar/ToolbarSearch.vue";
import TableHeaderCell from "../../../Shared/Table/TableHeaderCell.vue";
import ForceRefreshButton from "../../../Shared/Table/ForceRefreshButton.vue";
import LiveMonitorToggle from "../../../Shared/Table/LiveMonitorToggle.vue";
import FilterResetButton from "../../../Shared/TableToolbar/FilterResetButton.vue";
import TableCell from "../../../Shared/Table/TableCell.vue";
import RowNotFound from "../../../Shared/Table/RowNotFound.vue";
import TableFooter from "../../../Shared/Table/TableFooter.vue";
import EllipsisDropdown from "../../../Shared/Dropdown/EllipsisDropdown.vue";
import CreateButton from "../../../Shared/TableToolbar/CreateButton.vue";
import ProductCreateModal from "./Components/ProductCreateModal.vue";
import EditItem from "../../../Shared/Dropdown/EditItem.vue";
import ProductEditModal from "./Components/ProductEditModal.vue";
import DeleteItem from "../../../Shared/Dropdown/DeleteItem.vue";

const props = defineProps({
    products: { type: Array, required: true },
    filters: { type: Object, required: true },
    meta: { type: Object, default: () => ({}) },
    initialChecksum: { type: String, default: "" },
});

// Theme store for SweetAlert styling - removed as swal utils handle this

// Use props directly for navigation, reactive refs for polling updates
const currentFilters = computed(() => props.filters.current || {});
const filterOptions = computed(() => props.filters.options || {});
const searchTerm = ref(currentFilters.value.search || "");

// For polling updates only
const pollingProducts = ref([]);
const pollingFilters = ref({});
const pollingMeta = ref({});

// Modal state management
const isAnyModalOpen = ref(false);
const openModals = ref(new Set()); // Track multiple modals if needed

// Specific modal states
const isCreateModalOpen = ref(false);
const isEditModalOpen = ref(false);
const editProductSlug = ref(null);

// Delete state
const isDeletingProduct = ref(false);

// Loading states for UI control
const isTableLoading = ref(false);

// Determine which data to show: polling data if available, otherwise props
const displayProducts = computed(() => {
    return pollingProducts.value.length > 0
        ? pollingProducts.value
        : props.products;
});

const displayMeta = computed(() => {
    return Object.keys(pollingMeta.value).length > 0
        ? pollingMeta.value
        : props.meta;
});

const displayFilterOptions = computed(() => {
    return Object.keys(pollingFilters.value).length > 0
        ? pollingFilters.value
        : filterOptions.value;
});

// Check if any operations are in progress
const isAnyOperationInProgress = computed(() => {
    return (
        isTableLoading.value ||
        isPollingLoading.value ||
        isDeletingProduct.value
    );
});

// CLEAN REUSABLE POLLING with modal awareness
const {
    isPolling,
    isLoading: isPollingLoading,
    error: pollingError,
    isManuallyPaused,
    canPoll,
    isOnline,
    visibility,
    togglePolling,
    forceRefresh,
    initializeChecksum,
    getStatus,
} = usePolling({
    // Custom endpoints for this component
    checkUrl: route("products.index.check"),
    dataUrl: route("products.index.api"),
    forceRefreshUrl: route("products.index.refresh"),

    // Polling configuration
    interval: 20000,
    enabled: true,
    pauseOnTabHidden: true,
    pauseOnOffline: true,
    debug: false,
    retryAttempts: 3,
    retryDelay: 10000,

    // Custom pause condition - pause when modal is open
    customPauseCondition: () => {
        // Pause polling when any modal is open
        const shouldPause = isAnyModalOpen.value;
        if (shouldPause) {
            console.log("Polling paused: Modal is open");
        }
        return !shouldPause; // Return true to allow polling, false to pause
    },

    // Callbacks
    onDataUpdate: (data) => {
        // Only update polling state, don't interfere with navigation
        pollingProducts.value = data.products || [];
        pollingFilters.value = data.filters?.options || {};
        pollingMeta.value = data.meta || {};

        console.log("Polling detected changes and updated data!");
    },
    onError: (error) => {
        console.error("Polling error:", error);
        // Could add toast notifications, etc.
    },
});

// Watch search term to update from current filters
watch(
    () => currentFilters.value.search,
    (newSearch) => {
        searchTerm.value = newSearch || "";
    }
);

// Initialize component
onMounted(() => {
    initializeChecksum(props.initialChecksum);
});

// Computed properties
const hasActiveFilters = computed(() => searchTerm.value.length > 0);

// Modal state management functions
const handleModalOpen = (modalId = "default") => {
    openModals.value.add(modalId);
    isAnyModalOpen.value = true;
};

const handleModalClose = (modalId = "default") => {
    openModals.value.delete(modalId);
    isAnyModalOpen.value = openModals.value.size > 0;
    console.log(
        `Modal ${modalId} closed. Polling will ${
            isAnyModalOpen.value ? "remain paused" : "resume"
        }.`
    );
};

// Create modal specific handlers
const handleCreateModalOpen = () => {
    isCreateModalOpen.value = true;
    handleModalOpen("create");
};

const handleCreateModalClose = () => {
    isCreateModalOpen.value = false;
    handleModalClose("create");
};

// Edit modal specific handlers
const handleEditModalOpen = (slug) => {
    editProductSlug.value = slug;
    isEditModalOpen.value = true;
    handleModalOpen(`edit-${slug}`);
};

const handleEditModalClose = () => {
    const slug = editProductSlug.value;
    isEditModalOpen.value = false;
    editProductSlug.value = null;
    handleModalClose(`edit-${slug}`);
};

// Delete functionality
const handleDeleteProduct = async (product) => {
    // Show confirmation dialog using clean swal utils
    const result = await deleteConfirmDialog(`"${product.name}"`);

    if (result.isConfirmed) {
        await performDelete(product);
    }
};

// Perform the actual deletion
const performDelete = async (product) => {
    isDeletingProduct.value = true;

    const form = useForm({});

    form.delete(route("products.destroy", product.slug), {
        preserveState: true,
        preserveScroll: true,
        onSuccess: () => {
            handleDeleteSuccess(product.name);
        },
        onError: (errors) => {
            handleDeleteError(errors);
        },
        onFinish: () => {
            isDeletingProduct.value = false;
        },
    });
};

// Handle successful deletion
const handleDeleteSuccess = (productName) => {
    console.log("Product deleted successfully:", productName);

    successToast("Product deleted successfully!");
};

// Handle deletion error
const handleDeleteError = (errors) => {
    console.error("Delete operation failed:", errors);

    let errorMessage =
        "An unexpected error occurred while deleting the product.";

    // Check for specific error messages
    if (errors.message) {
        errorMessage = errors.message;
    } else if (errors.error) {
        errorMessage = errors.error;
    }

    errorToast(errorMessage);
};

// Build filter parameters helper
const buildFilterParams = (customParams = {}) => {
    const params = {
        search: searchTerm.value || undefined,
        sort_by: currentFilters.value.sortBy || undefined,
        sort_dir: currentFilters.value.sortDir || undefined,
        per_page: currentFilters.value.perPage || undefined,
        page: 1,
        ...customParams,
    };

    // Remove undefined values
    Object.keys(params).forEach((key) => {
        if (params[key] === undefined) delete params[key];
    });

    return params;
};

// Navigate with filters helper
const navigateWithFilters = (customParams = {}) => {
    // Set loading state
    isTableLoading.value = true;

    // Clear polling data to show fresh server data
    pollingProducts.value = [];
    pollingFilters.value = {};
    pollingMeta.value = {};

    router.get(route("products.index"), buildFilterParams(customParams), {
        preserveState: true,
        preserveScroll: true,
        replace: true,
        only: ["products", "filters", "meta", "initialChecksum"],
        onSuccess: (page) => {
            // Update checksum for polling
            if (page.props.initialChecksum) {
                initializeChecksum(page.props.initialChecksum);
            }

            console.log("Navigation successful, data updated");
        },
        onError: (errors) => {
            console.error("Navigation failed:", errors);
        },
        onFinish: () => {
            isTableLoading.value = false;
        },
    });
};

// Main filter application
const applyFilters = () => navigateWithFilters();

// Filter methods
const resetAllFilters = () => {
    searchTerm.value = "";
    applyFilters();
};

// Search and filter change handlers
const debouncedSearch = useDebounceFn(applyFilters, 300);
const handleSearchChange = () => {
    if (!isAnyOperationInProgress.value) {
        debouncedSearch();
    }
};

// Pagination and sorting
const goToPage = (page) => {
    if (!isAnyOperationInProgress.value) {
        navigateWithFilters({ page });
    }
};

const changePerPage = (perPage) => {
    if (!isAnyOperationInProgress.value) {
        navigateWithFilters({ per_page: perPage, page: 1 });
    }
};

const sortBy = (column) => {
    if (!isAnyOperationInProgress.value) {
        const sortDir =
            currentFilters.value.sortBy === column
                ? currentFilters.value.sortDir === "asc"
                    ? "desc"
                    : "asc"
                : "asc";
        navigateWithFilters({ sort_by: column, sort_dir: sortDir, page: 1 });
    }
};

// Event handlers for child components
const handleProductOperation = ({ type, action, data, error, message }) => {
    console.log(`Product ${action} operation:`, { type, message });

    switch (type) {
        case "success":
            switch (action) {
                case "create":
                    console.log("Product created successfully:", data);
                    break;

                case "update":
                    console.log("Product updated successfully:", data);
                    break;

                case "fetch":
                    console.log("Product fetched successfully:", data);
                    break;
            }
            break;

        case "error":
            switch (action) {
                case "create":
                    console.error("Failed to create product:", error);
                    break;

                case "update":
                    console.error("Failed to update product:", error);
                    break;

                case "fetch":
                    console.error("Failed to fetch product:", error);
                    break;
            }
            break;

        case "finish":
            switch (action) {
                case "create":
                    console.log("Create operation finished");
                    break;

                case "update":
                    console.log("Update operation finished");
                    break;
            }
            break;

        default:
            console.warn("Unknown operation type:", type);
    }
};
</script>

<template>
    <Tablewrapper>
        <div class="flex justify-between items-center mb-4">
            <!-- Polling Toggle Button -->
            <LiveMonitorToggle
                :isPolling="isPolling"
                :isPollingLoading="isPollingLoading"
                :isManuallyPaused="isManuallyPaused"
                :isOnline="isOnline"
                :visibility="visibility"
                :disabled="isAnyModalOpen"
                @click="togglePolling"
            />
            <!-- Force Refresh Button  -->
            <ForceRefreshButton
                :isPollingLoading="isPollingLoading"
                :isOnline="isOnline"
                :disabled="isAnyModalOpen || isTableLoading"
                @click="forceRefresh"
            />
        </div>

        <TableToolbar title="Products Table" description="">
            <template #left>
                <ToolbarSearch
                    v-model="searchTerm"
                    placeholder="Search product name or slug..."
                    :disabled="isAnyOperationInProgress"
                    @input="handleSearchChange"
                />

                <FilterResetButton
                    v-if="hasActiveFilters"
                    label="Reset Filters"
                    :disabled="isAnyOperationInProgress"
                    @click="resetAllFilters"
                />
            </template>

            <template #right>
                <CreateButton
                    v-if="isOnline"
                    label="Add Product"
                    title="Add Product"
                    :disabled="isAnyOperationInProgress"
                    @click="handleCreateModalOpen"
                />
            </template>
        </TableToolbar>

        <TableContainer>
            <!-- Loading overlay -->
            <div
                v-if="isTableLoading"
                class="z-10 absolute inset-0 flex justify-center items-center bg-white/10 dark:bg-gray-900/10"
            >
                <div
                    class="flex items-center space-x-2 text-gray-600 dark:text-gray-300"
                >
                    <svg
                        class="w-5 h-5 animate-spin"
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
                    <span>Loading...</span>
                </div>
            </div>

            <Table>
                <template #head>
                    <tr class="group/tr table-tr">
                        <TableHeaderCell
                            v-for="header in [
                                {
                                    label: 'Name',
                                    field: 'name',
                                    sortable: true,
                                },
                                {
                                    label: 'Slug',
                                    field: 'slug',
                                    sortable: true,
                                },
                                {
                                    label: 'Created At',
                                    field: 'created_at',
                                    sortable: true,
                                },
                                {
                                    label: 'Updated At',
                                    field: 'updated_at',
                                    sortable: true,
                                },
                                { label: 'Actions', sortable: false },
                            ]"
                            :key="header.label"
                            :label="header.label"
                            :is-sortable="
                                header.sortable && !isAnyOperationInProgress
                            "
                            :is-active="currentFilters.sortBy === header.field"
                            :sort-dir="currentFilters.sortDir"
                            :disabled="isAnyOperationInProgress"
                            @click="header.sortable && sortBy(header.field)"
                        />
                    </tr>
                </template>
                <template #body>
                    <tr
                        v-for="product in displayProducts"
                        v-if="displayProducts.length > 0"
                        :key="product.slug || product.id || Math.random()"
                        class="group/tr table-tr relative border-y border-transparent border-b-gray-200 dark:border-b-dark-500"
                        :class="{ 'opacity-50': isTableLoading }"
                    >
                        <!-- Name Column -->
                        <TableCell>
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100 capitalize"
                            >
                                {{ product?.name || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Slug Column -->
                        <TableCell>
                            <p class="text-gray-800 dark:text-dark-100">
                                {{ product?.slug || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Created At Column -->
                        <TableCell>
                            <p class="text-gray-800 dark:text-dark-100">
                                {{ product?.created_at || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Updated At Column -->
                        <TableCell>
                            <p class="text-gray-800 dark:text-dark-100">
                                {{ product?.updated_at || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Actions Column -->
                        <TableCell>
                            <EllipsisDropdown
                                :disabled="isAnyOperationInProgress"
                            >
                                <EditItem
                                    label="Edit Product"
                                    title="Edit Product"
                                    :disabled="isAnyOperationInProgress"
                                    @click="handleEditModalOpen(product.slug)"
                                />
                                <DeleteItem
                                    label="Delete Product"
                                    title="Delete Product"
                                    :disabled="isAnyOperationInProgress"
                                    @click="handleDeleteProduct(product)"
                                />
                            </EllipsisDropdown>
                        </TableCell>
                    </tr>

                    <RowNotFound
                        v-if="displayProducts.length === 0 && !isTableLoading"
                        label="No products found"
                    />
                </template>
            </Table>

            <TableFooter
                :meta="displayMeta"
                :currentFilters="currentFilters"
                :perPageOptions="
                    displayFilterOptions.perPageOptions || [5, 10, 25, 50]
                "
                :disabled="isAnyOperationInProgress"
                @goToPage="goToPage"
                @changePerPage="changePerPage"
            />
        </TableContainer>

        <!-- Product Create Modal - Teleported to body -->
        <Teleport to="body">
            <ProductCreateModal
                :is-open="isCreateModalOpen"
                @close="handleCreateModalClose"
                @operation="handleProductOperation"
            />
        </Teleport>

        <!-- Product Edit Modal - Teleported to body -->
        <Teleport to="body">
            <ProductEditModal
                v-if="editProductSlug"
                :is-open="isEditModalOpen"
                :slug="editProductSlug"
                @close="handleEditModalClose"
                @operation="handleProductOperation"
            />
        </Teleport>
    </Tablewrapper>
</template>
