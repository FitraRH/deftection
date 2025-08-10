<script setup>
import { ref, computed, onMounted, watch } from "vue";
import { router } from "@inertiajs/vue3";
import { useDebounceFn } from "@vueuse/core";
import { usePolling } from "@/composables/usePolling";
import { route } from "ziggy-js";
import { useForm } from "@inertiajs/vue3";
import { successToast, errorToast, deleteConfirmDialog } from "@/utils/swal";
import { Popover } from "@headlessui/vue";

// Import child components
import Table from "../../../Shared/Table/Table.vue";
import Tablewrapper from "../../../Shared/Table/Tablewrapper.vue";
import TableToolbar from "../../../Shared/Table/TableToolbar.vue";
import TableContainer from "../../../Shared/Table/TableContainer.vue";
import ToolbarSearch from "../../../Shared/TableToolbar/ToolbarSearch.vue";
import TableHeaderCell from "../../../Shared/Table/TableHeaderCell.vue";
import ForceRefreshButton from "../../../Shared/Table/ForceRefreshButton.vue";
import LiveMonitorToggle from "../../../Shared/Table/LiveMonitorToggle.vue";
import FilterPopoverButton from "../../../Shared/Popover/FilterPopoverButton.vue";
import FilterPopoverPanel from "../../../Shared/Popover/FilterPopoverPanel.vue";
import FilterResetButton from "../../../Shared/TableToolbar/FilterResetButton.vue";
import FilterPopoverCheckbox from "../../../Shared/Popover/FilterPopoverCheckbox.vue";
import TableCell from "../../../Shared/Table/TableCell.vue";
import RowNotFound from "../../../Shared/Table/RowNotFound.vue";
import TableFooter from "../../../Shared/Table/TableFooter.vue";
import EllipsisDropdown from "../../../Shared/Dropdown/EllipsisDropdown.vue";
import CreateButton from "../../../Shared/TableToolbar/CreateButton.vue";
import UserCreateModal from "./Components/UserCreateModal.vue";
import EditItem from "../../../Shared/Dropdown/EditItem.vue";
import UserEditModal from "./Components/UserEditModal.vue";
import DeleteItem from "../../../Shared/Dropdown/DeleteItem.vue";

const props = defineProps({
    users: { type: Array, required: true },
    filters: { type: Object, required: true },
    meta: { type: Object, default: () => ({}) },
    initialChecksum: { type: String, default: "" },
});

// Use props directly for navigation, reactive refs for polling updates
const currentFilters = computed(() => props.filters.current || {});
const filterOptions = computed(() => props.filters.options || {});
const searchTerm = ref(currentFilters.value.search || "");

// For polling updates only
const pollingUsers = ref([]);
const pollingFilters = ref({});
const pollingMeta = ref({});

// Modal state management
const isAnyModalOpen = ref(false);
const openModals = ref(new Set()); // Track multiple modals if needed

// Specific modal states
const isCreateModalOpen = ref(false);
const isEditModalOpen = ref(false);
const editUserId = ref(null);

// Delete state
const isDeletingUser = ref(false);

// Loading states for UI control
const isTableLoading = ref(false);

// Determine which data to show: polling data if available, otherwise props
const displayUsers = computed(() => {
    return pollingUsers.value.length > 0 ? pollingUsers.value : props.users;
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
        isTableLoading.value || isPollingLoading.value || isDeletingUser.value
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
    checkUrl: route("users.index.check"),
    dataUrl: route("users.index.api"),
    forceRefreshUrl: route("users.index.refresh"),

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
        pollingUsers.value = data.users || [];
        pollingFilters.value = data.filters?.options || {};
        pollingMeta.value = data.meta || {};

        // Reinitialize filter options with new data
        initializeFilters();

        console.log("Polling detected changes and updated data!");
    },
    onError: (error) => {
        console.error("Polling error:", error);
        // Could add toast notifications, etc.
    },
});

// Filter options
const roleOptions = ref([]);

// Initialize filter options
const initializeFilters = () => {
    const options = displayFilterOptions.value;

    const mapOptions = (options, selectedKey) =>
        options?.map((option) => ({
            ...option,
            selected: (currentFilters.value[selectedKey] || []).includes(
                option.value
            ),
        })) || [];

    roleOptions.value = mapOptions(options.roles, "roles");
};

// Watch for prop changes to reinitialize filters
watch(
    () => props.filters,
    () => {
        initializeFilters();
    },
    { deep: true }
);

// Watch search term to update from current filters
watch(
    () => currentFilters.value.search,
    (newSearch) => {
        searchTerm.value = newSearch || "";
    }
);

// Initialize component
onMounted(() => {
    initializeFilters();
    initializeChecksum(props.initialChecksum);
});

// Computed properties
const selectedRolesCount = computed(
    () => roleOptions.value.filter((option) => option.selected).length
);
const hasActiveFilters = computed(
    () => selectedRolesCount.value > 0 || searchTerm.value.length > 0
);
const selectedRoles = computed(() =>
    roleOptions.value
        .filter((option) => option.selected)
        .map((option) => option.value)
);

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
const handleEditModalOpen = (userId) => {
    editUserId.value = userId;
    isEditModalOpen.value = true;
    handleModalOpen(`edit-${userId}`);
};

const handleEditModalClose = () => {
    const userId = editUserId.value;
    isEditModalOpen.value = false;
    editUserId.value = null;
    handleModalClose(`edit-${userId}`);
};

// Delete functionality
const handleDeleteUser = async (user) => {
    // Show confirmation dialog using clean swal utils
    const result = await deleteConfirmDialog(`"${user.name}"`);

    if (result.isConfirmed) {
        await performDelete(user);
    }
};

// Perform the actual deletion
const performDelete = async (user) => {
    isDeletingUser.value = true;

    const form = useForm({});

    form.delete(route("users.destroy", user.id), {
        preserveState: true,
        preserveScroll: true,
        onSuccess: () => {
            handleDeleteSuccess(user.name);
        },
        onError: (errors) => {
            handleDeleteError(errors);
        },
        onFinish: () => {
            isDeletingUser.value = false;
        },
    });
};

// Handle successful deletion
const handleDeleteSuccess = (userName) => {
    console.log("User deleted successfully:", userName);

    successToast("User deleted successfully!");
};

// Handle deletion error
const handleDeleteError = (errors) => {
    console.error("Delete operation failed:", errors);

    let errorMessage = "An unexpected error occurred while deleting the user.";

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
        roles: selectedRoles.value.length > 0 ? selectedRoles.value : undefined,
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
    pollingUsers.value = [];
    pollingFilters.value = {};
    pollingMeta.value = {};

    router.get(route("users.index"), buildFilterParams(customParams), {
        preserveState: true,
        preserveScroll: true,
        replace: true,
        only: ["users", "filters", "meta", "initialChecksum"],
        onSuccess: (page) => {
            // Update checksum for polling
            if (page.props.initialChecksum) {
                initializeChecksum(page.props.initialChecksum);
            }

            // Reinitialize filters with new data
            setTimeout(() => {
                initializeFilters();
            }, 50);

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
const clearFilter = (optionsRef) => {
    optionsRef.value.forEach((option) => (option.selected = false));
    applyFilters();
};

const clearRoleFilters = () => clearFilter(roleOptions);

const resetAllFilters = () => {
    clearFilter(roleOptions);
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
const handleFilterChange = () => {
    if (!isAnyOperationInProgress.value) {
        setTimeout(applyFilters, 10);
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

// Get role badge class
const getRoleBadgeClass = (role) => {
    const classes = {
        admin: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
        user: "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300",
    };
    return (
        classes[role] ||
        "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300"
    );
};

// Event handlers for child components
const handleUserOperation = ({ type, action, data, error, message }) => {
    console.log(`User ${action} operation:`, { type, message });

    switch (type) {
        case "success":
            switch (action) {
                case "create":
                    console.log("User created successfully:", data);
                    break;

                case "update":
                    console.log("User updated successfully:", data);
                    break;

                case "fetch":
                    console.log("User fetched successfully:", data);
                    break;
            }
            break;

        case "error":
            switch (action) {
                case "create":
                    console.error("Failed to create user:", error);
                    break;

                case "update":
                    console.error("Failed to update user:", error);
                    break;

                case "fetch":
                    console.error("Failed to fetch user:", error);
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

        <TableToolbar
            title="Users Table"
            description="Manage user accounts, roles, and permissions"
        >
            <template #left>
                <ToolbarSearch
                    v-model="searchTerm"
                    placeholder="Search user name or email..."
                    :disabled="isAnyOperationInProgress"
                    @input="handleSearchChange"
                />

                <!-- Role Filter -->
                <Popover class="relative">
                    <FilterPopoverButton
                        label="Role"
                        :selected-options="selectedRolesCount"
                        :disabled="isAnyOperationInProgress"
                    />
                    <FilterPopoverPanel
                        :show-clear-button="selectedRolesCount > 0"
                        :selected-options="selectedRolesCount"
                        @click-clear="clearRoleFilters"
                    >
                        <FilterPopoverCheckbox
                            v-for="role in roleOptions"
                            :key="role.value"
                            v-model:checked="role.selected"
                            :id="`role-${role.value}`"
                            :value="role.value"
                            :label="role.label"
                            :count="role.count"
                            :disabled="isAnyOperationInProgress"
                            @change="handleFilterChange"
                        />
                        <div
                            v-if="roleOptions.length === 0"
                            class="px-3 py-2 text-gray-500 text-xs"
                        >
                            No roles found
                        </div>
                    </FilterPopoverPanel>
                </Popover>

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
                    label="Add User"
                    title="Add User"
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
                                    label: 'ID',
                                    field: 'id',
                                    sortable: true,
                                },
                                {
                                    label: 'Name',
                                    field: 'name',
                                    sortable: true,
                                },
                                {
                                    label: 'Email',
                                    field: 'email',
                                    sortable: true,
                                },
                                {
                                    label: 'Role',
                                    field: 'role',
                                    sortable: true,
                                },
                                {
                                    label: 'Created At',
                                    field: 'created_at',
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
                        v-for="user in displayUsers"
                        v-if="displayUsers.length > 0"
                        :key="user.id || Math.random()"
                        class="group/tr table-tr relative border-y border-transparent border-b-gray-200 dark:border-b-dark-500"
                        :class="{ 'opacity-50': isTableLoading }"
                    >
                        <!-- ID Column -->
                        <TableCell>
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100"
                            >
                                {{ user?.id || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Name Column -->
                        <TableCell>
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100"
                            >
                                {{ user?.name || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Email Column -->
                        <TableCell>
                            <p class="text-gray-800 dark:text-dark-100">
                                {{ user?.email || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Role Column -->
                        <TableCell>
                            <span
                                class="inline-flex items-center px-2.5 py-0.5 rounded-full font-medium text-xs capitalize"
                                :class="getRoleBadgeClass(user?.role)"
                            >
                                {{ user?.role || "N/A" }}
                            </span>
                        </TableCell>

                        <!-- Created At Column -->
                        <TableCell>
                            <p class="text-gray-800 dark:text-dark-100">
                                {{ user?.created_at || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Actions Column -->
                        <TableCell>
                            <EllipsisDropdown
                                :disabled="isAnyOperationInProgress"
                            >
                                <EditItem
                                    label="Edit User"
                                    title="Edit User"
                                    :disabled="isAnyOperationInProgress"
                                    @click="handleEditModalOpen(user.id)"
                                />
                                <DeleteItem
                                    label="Delete User"
                                    title="Delete User"
                                    :disabled="isAnyOperationInProgress"
                                    @click="handleDeleteUser(user)"
                                />
                            </EllipsisDropdown>
                        </TableCell>
                    </tr>

                    <RowNotFound
                        v-if="displayUsers.length === 0 && !isTableLoading"
                        label="No users found"
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

        <!-- User Create Modal - Teleported to body -->
        <Teleport to="body">
            <UserCreateModal
                :is-open="isCreateModalOpen"
                @close="handleCreateModalClose"
                @operation="handleUserOperation"
            />
        </Teleport>

        <!-- User Edit Modal - Teleported to body -->
        <Teleport to="body">
            <UserEditModal
                v-if="editUserId"
                :is-open="isEditModalOpen"
                :user-id="editUserId"
                @close="handleEditModalClose"
                @operation="handleUserOperation"
            />
        </Teleport>
    </Tablewrapper>
</template>
