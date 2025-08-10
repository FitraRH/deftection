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
import Table from "../../Shared/Table/Table.vue";
import Tablewrapper from "../../Shared/Table/Tablewrapper.vue";
import TableToolbar from "../../Shared/Table/TableToolbar.vue";
import TableContainer from "../../Shared/Table/TableContainer.vue";
import ToolbarSearch from "../../Shared/TableToolbar/ToolbarSearch.vue";
import TableHeaderCell from "../../Shared/Table/TableHeaderCell.vue";
import ForceRefreshButton from "../../Shared/Table/ForceRefreshButton.vue";
import LiveMonitorToggle from "../../Shared/Table/LiveMonitorToggle.vue";
import FilterPopoverButton from "../../Shared/Popover/FilterPopoverButton.vue";
import FilterPopoverPanel from "../../Shared/Popover/FilterPopoverPanel.vue";
import FilterResetButton from "../../Shared/TableToolbar/FilterResetButton.vue";
import FilterDateRange from "../../Shared/TableToolbar/FilterDateRange.vue";
import FilterPopoverCheckbox from "../../Shared/Popover/FilterPopoverCheckbox.vue";

import ToolbarExportButton from "../../Shared/TableToolbar/ToolbarExportButton.vue";
import TableCell from "../../Shared/Table/TableCell.vue";
import RowNotFound from "../../Shared/Table/RowNotFound.vue";
import TableFooter from "../../Shared/Table/TableFooter.vue";
import EllipsisDropdown from "../../Shared/Dropdown/EllipsisDropdown.vue";
import DetailViewList from "../../Shared/Dropdown/ViewItem.vue";
import ExportItem from "../../Shared/Dropdown/ExportItem.vue";
import DeleteItem from "../../Shared/Dropdown/DeleteItem.vue";

const props = defineProps({
    scans: { type: Array, required: true },
    filters: { type: Object, required: true },
    meta: { type: Object, default: () => ({}) },
    initialChecksum: { type: String, default: "" },
});

// Use props directly for navigation, reactive refs for polling updates
const currentFilters = computed(() => props.filters.current || {});
const filterOptions = computed(() => props.filters.options || {});
const searchTerm = ref(currentFilters.value.search || "");
const dateFrom = ref(currentFilters.value.dateFrom || "");
const dateTo = ref(currentFilters.value.dateTo || "");

// For polling updates only
const pollingScans = ref([]);
const pollingFilters = ref({});
const pollingMeta = ref({});

// Delete state
const isDeletingScan = ref(false);

// Loading states for UI control
const isTableLoading = ref(false);

// Determine which data to show: polling data if available, otherwise props
const displayScans = computed(() => {
    return pollingScans.value.length > 0 ? pollingScans.value : props.scans;
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
        isTableLoading.value || isPollingLoading.value || isDeletingScan.value
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
    checkUrl: route("scans.myscans.check"),
    dataUrl: route("scans.myscans.api"),
    forceRefreshUrl: route("scans.myscans.refresh"),

    // Polling configuration
    interval: 20000,
    enabled: true,
    pauseOnTabHidden: true,
    pauseOnOffline: true,
    debug: false,
    retryAttempts: 3,
    retryDelay: 10000,

    // Custom pause condition - can add custom logic if needed in the future
    customPauseCondition: () => {
        // Always allow polling for scan history page
        return true;
    },

    // Callbacks
    onDataUpdate: (data) => {
        // Only update polling state, don't interfere with navigation
        pollingScans.value = data.scans || [];
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

const formatDecimal = (value, defaultValue = 0, precision = 3) => {
    const cleaned = typeof value === "string" ? value.replace(/,/g, "") : value;
    const num = Number(cleaned);
    return isNaN(num) ? defaultValue : num.toFixed(precision);
};

// Filter options
const defectTypeOptions = ref([]);
const statusOptions = ref([]);

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

    defectTypeOptions.value = mapOptions(options.defectTypes, "defectTypes");
    statusOptions.value = mapOptions(options.status, "status");
};

// Add date filter handler
const handleDateFilterChange = () => {
    if (!isAnyOperationInProgress.value) {
        setTimeout(applyFilters, 10);
    }
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

// Watch for date filter changes from current filters
watch(
    () => [currentFilters.value.dateFrom, currentFilters.value.dateTo],
    ([newDateFrom, newDateTo]) => {
        dateFrom.value = newDateFrom || "";
        dateTo.value = newDateTo || "";
    }
);

// Initialize component
onMounted(() => {
    initializeFilters();
    initializeChecksum(props.initialChecksum);
});

// Computed properties
const selectedDefectTypesCount = computed(
    () => defectTypeOptions.value.filter((option) => option.selected).length
);
const selectedStatusCount = computed(
    () => statusOptions.value.filter((option) => option.selected).length
);
const hasActiveFilters = computed(
    () =>
        selectedDefectTypesCount.value > 0 ||
        selectedStatusCount.value > 0 ||
        searchTerm.value.length > 0 ||
        dateFrom.value.length > 0 ||
        dateTo.value.length > 0
);
const selectedDefectTypes = computed(() =>
    defectTypeOptions.value
        .filter((option) => option.selected)
        .map((option) => option.value)
);
const selectedStatus = computed(() =>
    statusOptions.value
        .filter((option) => option.selected)
        .map((option) => option.value)
);

// Delete functionality
const handleDeleteScan = async (scan) => {
    // Show confirmation dialog using clean swal utils
    const result = await deleteConfirmDialog(`"${scan.filename}"`);

    if (result.isConfirmed) {
        await performDelete(scan);
    }
};

// Perform the actual deletion
const performDelete = async (scan) => {
    isDeletingScan.value = true;

    const form = useForm({});

    form.delete(route("scans.destroy-myscan", scan.id), {
        preserveState: true,
        preserveScroll: true,
        onSuccess: () => {
            handleDeleteSuccess(scan.filename);
        },
        onError: (errors) => {
            handleDeleteError(errors);
        },
        onFinish: () => {
            isDeletingScan.value = false;
        },
    });
};

// Handle successful deletion
const handleDeleteSuccess = (scanFilename) => {
    console.log("Scan deleted successfully:", scanFilename);

    successToast("Analysis deleted successfully!");
};

// Handle deletion error
const handleDeleteError = (errors) => {
    console.error("Delete operation failed:", errors);

    let errorMessage =
        "An unexpected error occurred while deleting the analysis.";

    // Check for specific error messages
    if (errors.message) {
        errorMessage = errors.message;
    } else if (errors.error) {
        errorMessage = errors.error;
    }

    errorToast(errorMessage);
};

// Export functionality
const handleSingleExport = (scan) => {
    // Handle individual row export
    console.log("Exporting scan:", scan);
    // Add your export logic here
};

const handleBatchExport = () => {
    // Handle bulk export
    console.log("Exporting all scans with current filters");
    // Add your bulk export logic here
};

// Build filter parameters helper
const buildFilterParams = (customParams = {}) => {
    const params = {
        search: searchTerm.value || undefined,
        defect_types:
            selectedDefectTypes.value.length > 0
                ? selectedDefectTypes.value
                : undefined,
        status:
            selectedStatus.value.length > 0 ? selectedStatus.value : undefined,
        date_from: dateFrom.value || undefined,
        date_to: dateTo.value || undefined,
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
    pollingScans.value = [];
    pollingFilters.value = {};
    pollingMeta.value = {};

    router.get(route("scans.myscans"), buildFilterParams(customParams), {
        preserveState: true,
        preserveScroll: true,
        replace: true,
        only: ["scans", "filters", "meta", "initialChecksum"],
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

const clearDefectTypeFilters = () => clearFilter(defectTypeOptions);
const clearStatusFilters = () => clearFilter(statusOptions);

const resetAllFilters = () => {
    clearFilter(defectTypeOptions);
    clearFilter(statusOptions);
    searchTerm.value = "";
    dateFrom.value = "";
    dateTo.value = "";
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
                @click="togglePolling"
            />
            <!-- Force Refresh Button  -->
            <ForceRefreshButton
                :isPollingLoading="isPollingLoading"
                :isOnline="isOnline"
                :disabled="isTableLoading"
                @click="forceRefresh"
            />
        </div>

        <TableToolbar
            title="Analysis History"
            description="History of your analysis"
        >
            <template #left>
                <ToolbarSearch
                    v-model="searchTerm"
                    placeholder="Search filename, confidence level, or defects..."
                    :disabled="isAnyOperationInProgress"
                    @input="handleSearchChange"
                />

                <!-- Date Range Filter -->
                <FilterDateRange
                    v-model:dateFrom="dateFrom"
                    v-model:dateTo="dateTo"
                    :disabled="isAnyOperationInProgress"
                    @change="handleDateFilterChange"
                />

                <!-- Defect Type Filter -->
                <Popover class="relative">
                    <FilterPopoverButton
                        label="Defect Type"
                        :selected-options="selectedDefectTypesCount"
                        :disabled="isAnyOperationInProgress"
                    />
                    <FilterPopoverPanel
                        :show-clear-button="selectedDefectTypesCount > 0"
                        :selected-options="selectedDefectTypesCount"
                        @click-clear="clearDefectTypeFilters"
                    >
                        <FilterPopoverCheckbox
                            v-for="defectType in defectTypeOptions"
                            :key="defectType.value"
                            v-model:checked="defectType.selected"
                            :id="`defect-type-${defectType.value}`"
                            :value="defectType.value"
                            :label="defectType.label"
                            :count="defectType.count"
                            :disabled="isAnyOperationInProgress"
                            @change="handleFilterChange"
                        />
                        <div
                            v-if="defectTypeOptions.length === 0"
                            class="px-3 py-2 text-gray-500 text-xs"
                        >
                            No defect types found
                        </div>
                    </FilterPopoverPanel>
                </Popover>

                <!-- Status Filter -->
                <Popover class="relative">
                    <FilterPopoverButton
                        label="Status"
                        :selected-options="selectedStatusCount"
                        :disabled="isAnyOperationInProgress"
                    />
                    <FilterPopoverPanel
                        :show-clear-button="selectedStatusCount > 0"
                        :selected-options="selectedStatusCount"
                        @click-clear="clearStatusFilters"
                    >
                        <FilterPopoverCheckbox
                            v-for="status in statusOptions"
                            :key="status.value"
                            v-model:checked="status.selected"
                            :id="`status-${status.value}`"
                            :value="status.value"
                            :label="status.label"
                            :count="status.count"
                            :icon="status.value === 'good' ? 'check' : 'cross'"
                            :disabled="isAnyOperationInProgress"
                            @change="handleFilterChange"
                        />
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
                <ToolbarExportButton
                    label="Export PDF"
                    :disabled="
                        isAnyOperationInProgress || displayScans.length === 0
                    "
                    @export="handleBulkExport"
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

            <Table ref="tableRef">
                <template #head>
                    <tr class="group/tr table-tr">
                        <TableHeaderCell
                            v-for="header in [
                                {
                                    label: 'image',
                                    field: 'filename',
                                    sortable: true,
                                },
                                {
                                    label: 'analysis date',
                                    field: 'created_at',
                                    sortable: true,
                                },
                                { label: 'defects', sortable: false },
                                {
                                    label: 'status',
                                    field: 'is_defect',
                                    sortable: true,
                                },
                                {
                                    label: 'confidence',
                                    field: 'anomaly_confidence_level',
                                    sortable: true,
                                },
                                {
                                    label: 'score',
                                    field: 'anomaly_score',
                                    sortable: true,
                                },
                                { label: 'actions', sortable: false },
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
                        v-for="scan in displayScans"
                        v-if="displayScans.length > 0"
                        :key="scan.id || Math.random()"
                        class="group/tr table-tr relative border-y border-transparent border-b-gray-200 dark:border-b-dark-500"
                        :class="{ 'opacity-50': isTableLoading }"
                    >
                        <!-- Image Column -->
                        <TableCell>
                            <div class="flex items-center">
                                <img
                                    class="flex-shrink-0 rounded-md w-20 h-12 object-cover"
                                    :src="
                                        scan.annotated_path ||
                                        scan.original_path
                                    "
                                    alt="Analysis Image"
                                />
                                <p
                                    class="ml-4 font-medium text-gray-800 dark:text-dark-100"
                                >
                                    {{ scan?.filename || "N/A" }}
                                </p>
                            </div>
                        </TableCell>

                        <!-- Analysis Date Column -->
                        <TableCell>
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100"
                            >
                                {{ scan?.analysis_date || "N/A" }}
                            </p>
                            <p
                                class="mt-1 text-gray-400 dark:text-dark-300 text-xs"
                            >
                                {{ scan?.analysis_time || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Defects Column -->
                        <TableCell>
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100"
                            >
                                {{ scan?.defect_scanned || "0" }}
                            </p>
                            <p
                                class="mt-1 text-gray-400 dark:text-dark-300 text-xs"
                            >
                                {{ scan?.total_defect || "0" }} types
                            </p>
                        </TableCell>

                        <!-- Status Column -->
                        <TableCell>
                            <span
                                :class="[
                                    'inline-flex px-2 rounded-full font-semibold text-xs uppercase leading-5',
                                    scan.status === 'good'
                                        ? 'bg-green-100 text-green-800 dark:bg-green-500/10 dark:text-green-400'
                                        : scan.status === 'defect'
                                        ? 'bg-red-100 text-red-800 dark:bg-red-500/10 dark:text-red-400'
                                        : 'bg-gray-100 text-gray-800 dark:bg-gray-500/10 dark:text-gray-400',
                                ]"
                            >
                                {{ scan?.status || "unknown" }}
                            </span>
                        </TableCell>

                        <!-- Confidence Column -->
                        <TableCell>
                            <p
                                class="font-medium text-gray-800 dark:text-dark-100 capitalize"
                            >
                                {{ scan?.anomaly_confidence_level || "N/A" }}
                            </p>
                        </TableCell>

                        <!-- Score Column -->
                        <TableCell>
                            <div
                                class="font-medium text-gray-800 dark:text-dark-100"
                            >
                                {{
                                    formatDecimal(scan?.anomaly_score, "N/A", 3)
                                }}
                            </div>
                            <div
                                class="mt-1 text-gray-400 dark:text-dark-300 text-xs"
                            >
                                {{
                                    formatDecimal(
                                        scan?.total_processing_time,
                                        "0",
                                        3
                                    ) + " s"
                                }}
                            </div>
                        </TableCell>

                        <!-- Actions Column -->
                        <TableCell>
                            <EllipsisDropdown
                                :disabled="isAnyOperationInProgress"
                                @dropdown-opened="handleDropdownOpen"
                            >
                                <DetailViewList
                                    :href="route('scans.show', scan.id)"
                                    label="View"
                                    title="View Analysis Details"
                                    :disabled="isAnyOperationInProgress"
                                />
                                <ExportItem
                                    label="Export"
                                    title="Export Analysis"
                                    :href="
                                        route(
                                            'reports.single.generate',
                                            scan.id
                                        )
                                    "
                                    :disabled="isAnyOperationInProgress"
                                />
                                <DeleteItem
                                    label="Delete"
                                    title="Delete Analysis"
                                    :disabled="isAnyOperationInProgress"
                                    @click="handleDeleteScan(scan)"
                                />
                            </EllipsisDropdown>
                        </TableCell>
                    </tr>

                    <RowNotFound
                        v-if="displayScans.length === 0 && !isTableLoading"
                        label="No scans found"
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
    </Tablewrapper>
</template>
