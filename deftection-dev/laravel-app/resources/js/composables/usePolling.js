// composables/usePolling.js - Enhanced Reusable Version with VueUse (No Metrics)
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { router } from "@inertiajs/vue3";
import { useDocumentVisibility, useOnline, useIntervalFn } from "@vueuse/core";
import axios from "axios";

export function usePolling(options = {}) {
    const {
        // API endpoints
        checkUrl = "/api/check-updates",
        dataUrl = "/api/poll-data",
        forceRefreshUrl = "/api/force-refresh",

        // Polling configuration
        interval = 5000,
        enabled = true,

        // Callbacks
        onDataUpdate = null,
        onError = null,

        // Debug and advanced options
        debug = false,
        retryAttempts = 3,
        retryDelay = 2500,

        // Visibility and network options
        pauseOnTabHidden = true,
        pauseOnOffline = true,

        // Custom conditions
        customPauseCondition = null,
    } = options;

    // ===============================
    // CORE STATE
    // ===============================
    const isPolling = ref(false);
    const lastChecksum = ref("");
    const isLoading = ref(false);
    const error = ref(null);
    const isManuallyPaused = ref(false);
    const retryCount = ref(0);

    // ===============================
    // VUEUSE INTEGRATIONS
    // ===============================
    const visibility = useDocumentVisibility();
    const isOnline = useOnline();

    // ===============================
    // COMPUTED CONDITIONS
    // ===============================
    const canPoll = computed(() => {
        const baseConditions = enabled && isOnline.value;
        const visibilityCondition = pauseOnTabHidden
            ? visibility.value === "visible"
            : true;
        const offlineCondition = pauseOnOffline ? isOnline.value : true;
        const customCondition = customPauseCondition
            ? customPauseCondition()
            : true;
        const manualCondition = !isManuallyPaused.value;

        return (
            baseConditions &&
            visibilityCondition &&
            offlineCondition &&
            customCondition &&
            manualCondition
        );
    });

    // ===============================
    // INTERVAL MANAGEMENT WITH VUEUSE
    // ===============================
    const { pause: pauseInterval, resume: resumeInterval } = useIntervalFn(
        async () => {
            if (canPoll.value) {
                await checkForUpdates();
            }
        },
        interval,
        { immediate: false }
    );

    // ===============================
    // DEBUG UTILITIES
    // ===============================
    const debugLog = (message, data = {}) => {
        if (debug) {
            console.log(`[usePolling] ${message}`, {
                timestamp: new Date().toISOString(),
                ...data,
            });
        }
    };

    // ===============================
    // POLLING CONTROL METHODS
    // ===============================
    const startPolling = () => {
        if (!canPoll.value || isPolling.value) {
            debugLog("Cannot start polling", {
                canPoll: canPoll.value,
                isPolling: isPolling.value,
                reasons: {
                    enabled,
                    isOnline: isOnline.value,
                    visibility: visibility.value,
                    manuallyPaused: isManuallyPaused.value,
                    customCondition: customPauseCondition
                        ? customPauseCondition()
                        : true,
                },
            });
            return false;
        }

        isPolling.value = true;
        resumeInterval();

        debugLog(`Polling started with ${interval}ms interval`);
        return true;
    };

    const stopPolling = () => {
        isPolling.value = false;
        pauseInterval();
        debugLog("Polling stopped");
    };

    const togglePolling = () => {
        isManuallyPaused.value = !isManuallyPaused.value;

        if (isManuallyPaused.value) {
            stopPolling();
            debugLog("Polling manually paused");
        } else if (canPoll.value) {
            startPolling();
            debugLog("Polling manually resumed");
        }
    };

    // ===============================
    // CORE POLLING LOGIC
    // ===============================
    const checkForUpdates = async () => {
        if (isLoading.value) {
            debugLog("Check skipped - already loading");
            return;
        }

        const startTime = performance.now();

        try {
            debugLog("Checking for updates...", {
                checksum: lastChecksum.value,
                attempt: retryCount.value + 1,
            });

            const response = await axios.get(checkUrl, {
                params: { checksum: lastChecksum.value },
                timeout: 10000,
            });

            const { has_changes, checksum } = response.data;
            const responseTime = performance.now() - startTime;

            if (has_changes) {
                debugLog("DATA CHANGES DETECTED!", {
                    oldChecksum: lastChecksum.value,
                    newChecksum: checksum,
                    responseTime: `${Math.round(responseTime)}ms`,
                });

                await fetchData();
            } else {
                debugLog("No changes detected", {
                    checksum,
                    responseTime: `${Math.round(responseTime)}ms`,
                });
            }

            lastChecksum.value = checksum;
            error.value = null;
            retryCount.value = 0;
        } catch (err) {
            handleError(err, "checkForUpdates");
        }
    };

    const fetchData = async () => {
        if (isLoading.value) {
            debugLog("Fetch skipped - already loading");
            return;
        }

        try {
            isLoading.value = true;

            debugLog("FETCHING updated data...");

            // Get current URL parameters for context
            const currentParams = new URLSearchParams(window.location.search);

            const response = await axios.get(dataUrl, {
                params: Object.fromEntries(currentParams),
                timeout: 15000,
            });

            const { checksum, ...data } = response.data;
            lastChecksum.value = checksum;

            // Call update callback or default behavior
            if (onDataUpdate) {
                onDataUpdate(data);
                debugLog("Data updated via callback");
            } else {
                // Default Inertia behavior
                router.reload({ only: ["scans", "filters", "meta"] });
                debugLog("Data updated via Inertia reload");
            }

            error.value = null;
        } catch (err) {
            handleError(err, "fetchData");
        } finally {
            isLoading.value = false;
        }
    };

    const forceRefresh = async () => {
        try {
            debugLog("FORCE REFRESH initiated");

            // Reset checksum to force change detection
            const oldChecksum = lastChecksum.value;
            lastChecksum.value = "";

            // Use force refresh endpoint if available, otherwise use fetchData
            if (forceRefreshUrl !== dataUrl) {
                const currentParams = new URLSearchParams(
                    window.location.search
                );
                const response = await axios.get(forceRefreshUrl, {
                    params: Object.fromEntries(currentParams),
                    timeout: 15000,
                });

                const { checksum, ...data } = response.data;
                lastChecksum.value = checksum;

                if (onDataUpdate) {
                    onDataUpdate(data);
                } else {
                    router.reload({ only: ["scans", "filters", "meta"] });
                }
            } else {
                await fetchData();
            }

            debugLog("Force refresh completed");
        } catch (err) {
            handleError(err, "forceRefresh");
        }
    };

    // ===============================
    // ERROR HANDLING & RETRY LOGIC
    // ===============================
    const handleError = (err, context) => {
        retryCount.value++;

        const errorMessage = err.response?.data?.error || err.message;
        error.value = errorMessage;

        debugLog(`${context} failed`, {
            error: errorMessage,
            retryCount: retryCount.value,
            maxRetries: retryAttempts,
        });

        if (onError) {
            onError(err);
        }

        // Implement retry logic
        if (retryCount.value < retryAttempts) {
            const delay = retryDelay * Math.pow(2, retryCount.value - 1); // Exponential backoff
            debugLog(`Retrying in ${delay}ms...`);

            setTimeout(() => {
                if (context === "checkForUpdates") {
                    checkForUpdates();
                } else if (context === "fetchData") {
                    fetchData();
                }
            }, delay);
        } else {
            debugLog("Max retry attempts reached");
            retryCount.value = 0;
        }
    };

    // ===============================
    // UTILITY METHODS
    // ===============================
    const initializeChecksum = (checksum) => {
        lastChecksum.value = checksum || "";
        debugLog("Checksum initialized", { checksum: lastChecksum.value });
    };

    const getStatus = () => ({
        isPolling: isPolling.value,
        isLoading: isLoading.value,
        canPoll: canPoll.value,
        error: error.value,
        isManuallyPaused: isManuallyPaused.value,
        conditions: {
            enabled,
            isOnline: isOnline.value,
            visibility: visibility.value,
            customCondition: customPauseCondition
                ? customPauseCondition()
                : true,
        },
    });

    // ===============================
    // WATCHERS
    // ===============================
    watch(canPoll, (newValue, oldValue) => {
        debugLog("canPoll changed", { from: oldValue, to: newValue });

        if (newValue && !isPolling.value) {
            startPolling();
        } else if (!newValue && isPolling.value) {
            stopPolling();
        }
    });

    // Watch online status
    watch(isOnline, (online) => {
        debugLog("Online status changed", { online });
        if (!online && pauseOnOffline) {
            error.value = "Connection lost. Polling paused.";
        } else if (
            online &&
            error.value === "Connection lost. Polling paused."
        ) {
            error.value = null;
        }
    });

    // Watch visibility
    watch(visibility, (visible) => {
        debugLog("Visibility changed", { visible });
    });

    // ===============================
    // LIFECYCLE
    // ===============================
    onMounted(() => {
        debugLog("usePolling mounted", {
            enabled,
            interval,
            canPoll: canPoll.value,
            conditions: {
                isOnline: isOnline.value,
                visibility: visibility.value,
            },
        });

        if (canPoll.value) {
            startPolling();
        }
    });

    onUnmounted(() => {
        debugLog("usePolling unmounting");
        stopPolling();
    });

    // ===============================
    // PUBLIC API
    // ===============================
    return {
        // State
        isPolling: computed(() => isPolling.value),
        isLoading: computed(() => isLoading.value),
        error: computed(() => error.value),
        isManuallyPaused: computed(() => isManuallyPaused.value),
        canPoll,

        // External conditions
        isOnline,
        visibility,

        // Control methods
        startPolling,
        stopPolling,
        togglePolling,
        forceRefresh,

        // Data methods
        checkForUpdates,
        fetchData,

        // Utility methods
        initializeChecksum,
        getStatus,
    };
}
