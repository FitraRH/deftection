<script setup>
import { ref, onMounted, watch } from "vue";

// --- Component State ---
// Holds the list of available video input devices
const cameraDevices = ref([]);
// Holds the v-model values for the form inputs
const selectedCameraId = ref(localStorage.getItem("selectedCameraId") || null);
const selectedInterval = ref(500);
const autoCaptureEnabled = ref(true);

// --- Emits ---
// Defines the events this component can send to its parent
const emit = defineEmits([
    "update:cameraId",
    "update:scanInterval",
    "update:autoCapture",
    "reset-stats",
]);

// --- Methods ---

/**
 * Fetches the list of available video cameras from the browser.
 */
const getCameraDevices = async () => {
    try {
        // We need to request permission first to get the device labels
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        cameraDevices.value = devices.filter(
            (device) => device.kind === "videoinput"
        );

        // If no camera is selected, or the saved one is gone, select the first available one.
        if (
            cameraDevices.value.length > 0 &&
            (!selectedCameraId.value ||
                !cameraDevices.value.find(
                    (d) => d.deviceId === selectedCameraId.value
                ))
        ) {
            selectedCameraId.value = cameraDevices.value[0].deviceId;
        }
    } catch (error) {
        console.error("Could not get camera devices:", error);
        // Handle cases where camera access is denied
    }
};

/**
 * Emits an event to the parent component to reset statistics.
 */
const handleResetClick = () => {
    emit("reset-stats");
};

// --- Lifecycle Hooks ---

// When the component is first mounted, fetch the camera list and emit initial values.
onMounted(() => {
    getCameraDevices();
    // Emit initial default values to the parent on load
    emit("update:cameraId", selectedCameraId.value);
    emit("update:scanInterval", selectedInterval.value);
    emit("update:autoCapture", autoCaptureEnabled.value);
});

// --- Watchers ---
// Watch for changes in the selected values and emit them to the parent.

watch(selectedCameraId, (newId) => {
    emit("update:cameraId", newId);
    // Save the user's preference for next time
    if (newId) {
        localStorage.setItem("selectedCameraId", newId);
    }
});

watch(selectedInterval, (newInterval) => {
    emit("update:scanInterval", newInterval);
});

watch(autoCaptureEnabled, (isEnabled) => {
    emit("update:autoCapture", isEnabled);
});
</script>

<template>
    <div
        class="flex flex-col justify-start border border-gray-200 dark:border-dark-700 rounded-lg p-6 gap-3"
    >
        <h2
            class="truncate text-base font-medium tracking-wide text-gray-800 dark:text-dark-100"
        >
            Session Controls
        </h2>

        <div class="space-y-4">
            <!-- Camera Source Selector -->
            <div>
                <label for="camera-source" class="input-label undefined">
                    Camera Source
                </label>
                <div class="input-wrapper relative mt-1.5">
                    <select
                        id="camera-source"
                        v-model="selectedCameraId"
                        class="form-select-base form-select ltr:pr-9 rtl:pl-9 peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                    >
                        <option
                            v-if="cameraDevices.length === 0"
                            disabled
                            :value="null"
                        >
                            Loading cameras...
                        </option>
                        <option
                            v-for="device in cameraDevices"
                            :key="device.deviceId"
                            :value="device.deviceId"
                        >
                            {{
                                device.label ||
                                `Camera ${cameraDevices.indexOf(device) + 1}`
                            }}
                        </option>
                    </select>
                    <div
                        class="suffix ltr:right-0 rtl:left-0 pointer-events-none absolute top-0 flex h-full w-9 items-center justify-center transition-colors text-gray-400 peer-focus:text-primary-600 dark:text-dark-300 dark:peer-focus:text-primary-500"
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                            aria-hidden="true"
                            data-slot="icon"
                            class="w-2/3"
                        >
                            <path
                                fill-rule="evenodd"
                                d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                                clip-rule="evenodd"
                            ></path>
                        </svg>
                    </div>
                </div>
            </div>

            <!-- Detection Interval Selector -->
            <div>
                <label for="detection-interval" class="input-label undefined">
                    Detection Interval
                </label>
                <div class="input-wrapper relative mt-1.5">
                    <select
                        id="detection-interval"
                        v-model="selectedInterval"
                        class="form-select-base form-select ltr:pr-9 rtl:pl-9 peer border-gray-300 hover:border-gray-400 focus:border-primary-600 dark:border-dark-450 dark:hover:border-dark-400 dark:focus:border-primary-500"
                    >
                        <option :value="1000">1 Second (1 FPS)</option>
                        <option :value="500">0.5 Seconds (2 FPS)</option>
                        <option :value="250">0.25 Seconds (4 FPS)</option>
                        <option :value="100">0.1 Seconds (10 FPS)</option>
                    </select>
                    <div
                        class="suffix ltr:right-0 rtl:left-0 pointer-events-none absolute top-0 flex h-full w-9 items-center justify-center transition-colors text-gray-400 peer-focus:text-primary-600 dark:text-dark-300 dark:peer-focus:text-primary-500"
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                            aria-hidden="true"
                            data-slot="icon"
                            class="w-2/3"
                        >
                            <path
                                fill-rule="evenodd"
                                d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
                                clip-rule="evenodd"
                            ></path>
                        </svg>
                    </div>
                </div>
            </div>

            <!-- Auto-capture Checkbox -->
            <label class="input-label inline-flex items-center gap-2">
                <input
                    id="auto-capture"
                    v-model="autoCaptureEnabled"
                    class="form-checkbox this:primary border-gray-400/70 bg-origin-border before:bg-center before:bg-no-repeat before:[background-size:100%_100%] before:[background-image:var(--tw-thumb)] checked:border-this checked:bg-this indeterminate:border-this indeterminate:bg-this hover:border-this focus:border-this dark:border-dark-400 dark:checked:border-this-light dark:checked:bg-this-light dark:indeterminate:border-this-light dark:indeterminate:bg-this-light dark:hover:border-this-light dark:focus:border-this-light"
                    type="checkbox"
                    checked=""
                />
                <span class="label">Auto-scan for defects</span>
            </label>

            <!-- Reset Statistics Button -->
            <div>
                <button
                    @click="handleResetClick"
                    type="button"
                    class="btn-base btn w-full gap-2 this:primary bg-this text-white hover:bg-this-darker focus:bg-this-darker active:bg-this-darker/90 disabled:bg-this-light dark:disabled:bg-this-darker"
                >
                    <font-awesome-icon icon="fa-solid fa-arrow-rotate-left" />
                    Reset Statistics
                </button>
            </div>
        </div>
    </div>
</template>

<style scoped>
.dark select {
    --bg-color: rgb(14, 15, 17);
}

select {
    --bg-color: rgb(255, 255, 255);
}
</style>
