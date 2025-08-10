<script setup>
import { ref, reactive, computed } from "vue";

// --- Import all the child components ---
import CameraFeed from "./Components/CameraFeed.vue";
import SessionControls from "./Components/SessionControls.vue";
import LiveStatistics from "./Components/LiveStatistics.vue";
import ScreenshotGallery from "./Components/ScreenshotGallery.vue";
import RecentDetections from "./Components/RecentDetections.vue";

// --- Component Refs ---
const cameraFeedRef = ref(null);

// --- State Management ---
const sessionConfig = reactive({
    selectedCameraId: null,
    scanInterval: 1500,
    autoCaptureEnabled: true,
});
const currentDetections = ref([]);
const stats = reactive({
    totalFrames: 0,
    goodProducts: 0,
    defectiveProducts: 0,
    screenshots: 0,
});
const allScreenshots = ref([]);

// --- Computed Properties ---
const detectionRate = computed(() => {
    if (stats.totalFrames === 0) return 0;
    return stats.defectiveProducts / stats.totalFrames;
});

// A computed property to get the most recent DEFECTS for the sidebar feed
const recentDetections = computed(() => {
    return allScreenshots.value.filter((s) => s.type === "Defect").slice(0, 5);
});

// --- Main Analysis Logic ---
const handleFrameForAnalysis = async (frameData) => {
    // 1. Simulate API call
    const fakeApiResponse = await simulateApiCall();

    // 2. Pass detection data down to CameraFeed to draw the boxes
    currentDetections.value = fakeApiResponse.detections;

    // 3. Wait for the next frame to ensure boxes are drawn
    await new Promise((resolve) => requestAnimationFrame(resolve));

    // 4. Call the child component to get a NEW screenshot that includes the overlay
    if (cameraFeedRef.value) {
        const blobWithOverlay = await cameraFeedRef.value.saveAnalyzedFrame();

        if (blobWithOverlay) {
            // 5. Process the result using the NEW blob
            stats.totalFrames++;
            if (fakeApiResponse.status === "good") {
                stats.goodProducts++;
                addScreenshot(blobWithOverlay, "Good", frameData.timestamp);
            } else if (fakeApiResponse.status === "defected") {
                stats.defectiveProducts++;
                addScreenshot(blobWithOverlay, "Defect", frameData.timestamp);
            }
        }
    }

    // 6. Clear bounding boxes after a delay
    setTimeout(() => {
        currentDetections.value = [];
    }, 2500);
};

const simulateApiCall = () => {
    return new Promise((resolve) => {
        setTimeout(() => {
            const isDefect = Math.random() > 0.6; // 40% chance of defect
            resolve({
                status: isDefect ? "defected" : "good",
                detections: isDefect
                    ? [
                          { label: "Product", bbox: [150, 200, 400, 250] },
                          { label: "Scratch", bbox: [250, 280, 150, 20] },
                      ]
                    : [{ label: "Product", bbox: [150, 200, 400, 250] }],
            });
        }, 500); // Simulate 500ms network latency
    });
};

// --- Helper Functions ---
/**
 * FIX: The first argument is the image blob, not the whole data object.
 */
const addScreenshot = (blob, type, timestamp) => {
    const screenshot = {
        id: `${type}-${timestamp.getTime()}`,
        url: URL.createObjectURL(blob),
        timestamp: timestamp,
        type: type,
    };
    allScreenshots.value.unshift(screenshot);
    stats.screenshots++;
};

const handleManualCapture = (screenshotData) => {
    // The blob from manual capture now correctly includes the overlay
    addScreenshot(screenshotData.blob, "Manual", screenshotData.timestamp);
};

const resetSession = () => {
    stats.totalFrames = 0;
    stats.goodProducts = 0;
    stats.defectiveProducts = 0;
    stats.screenshots = 0;
    allScreenshots.value.forEach((img) => URL.revokeObjectURL(img.url));
    allScreenshots.value = [];
    currentDetections.value = [];
    console.log("Session has been reset.");
};
</script>

<template>
    <div class="w-full">
        <div class="gap-6 grid grid-cols-1 lg:grid-cols-3">
            <!-- Main Content: Camera Feed -->
            <div class="lg:col-span-2">
                <CameraFeed
                    ref="cameraFeedRef"
                    :device-id="sessionConfig.selectedCameraId"
                    :auto-scan-enabled="sessionConfig.autoCaptureEnabled"
                    :scan-interval="sessionConfig.scanInterval"
                    :detections="currentDetections"
                    @frame-for-analysis="handleFrameForAnalysis"
                    @capture="handleManualCapture"
                    @started="handleDetectionStarted"
                    @stopped="handleDetectionStopped"
                />
            </div>

            <!-- Sidebar: Controls and Stats -->
            <div class="flex flex-col justify-between">
                <LiveStatistics
                    :total-frames="stats.totalFrames"
                    :good-products="stats.goodProducts"
                    :defective-products="stats.defectiveProducts"
                    :detection-rate="detectionRate"
                    :screenshots="stats.screenshots"
                />

                <div
                    class="bg-gray-200 dark:bg-dark-500 max-2xl:my-6 h-px"
                ></div>

                <SessionControls
                    @update:cameraId="
                        (id) => (sessionConfig.selectedCameraId = id)
                    "
                    @update:scanInterval="
                        (val) => (sessionConfig.scanInterval = val)
                    "
                    @update:autoCapture="
                        (val) => (sessionConfig.autoCaptureEnabled = val)
                    "
                    @reset-stats="resetSession"
                />
            </div>
        </div>

        <!-- Screenshot Gallery & Recent Detections -->
        <div class="gap-6 grid grid-cols-1 lg:grid-cols-3 mt-6">
            <RecentDetections :detections="recentDetections" />
            <div class="lg:col-span-2">
                <ScreenshotGallery
                    :screenshots="allScreenshots"
                    @clear-all="resetSession"
                />
            </div>
        </div>
    </div>
</template>
