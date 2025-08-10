<script setup>
import { route } from "ziggy-js";

const props = defineProps({
    averageTime: Number,
    successRate: Number,
    defectRate: Number,
    AIConfidence: Number,
});
</script>

<template>
    <div class="flex flex-col gap-2">
        <div
            class="flex justify-between bg-this-darker/[0.07] dark:bg-this-lighter/10 py-2 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:primary"
        >
            <div class="flex items-center gap-2">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    class="size-5"
                >
                    <path
                        fill-rule="evenodd"
                        d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25ZM12.75 6a.75.75 0 0 0-1.5 0v6c0 .414.336.75.75.75h4.5a.75.75 0 0 0 0-1.5h-3.75V6Z"
                        clip-rule="evenodd"
                    />
                </svg>
                <span>Avg Processing Time</span>
            </div>
            {{ props.averageTime }}s
        </div>
        <div
            class="flex justify-between bg-this-darker/[0.07] dark:bg-this-lighter/10 py-2 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:success"
        >
            <div class="flex items-center gap-2">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    class="size-5"
                >
                    <path
                        fill-rule="evenodd"
                        d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12Zm13.36-1.814a.75.75 0 1 0-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 0 0-1.06 1.06l2.25 2.25a.75.75 0 0 0 1.14-.094l3.75-5.25Z"
                        clip-rule="evenodd"
                    />
                </svg>
                <span>Success Rate</span>
            </div>
            {{ props.successRate }}%
        </div>
        <div
            class="flex justify-between bg-this-darker/[0.07] dark:bg-this-lighter/10 py-2 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:error"
        >
            <div class="flex items-center gap-2">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    class="size-5"
                >
                    <path
                        fill-rule="evenodd"
                        d="M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003ZM12 8.25a.75.75 0 0 1 .75.75v3.75a.75.75 0 0 1-1.5 0V9a.75.75 0 0 1 .75-.75Zm0 8.25a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"
                        clip-rule="evenodd"
                    />
                </svg>
                <span>Defect Rate</span>
            </div>
            {{ props.defectRate }}%
        </div>
        <div
            class="flex justify-between bg-this-darker/[0.07] dark:bg-this-lighter/10 py-2 border border-this-darker/20 dark:border-this-lighter/20 text-this-darker dark:text-this-lighter badge-base badge this:secondary"
        >
            <div class="flex items-center gap-2">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    class="size-5"
                >
                    <path
                        fill-rule="evenodd"
                        d="M12.516 2.17a.75.75 0 0 0-1.032 0 11.209 11.209 0 0 1-7.877 3.08.75.75 0 0 0-.722.515A12.74 12.74 0 0 0 2.25 9.75c0 5.942 4.064 10.933 9.563 12.348a.749.749 0 0 0 .374 0c5.499-1.415 9.563-6.406 9.563-12.348 0-1.39-.223-2.73-.635-3.985a.75.75 0 0 0-.722-.516l-.143.001c-2.996 0-5.717-1.17-7.734-3.08Zm3.094 8.016a.75.75 0 1 0-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 0 0-1.06 1.06l2.25 2.25a.75.75 0 0 0 1.14-.094l3.75-5.25Z"
                        clip-rule="evenodd"
                    />
                </svg>
                <span>AI Confidence</span>
            </div>
            {{ props.AIConfidence }}%
        </div>
    </div>
    <div class="flex flex-col gap-4 mt-3">
        <h2
            class="font-medium text-gray-800 dark:text-dark-100 text-sm-plus tracking-wide"
        >
            Quick Actions
        </h2>
        <Link :href="route('scans.create')">
            <button
                class="gap-2 hover:bg-this-darker/[.05] focus:bg-this-darker/[.05] active:bg-this-darker/10 dark:hover:bg-this-lighter/[.05] dark:focus:bg-this-lighter/[.05] dark:active:bg-this-lighter/10 py-2 border border-this-darker dark:border-this-lighter w-full text-this-darker dark:text-this-lighter btn-base btn this:primary"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    class="size-5"
                >
                    <path
                        fill-rule="evenodd"
                        d="M1.5 6a2.25 2.25 0 0 1 2.25-2.25h16.5A2.25 2.25 0 0 1 22.5 6v12a2.25 2.25 0 0 1-2.25 2.25H3.75A2.25 2.25 0 0 1 1.5 18V6ZM3 16.06V18c0 .414.336.75.75.75h16.5A.75.75 0 0 0 21 18v-1.94l-2.69-2.689a1.5 1.5 0 0 0-2.12 0l-.88.879.97.97a.75.75 0 1 1-1.06 1.06l-5.16-5.159a1.5 1.5 0 0 0-2.12 0L3 16.061Zm10.125-7.81a1.125 1.125 0 1 1 2.25 0 1.125 1.125 0 0 1-2.25 0Z"
                        clip-rule="evenodd"
                    />
                </svg>
                <span>Image Analysis</span>
            </button>
        </Link>
        <Link :href="route('sessions.create')">
            <button
                class="gap-2 hover:bg-this-darker/[.05] focus:bg-this-darker/[.05] active:bg-this-darker/10 dark:hover:bg-this-lighter/[.05] dark:focus:bg-this-lighter/[.05] dark:active:bg-this-lighter/10 py-2 border border-this-darker dark:border-this-lighter w-full text-this-darker dark:text-this-lighter btn-base btn this:success"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    class="size-5"
                >
                    <path
                        fill-rule="evenodd"
                        d="M5.636 4.575a.75.75 0 0 1 0 1.061 9 9 0 0 0 0 12.728.75.75 0 1 1-1.06 1.06c-4.101-4.1-4.101-10.748 0-14.849a.75.75 0 0 1 1.06 0Zm12.728 0a.75.75 0 0 1 1.06 0c4.101 4.1 4.101 10.75 0 14.85a.75.75 0 1 1-1.06-1.061 9 9 0 0 0 0-12.728.75.75 0 0 1 0-1.06ZM7.757 6.697a.75.75 0 0 1 0 1.06 6 6 0 0 0 0 8.486.75.75 0 0 1-1.06 1.06 7.5 7.5 0 0 1 0-10.606.75.75 0 0 1 1.06 0Zm8.486 0a.75.75 0 0 1 1.06 0 7.5 7.5 0 0 1 0 10.606.75.75 0 0 1-1.06-1.06 6 6 0 0 0 0-8.486.75.75 0 0 1 0-1.06ZM9.879 8.818a.75.75 0 0 1 0 1.06 3 3 0 0 0 0 4.243.75.75 0 1 1-1.061 1.061 4.5 4.5 0 0 1 0-6.364.75.75 0 0 1 1.06 0Zm4.242 0a.75.75 0 0 1 1.061 0 4.5 4.5 0 0 1 0 6.364.75.75 0 0 1-1.06-1.06 3 3 0 0 0 0-4.243.75.75 0 0 1 0-1.061ZM10.875 12a1.125 1.125 0 1 1 2.25 0 1.125 1.125 0 0 1-2.25 0Z"
                        clip-rule="evenodd"
                    />
                </svg>

                <span>Real-Time Analysis</span>
            </button>
        </Link>
        <Link :href="route('scans.myscans')">
            <button
                class="gap-2 hover:bg-this-darker/[.05] focus:bg-this-darker/[.05] active:bg-this-darker/10 dark:hover:bg-this-lighter/[.05] dark:focus:bg-this-lighter/[.05] dark:active:bg-this-lighter/10 py-2 border border-this-darker dark:border-this-lighter w-full text-this-darker dark:text-this-lighter btn-base btn this:error"
            >
                <font-awesome-icon icon="fa-solid fa-clock-rotate-left" />

                <span>History Analysis</span>
            </button>
        </Link>
        <Link :href="route('settings')">
            <button
                class="gap-2 hover:bg-this-darker/[.05] focus:bg-this-darker/[.05] active:bg-this-darker/10 dark:hover:bg-this-lighter/[.05] dark:focus:bg-this-lighter/[.05] dark:active:bg-this-lighter/10 py-2 border border-this-darker dark:border-this-lighter w-full text-this-darker dark:text-this-lighter btn-base btn this:secondary"
            >
                <font-awesome-icon icon="fa-solid fa-cogs" />

                <span>Settings</span>
            </button>
        </Link>
    </div>
</template>
