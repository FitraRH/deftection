<script setup></script>

<template>
    <div
        class="px-6 py-8 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-dark-800 dark:shadow-none dark:border-none"
    >
        <div class="flex flex-col items-center justify-center">
            <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 154 154"
                fill="none"
                data-animated-tick="true"
                class="stroke-current mx-auto size-24 shrink-0 text-success"
            >
                <path
                    d="M77 141C112.346 141 141 112.346 141 77C141 41.6538 112.346 13 77 13C41.6538 13 13 41.6538 13 77C13 112.346 41.6538 141 77 141Z"
                    stroke-width="10"
                    class="checkmark-circle"
                    style="
                        stroke-dasharray: 480px, 480px;
                        stroke-dashoffset: 960px;
                    "
                ></path>
                <path
                    d="M46 80.2444L63.9556 98.1111L107.067 55"
                    stroke-width="10"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    class="checkmark-tick"
                    style="
                        stroke-dasharray: 100px, 100px;
                        stroke-dashoffset: 200px;
                    "
                ></path>
            </svg>

            <!-- Main Title -->
            <h3
                class="mt-3 text-base font-medium text-gray-800 dark:text-dark-100"
            >
                Product Quality Approved
            </h3>

            <!-- Subtitle -->
            <p class="mt-1 text-gray-400 dark:text-dark-300">
                No defects were detected. This product meets all quality
                standards and is ready for distribution.
            </p>

            <!-- "PASSED" Button/Badge -->
            <div class="mt-3">
                <span
                    class="inline-flex items-center gap-2 px-4 py-2 font-medium text-green-700 bg-green-100 rounded-full dark:bg-green-500/10 dark:text-green-400"
                >
                    <font-awesome-icon icon="fa-solid fa-thumbs-up" />
                    Quality Assurance: PASSED
                </span>
            </div>
        </div>
    </div>
</template>
