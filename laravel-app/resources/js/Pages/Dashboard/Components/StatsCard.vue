<script setup>
import { computed } from "vue";

const props = defineProps({
    title: String,
    value: [String, Number],
    description: String,
    badgeColor: String,
    icon: String,
    changeRate: String,
});

const isPositiveChange = computed(() => {
    // Assuming changeRate is a string like "+10%" or "-5%"
    return props.changeRate.startsWith("+");
});
</script>

<template>
    <div
        class="relative break-words print:border card rounded-lg border border-gray-200 dark:border-dark-600 print:border-0 p-6"
    >
        <div class="flex min-w-0 items-center justify-between">
            <div class="">
                <p class="text-base text-dark-800 dark:text-dark-100">
                    {{ props.value }}
                </p>
                <p class="text-base truncate font-semibold">
                    {{ props.title }}
                </p>
            </div>
            <div
                class="avatar relative inline-flex shrink-0"
                style="height: 2.5rem; width: 2.5rem"
            >
                <div
                    class="avatar-initial avatar-display flex h-full w-full select-none items-center justify-center font-medium uppercase bg-this text-white mask is-star rounded-none"
                    :class="[
                        props.badgeColor === 'primary' ? 'this:primary' : '',
                        props.badgeColor === 'secondary'
                            ? 'this:secondary'
                            : '',
                        props.badgeColor === 'info' ? 'this:info' : '',
                        props.badgeColor === 'success' ? 'this:success' : '',
                        props.badgeColor === 'warning' ? 'this:warning' : '',
                        props.badgeColor === 'error' ? 'this:error' : '',
                    ]"
                >
                    <slot name="icon" />
                </div>
            </div>
        </div>
        <div class="mt-2">
            <div
                class="badge-base badge text-this-darker bg-this-darker/[0.07] dark:text-this-lighter dark:bg-this-lighter/10 gap-1 rounded-full"
                :class="{
                    'this:success': isPositiveChange,
                    'this:error': !isPositiveChange,
                }"
            >
                <span>{{ props.changeRate }}</span>

                <!-- Positive -->
                <svg
                    v-if="isPositiveChange"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke-width="1.5"
                    stroke="currentColor"
                    aria-hidden="true"
                    data-slot="icon"
                    class="size-4"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="M2.25 18 9 11.25l4.306 4.306a11.95 11.95 0 0 1 5.814-5.518l2.74-1.22m0 0-5.94-2.281m5.94 2.28-2.28 5.941"
                    ></path>
                </svg>

                <!-- Negative -->
                <svg
                    v-else
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke-width="1.5"
                    stroke="currentColor"
                    aria-hidden="true"
                    data-slot="icon"
                    class="size-4"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="M2.25 6 9 12.75l4.286-4.286a11.948 11.948 0 0 1 4.306 6.43l.776 2.898m0 0 3.182-5.511m-3.182 5.51-5.511-3.181"
                    ></path>
                </svg>
            </div>
        </div>
    </div>
</template>
