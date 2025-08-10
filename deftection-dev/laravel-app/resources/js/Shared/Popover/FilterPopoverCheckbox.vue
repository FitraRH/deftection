<script setup>
import { defineModel, defineProps } from "vue";

const checked = defineModel("checked", {
    type: Boolean,
    default: false,
});

defineProps({
    id: {
        type: String,
        required: true,
    },
    value: {
        type: [String, Number],
        required: true,
    },
    label: {
        type: String,
        required: true,
    },
    icon: {
        type: String,
        default: "",
    },
    count: {
        type: Number,
        default: 0,
    },
});
</script>

<template>
    <div class="flex justify-between items-center">
        <div class="flex">
            <input
                :id="id"
                v-model="checked"
                :value="value"
                type="checkbox"
                class="checked:bg-this indeterminate:bg-this dark:checked:bg-this-light dark:indeterminate:bg-this-light bg-origin-border before:bg-no-repeat before:bg-center border-gray-400/70 dark:border-dark-400 checked:border-this indeterminate:border-this hover:border-this focus:border-this dark:checked:border-this-light dark:indeterminate:border-this-light dark:hover:border-this-light dark:focus:border-this-light form-checkbox this:primary before:[background-size:100%_100%] before:[background-image:var(--tw-thumb)]"
            />
            <label :for="id" class="ml-3">
                <div class="flex gap-2">
                    <slot name="icon">
                        <svg
                            v-if="icon === 'check'"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            aria-hidden="true"
                            data-slot="icon"
                            class="stroke-1 size-4.5"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="M9 12.75 11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 0 1-1.043 3.296 3.745 3.745 0 0 1-3.296 1.043A3.745 3.745 0 0 1 12 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 0 1-3.296-1.043 3.745 3.745 0 0 1-1.043-3.296A3.745 3.745 0 0 1 3 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 0 1 1.043-3.296 3.746 3.746 0 0 1 3.296-1.043A3.746 3.746 0 0 1 12 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 0 1 3.296 1.043 3.746 3.746 0 0 1 1.043 3.296A3.745 3.745 0 0 1 21 12Z"
                            ></path>
                        </svg>
                        <svg
                            v-else-if="icon === 'cross'"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            aria-hidden="true"
                            data-slot="icon"
                            class="stroke-1 size-4.5"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="m9.75 9.75 4.5 4.5m0-4.5-4.5 4.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
                            ></path>
                        </svg>
                    </slot>
                    <span class="block text-xs-plus truncate">
                        {{ label }}
                    </span>
                </div>
            </label>
        </div>
        <span class="font-mono text-xs">
            {{ count }}
        </span>
    </div>
</template>
