<script setup>
import { Link } from "@inertiajs/vue3";
import { defineProps } from "vue";

defineProps({
    active: Boolean,
    href: String,
    as: String,
    method: String,
});
</script>

<template>
    <li class="relative flex px-3">
        <Link
            class="group flex-1 px-3 py-2 rounded-md outline-hidden min-w-0 font-medium transition-colors ease-in-out"
            :class="{
                'text-primary-600 dark:text-primary-400': active,
                'text-gray-800 hover:bg-gray-100 hover:text-gray-950 focus:bg-gray-100 focus:text-gray-950 dark:text-dark-200 dark:hover:bg-dark-300/10 dark:hover:text-dark-50 dark:focus:bg-dark-300/10':
                    !active,
            }"
            :href="href"
            :as="as"
            :method="method"
        >
            <div class="flex items-center gap-3 min-w-0">
                <slot />
            </div>
            <div
                v-if="active"
                class="top-1 rtl:right-0 bottom-1 ltr:left-0 absolute bg-primary-600 dark:bg-primary-400 ltr:rounded-r-full rtl:rounded-l-lg w-1"
            ></div>
        </Link>
    </li>
</template>
