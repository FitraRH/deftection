<script setup>
defineProps({ active: Boolean, href: String });
</script>

<template>
    <div class="relative flex pt-2">
        <Link
            class="group flex-1 px-3 py-2 rounded-md outline-hidden min-w-0 font-medium transition-colors ease-in-out"
            :class="{
                'text-primary-600 dark:text-primary-400': active,
                'text-gray-800 hover:bg-gray-100 hover:text-gray-950 focus:bg-gray-100 focus:text-gray-950 dark:text-dark-200 dark:hover:bg-dark-300/10 dark:hover:text-dark-50 dark:focus:bg-dark-300/10':
                    !active,
            }"
            :href="href"
        >
            <div class="flex justify-between items-center gap-2.5 min-w-0">
                <div class="flex items-center gap-3 min-w-0">
                    <div
                        v-if="active"
                        class="bg-primary-600 dark:bg-primary-400 opacity-80 border border-current rounded-full size-2"
                    ></div>
                    <div
                        v-else
                        class="opacity-50 border border-current rounded-full size-2 transition-all"
                    ></div>
                    <slot />
                </div>
            </div>
        </Link>
    </div>
</template>
