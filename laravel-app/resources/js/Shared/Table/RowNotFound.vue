<script setup>
import { defineProps } from "vue";
defineProps({ label: { type: String, default: "No results found" } });
</script>
<template>
    <tr>
        <td colspan="6" class="px-6 py-12 text-center">
            <div class="text-gray-400">
                <svg
                    class="mx-auto w-12 h-12"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        stroke-width="1"
                        d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                </svg>
                <p class="mt-2 font-medium text-sm">{{ label }}</p>
                <p class="mt-1 text-xs">
                    Try adjusting your search or filter criteria.
                </p>
            </div>
        </td>
    </tr>
</template>
