<script setup>
const { meta } = defineProps({
    meta: {
        type: Object,
        required: true,
    },
});
</script>

<template>
    <div class="text-gray-400 text-sm">
        {{ meta.from || 0 }} - {{ meta.to || 0 }} of {{ meta.total }} entries
    </div>
</template>
