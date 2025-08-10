<script setup>
defineProps({ show: Boolean });

defineEmits(["close"]);
</script>

<template>
    <div
        v-if="show"
        @click.self="$emit('close')"
        class="z-40 fixed inset-0 bg-tailux-primary"
    ></div>
</template>
