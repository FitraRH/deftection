<script setup>
import ToolbarButton from "./ToolbarButton.vue";
import { defineProps, defineEmits } from "vue";

const emit = defineEmits(["click"]);

defineProps({
    label: {
        type: String,
        default: "Create",
    },
    disabled: {
        type: Boolean,
        default: false,
    },
    title: {
        type: String,
        default: "",
    },
});
</script>
<template>
    <ToolbarButton
        :label="label"
        @click="!disabled && emit('click')"
        :disabled="disabled"
        :title="title"
    >
        <template #icon>
            <font-awesome-icon icon="fa-solid fa-square-plus" />
        </template>
    </ToolbarButton>
</template>
