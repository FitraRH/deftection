<script setup>
import { ref } from "vue";

defineProps({
    message: String,
});

const count = ref(0);
const increment = () => {
    count.value++;
};
</script>

<template>
    <div class="container">
        <h1 class="">Welcome to Inertia.js</h1>
        <p class="text-gray-600">{{ message }}</p>
        <p>Counter: {{ count }}</p>
        <button @click="increment">Increase</button>
    </div>
</template>

<style scoped>
.container {
    text-align: center;
    margin-top: 50px;
}

h1 {
    color: #3498db;
}

button {
    padding: 10px 20px;
    margin-top: 10px;
    background-color: #2ecc71;
    color: white;
    border: none;
    cursor: pointer;
}
</style>
