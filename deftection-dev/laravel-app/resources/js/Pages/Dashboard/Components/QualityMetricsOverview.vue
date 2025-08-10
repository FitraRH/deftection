<script setup>
import VueApexCharts from "vue3-apexcharts";
import { ref } from "vue";

const series = ref([
    {
        name: "Anomalib",
        data: [85, 70, 90, 80, 75], // Accuracy, Speed, Reliability, Precision, Coverage
    },
    {
        name: "HRNet",
        data: [90, 60, 85, 88, 70],
    },
]);

const chartOptions = ref({
    chart: {
        type: "radar",
        toolbar: { show: false },
    },
    xaxis: {
        categories: [
            "Accuracy",
            "Speed",
            "Reliability",
            "Precision",
            "Coverage",
        ],
        labels: {
            style: {
                colors: "#cbd5e1",
            },
        },
    },
    stroke: {
        width: 2,
    },
    fill: {
        opacity: 0.25,
    },
    markers: {
        size: 4,
        strokeWidth: 2,
        strokeColor: "#fff",
    },
    colors: ["#3b82f6", "#8b5cf6"], // Anomalib = blue, HRNet = violet
    legend: {
        show: true,
        position: "top",
        labels: {
            colors: "#cbd5e1",
        },
    },
    yaxis: {
        min: 0,
        max: 100,
        tickAmount: 5,
        labels: {
            style: {
                colors: "#94a3b8",
            },
        },
    },
});
</script>
<template>
    <div class="w-full max-w-4xl mx-auto">
        <VueApexCharts
            type="radar"
            height="350"
            :series="series"
            :options="chartOptions"
        />
    </div>
</template>
