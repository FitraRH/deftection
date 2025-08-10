<script setup>
import VueApexCharts from "vue3-apexcharts";
import { ref } from "vue";

const series = ref([10, 5, 2, 4]); // Mock data

const chartOptions = ref({
    labels: ["Crack", "Dent", "Scratch", "Opened"], // KEEP - This is data
    fill: {
        // KEEP - This is specific to this chart's look
        opacity: 1,
    },
    stroke: {
        // KEEP - Also specific styling
        width: 1,
        colors: undefined,
    },
    yaxis: {
        // KEEP - This is a structural change
        show: false,
    },
    legend: {
        position: "bottom", // KEEP - This is layout
        horizontalAlign: "center", // KEEP - This is layout
        labels: {
            // REMOVE THIS - Your new CSS file now controls the legend text color
            // colors: "#cbd5e1",
        },
    },
    plotOptions: {
        // KEEP - This is structural
        polarArea: {
            rings: {
                strokeWidth: 0,
            },
            spokes: {
                strokeWidth: 0,
            },
        },
    },
    theme: {
        // KEEP - The monochrome theme generates the color palette
        monochrome: {
            enabled: true,
            color: "#155dfc",
            shadeTo: "light",
            shadeIntensity: 0.65,
        },
    },
});
</script>

<template>
    <div class="mx-auto w-full max-w-4xl">
        <VueApexCharts
            type="pie"
            height="500"
            :options="chartOptions"
            :series="series"
        />
    </div>
</template>
