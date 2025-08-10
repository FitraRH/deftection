<script setup>
import VueApexCharts from "vue3-apexcharts";
import { ref } from "vue";

const series = ref([
    {
        name: "Defects Detected",
        data: [10, 20, 15, 30, 25, 40, 35],
    },
]);

const chartOptions = ref({
    colors: ["#a855f7"],
    chart: {
        dropShadow: {
            enabled: true,
            color: "#1E202C",
            top: 18,
            left: 6,
            blur: 8,
            opacity: 0.1,
        },
        toolbar: {
            show: false,
        },
    },
    stroke: {
        width: 8,
        curve: "smooth",
    },
    xaxis: {
        type: "datetime",
        categories: [
            "2000-11-01",
            "2000-11-02",
            "2000-11-03",
            "2000-11-04",
            "2000-11-05",
            "2000-11-06",
            "2000-11-07",
        ],
        tickAmount: 10,
        labels: {
            format: "dd MMM",
        },
    },
    fill: {
        type: "gradient",
        gradient: {
            shade: "dark",
            gradientToColors: ["#86efac"],
            shadeIntensity: 1,
            type: "horizontal",
            opacityFrom: 1,
            opacityTo: 0.95,
            stops: [0, 100, 0, 100],
        },
    },
    yaxis: {
        min: -10,
        max: 40,
    },
});
</script>

<template>
    <div class="mx-auto w-full">
        <VueApexCharts
            type="line"
            height="350"
            :options="chartOptions"
            :series="series"
        />
    </div>
</template>
