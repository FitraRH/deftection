<script setup>
import VueApexCharts from "vue3-apexcharts";
import { ref } from "vue";

const series = ref([
    {
        name: "Defect Count",
        data: [44, 55, 57, 56, 61, 58, 63],
    },
    {
        name: "Images Processed",
        data: [76, 85, 101, 98, 87, 105, 91],
    },
]);

const chartOptions = ref({
    chart: {
        toolbar: {
            show: false,
        },
    },
    plotOptions: {
        bar: {
            horizontal: true,
            columnWidth: "55%",
            borderRadius: 5,
        },
    },
    dataLabels: {
        enabled: false,
    },
    stroke: {
        show: true,
        width: 2,
        colors: ["transparent"],
    },
    fill: {
        opacity: 1,
    },
    tooltip: {
        y: {
            formatter: (val) => `${val} images`,
        },
    },
    legend: {
        position: "top",
        horizontalAlign: "right",
        fontSize: "14px",
        markers: {
            radius: 12,
        },
        labels: {
            colors: "#cbd5e1", // optional: dark mode-friendly
        },
    },
    colors: ["#155dfc", "#009966"], // blue + green
    xaxis: {
        categories: [
            "Jul 4",
            "Jul 5",
            "Jul 6",
            "Jul 7",
            "Jul 8",
            "Jul 9",
            "Jul 10",
        ],
        labels: {
            style: {
                colors: "#cbd5e1", // optional: for dark UI
            },
        },
    },
});
</script>

<template>
    <div class="mx-auto w-full max-w-4xl">
        <VueApexCharts
            type="bar"
            height="500"
            :options="chartOptions"
            :series="series"
        />
    </div>
</template>
