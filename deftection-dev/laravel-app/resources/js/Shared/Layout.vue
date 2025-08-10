<script setup>
import Sidebar from "./Sidebar/Index.vue";
import MainHeader from "./MainHeader.vue";
import { computed } from "vue";
import { usePage } from "@inertiajs/vue3";

const page = usePage();

// Derive header title from current page component
const headerTitle = computed(() => {
    const component = page.component;

    const pageMapping = {
        "Dashboard/Index": "Dashboard",
        "ImageAnalysis/Index": "Image Analysis",
        "RealTimeAnalysis/Index": "Real-Time Analysis",
        "ScanHistory/Index": "Scan History",
        "ScanAnalysis/Index": "Detail Image Analysis",
        "History/Index": "Session History",
        "Database/Product/Index": "Products",
        "Database/User/Index": "Users",
        "Settings/Index": "Settings",
    };

    return pageMapping[component] || "Dashboard";
});
</script>

<template>
    <Head>
        <meta
            type="description"
            content="Information about my app"
            head-key="description"
        />
    </Head>

    <!-- Header -->
    <MainHeader>{{ headerTitle }}</MainHeader>

    <!-- Main Content -->
    <main class="grid grid-cols-1 transition-content main-content">
        <div
            class="transition-content mt-4 px-(--margin-x) pb-8 sm:mt-5 lg:mt-6"
        >
            <slot />
        </div>
    </main>
    <!-- Sidebar -->
    <Sidebar :user="$page.props.auth?.user"></Sidebar>
</template>
