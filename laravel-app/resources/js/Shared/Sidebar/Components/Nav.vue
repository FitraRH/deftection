<script setup>
import NavLink from "./NavLink.vue";
import NavCollapse from "./NavCollapse.vue";
import NavCollapseLink from "./NavCollapseLink.vue";
import { route } from "ziggy-js";
</script>
<template>
    <nav class="pt-3">
        <div
            class="relative print:border break-words"
            aria-hidden="false"
            style="box-sizing: border-box"
        >
            <ul class="space-y-3">
                <NavLink
                    :href="route('dashboard')"
                    :active="$page.component === 'Dashboard/Index'"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        fill="none"
                        stroke-width="1.5"
                        viewBox="0 0 24 24"
                        class="stroke-[1.5] size-5 shrink-0"
                    >
                        <path
                            stroke="currentColor"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-miterlimit="10"
                            d="M2 22h20"
                        ></path>
                        <path
                            stroke="currentColor"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            d="M9.75 4v18h4.5V4c0-1.1-.45-2-1.8-2h-.9c-1.35 0-1.8.9-1.8 2ZM3 10v12h4V10c0-1.1-.4-2-1.6-2h-.8C3.4 8 3 8.9 3 10Zm14 5v7h4v-7c0-1.1-.4-2-1.6-2h-.8c-1.2 0-1.6.9-1.6 2Z"
                        ></path>
                    </svg>
                    <span class="truncate"> Dashboard</span>
                </NavLink>
                <NavCollapse
                    :active="
                        $page.component === 'ImageAnalysis/Index' ||
                        $page.component === 'ScanHistory/Index'
                    "
                >
                    <template #title>
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            class="stroke-[1.5] size-5 shrink-0"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Zm10.5-11.25h.008v.008h-.008V8.25Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z"
                            />
                        </svg>

                        <span class="truncate"> Singular / Batch Image </span>
                    </template>

                    <template #content>
                        <NavCollapseLink
                            :href="route('scans.create')"
                            :active="$page.component === 'ImageAnalysis/Index'"
                        >
                            New Scan
                        </NavCollapseLink>
                        <NavCollapseLink
                            :href="route('scans.myscans')"
                            :active="$page.component === 'ScanHistory/Index'"
                        >
                            Scan History
                        </NavCollapseLink>
                    </template>
                </NavCollapse>
                <NavCollapse
                    :active="
                        $page.component === 'RealTimeAnalysis/Index' ||
                        $page.component === 'SessionHistory/Index'
                    "
                >
                    <template #title>
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            class="stroke-[1.5] size-5 shrink-0"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
                            />
                        </svg>

                        <span class="truncate"> Live Monitor </span>
                    </template>

                    <template #content>
                        <NavCollapseLink
                            :href="route('sessions.create')"
                            :active="
                                $page.component === 'RealTimeAnalysis/Index'
                            "
                        >
                            New Session
                        </NavCollapseLink>
                        <NavCollapseLink
                            href="/history"
                            :active="$page.component === 'History/Index'"
                        >
                            Session History
                        </NavCollapseLink>
                    </template>
                </NavCollapse>
                <NavCollapse
                    :active="
                        $page.component === 'Database/Product/Index' ||
                        $page.component === 'Database/User/Index'
                    "
                >
                    <template #title>
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            class="stroke-[1.5] size-5 shrink-0"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125"
                            />
                        </svg>

                        Database
                    </template>

                    <template #content>
                        <NavCollapseLink
                            :href="route('products.index')"
                            :active="
                                $page.component === 'Database/Product/Index'
                            "
                        >
                            Products
                        </NavCollapseLink>
                        <NavCollapseLink
                            :href="route('users.index')"
                            :active="$page.component === 'Database/User/Index'"
                        >
                            Users
                        </NavCollapseLink>
                    </template>
                </NavCollapse>
                <NavLink
                    href="/settings"
                    :active="$page.component === 'Settings/Index'"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke-width="1.5"
                        stroke="currentColor"
                        class="opacity-80 group-hover:opacity-100 stroke-[1.5] size-5 shrink-0"
                    >
                        <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            d="M11.42 15.17 17.25 21A2.652 2.652 0 0 0 21 17.25l-5.877-5.877M11.42 15.17l2.496-3.03c.317-.384.74-.626 1.208-.766M11.42 15.17l-4.655 5.653a2.548 2.548 0 1 1-3.586-3.586l6.837-5.63m5.108-.233c.55-.164 1.163-.188 1.743-.14a4.5 4.5 0 0 0 4.486-6.336l-3.276 3.277a3.004 3.004 0 0 1-2.25-2.25l3.276-3.276a4.5 4.5 0 0 0-6.336 4.486c.091 1.076-.071 2.264-.904 2.95l-.102.085m-1.745 1.437L5.909 7.5H4.5L2.25 3.75l1.5-1.5L7.5 4.5v1.409l4.26 4.26m-1.745 1.437 1.745-1.437m6.615 8.206L15.75 15.75M4.867 19.125h.008v.008h-.008v-.008Z"
                        ></path>
                    </svg>
                    Settings
                </NavLink>
                <NavLink :href="route('logout')" method="post" as="button">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke-width="1.5"
                        stroke="currentColor"
                        class="opacity-80 group-hover:opacity-100 stroke-[1.5] size-5 shrink-0"
                    >
                        <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            d="M8.25 9V5.25A2.25 2.25 0 0 1 10.5 3h6a2.25 2.25 0 0 1 2.25 2.25v13.5A2.25 2.25 0 0 1 16.5 21h-6a2.25 2.25 0 0 1-2.25-2.25V15m-3 0-3-3m0 0 3-3m-3 3H15"
                        />
                    </svg>

                    <span class="truncate">Logout</span>
                </NavLink>
            </ul>
        </div>
    </nav>
</template>
