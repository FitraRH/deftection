<script setup>
import { Link } from "@inertiajs/vue3";
import { route } from "ziggy-js";

const props = defineProps({
    analyses: {
        type: Array,
        default: () => [],
    },
});
</script>

<template>
    <div class="transition-content flex grow flex-col">
        <table class="table w-full text-left rtl:text-right is-hoverable">
            <thead class="group/thead table-thead">
                <tr class="group/tr table-tr">
                    <th
                        class="group/th table-th bg-gray-200 dark:bg-dark-800 px-3 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg w-px font-semibold text-gray-800 dark:text-dark-100 uppercase"
                    >
                        Image
                    </th>
                    <th
                        class="group/th table-th bg-gray-200 dark:bg-dark-800 px-3 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg w-px font-semibold text-gray-800 dark:text-dark-100 uppercase"
                    >
                        Result
                    </th>
                    <th
                        class="group/th table-th bg-gray-200 dark:bg-dark-800 px-3 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg w-px font-semibold text-gray-800 dark:text-dark-100 uppercase"
                    >
                        Score
                    </th>
                    <th
                        class="group/th table-th bg-gray-200 dark:bg-dark-800 px-3 first:ltr:rounded-tl-lg last:rtl:rounded-tl-lg first:rtl:rounded-tr-lg last:ltr:rounded-tr-lg w-px font-semibold text-gray-800 dark:text-dark-100 uppercase"
                    >
                        Actions
                    </th>
                </tr>
            </thead>
            <tbody class="group/tbody table-tbody">
                <tr
                    v-for="(item, index) in analyses"
                    :key="index"
                    class="group/tr table-tr relative border-y border-transparent border-b-gray-200 dark:border-b-dark-500"
                >
                    <td
                        class="group/td table-td relative dark:bg-dark-900 py-5 pl-3"
                    >
                        <div class="flex items-center gap-2">
                            <font-awesome-icon
                                icon="fa-solid fa-image"
                                size="lg"
                            />
                            <div class="flex flex-col gap-1">
                                {{ item.imageName }}
                                <span class="text-xs">About 5 days ago</span>
                            </div>
                        </div>
                    </td>
                    <td
                        class="group/td table-td relative dark:bg-dark-900 py-5 pl-3"
                    >
                        <span
                            class="inline-block px-2 py-1 rounded font-semibold text-xs"
                            :class="
                                item.result === 'GOOD'
                                    ? 'bg-green-100 dark:bg-green-900/20 text-green-800'
                                    : 'bg-red-100 dark:bg-red-900/20 text-red-800'
                            "
                        >
                            {{ item.result }}
                        </span>
                    </td>
                    <td
                        class="group/td table-td relative dark:bg-dark-900 py-5 pl-3"
                    >
                        {{ item.score }}
                    </td>
                    <!-- route('view.detail.analysis', item.id) -->
                    <td
                        class="group/td table-td relative dark:bg-dark-900 py-5 pl-3"
                    >
                        <Link :href="route('scans.myscans')">
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 24 24"
                                fill="currentColor"
                                class="size-5 text-primary-500 hover:text-primary-600"
                            >
                                <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
                                <path
                                    fill-rule="evenodd"
                                    d="M1.323 11.447C2.811 6.976 7.028 3.75 12.001 3.75c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113-1.487 4.471-5.705 7.697-10.677 7.697-4.97 0-9.186-3.223-10.675-7.69a1.762 1.762 0 0 1 0-1.113ZM17.25 12a5.25 5.25 0 1 1-10.5 0 5.25 5.25 0 0 1 10.5 0Z"
                                    clip-rule="evenodd"
                                />
                            </svg>
                        </Link>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</template>
