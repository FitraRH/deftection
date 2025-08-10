<script setup>
import { Menu, MenuButton, MenuItems } from "@headlessui/vue";

defineProps({
    disabled: {
        type: Boolean,
        default: false,
    },
});
</script>
<template>
    <Menu as="div" class="inline-block relative text-left">
        <MenuButton
            :disabled="disabled"
            class="border-0 btn btn-base btn-style"
        >
            <font-awesome-icon icon="fa-solid fa-ellipsis" />
        </MenuButton>

        <Transition
            enter-active-class="transition ease-out duration-100"
            enter-from-class="opacity-0 translate-y-2"
            enter-to-class="opacity-100 translate-y-0"
            leave-active-class="transition ease-in duration-75"
            leave-from-class="opacity-100 translate-y-0"
            leave-to-class="opacity-0 translate-y-2"
        >
            <MenuItems
                class="right-0 z-[100] absolute bg-white dark:bg-dark-800 shadow-gray-200/50 shadow-lg dark:shadow-none mt-1.5 py-1 border border-gray-200 dark:border-dark-500 rounded-lg outline-none focus-visible:outline-none min-w-[10rem] text-gray-800 dark:text-dark-100"
            >
                <slot />
            </MenuItems>
        </Transition>
    </Menu>
</template>
