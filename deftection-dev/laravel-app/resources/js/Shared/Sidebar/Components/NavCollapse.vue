<script setup>
import { ref, defineProps } from "vue";

defineProps({ active: Boolean });

const isOpen = ref(false);

const toggleMenu = () => {
    isOpen.value = !isOpen.value;
};

const onEnter = (el, done) => {
    // Get the actual height
    const height = el.scrollHeight;

    // Set initial state
    el.style.height = "0";
    el.style.overflow = "hidden";

    // Force reflow
    el.offsetHeight;

    // Animate to full height
    requestAnimationFrame(() => {
        el.style.height = height + "px";
    });

    // Listen for transition end
    el.addEventListener("transitionend", done, { once: true });
};

const onAfterEnter = (el) => {
    // Clean up inline styles and let CSS classes take over
    el.style.height = "auto";
    el.style.overflow = "";
};

const onLeave = (el, done) => {
    // Set current dimensions
    const height = el.scrollHeight;
    el.style.height = height + "px";
    el.style.overflow = "hidden";

    // Force reflow
    el.offsetHeight;

    // Animate to collapsed state
    requestAnimationFrame(() => {
        el.style.height = "0";
    });

    // Listen for transition end
    el.addEventListener("transitionend", done, { once: true });
};
</script>

<template>
    <li>
        <div
            class="relative flex flex-col flex-1 px-3 print:border break-words"
        >
            <button
                @click="toggleMenu"
                type="button"
                :class="[
                    'group cursor-pointer flex flex-1 items-center justify-between rounded-lg px-3 py-2 font-medium outline-hidden transition-colors duration-300 ease-in-out',
                    {
                        'text-primary-600 dark:text-primary-400 hover:bg-primary-600/10 hover:text-primary-700 dark:hover:bg-dark-300/10 dark:hover:text-primary-400':
                            active && !isOpen,
                        'text-gray-800 dark:text-dark-50': isOpen,
                        'text-gray-800 hover:text-gray-950 dark:text-dark-200 dark:hover:bg-dark-300/10':
                            !isOpen && !active,
                    },
                ]"
            >
                <div class="flex items-center gap-3 min-w-0">
                    <slot name="title"></slot>
                </div>
                <svg
                    :class="{
                        'rotate-90': isOpen,
                    }"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke-width="1.5"
                    stroke="currentColor"
                    aria-hidden="true"
                    data-slot="icon"
                    class="size-4 transition-transform shrink-0"
                >
                    <path
                        stroke-linecap="round"
                        stroke-linejoin="round"
                        d="m8.25 4.5 7.5 7.5-7.5 7.5"
                    ></path>
                </svg>
            </button>

            <!-- Collapsible Items -->
            <Transition
                name="collapse"
                @enter="onEnter"
                @after-enter="onAfterEnter"
                @leave="onLeave"
            >
                <div v-show="isOpen">
                    <div class="flex flex-col space-y-1 px-3 py-1.5">
                        <slot name="content"></slot>
                    </div>
                </div>
            </Transition>
        </div>
    </li>
</template>

<style>
.collapse-enter-active,
.collapse-leave-active {
    transition: height 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
</style>
