<template>
    <div class="table-wrapper min-w-full overflow-x-auto grow">
        <table class="table w-full text-left rtl:text-right is-hoverable">
            <thead class="group/thead table-thead">
                <slot name="head"></slot>
            </thead>
            <tbody class="group/tbody table-tbody">
                <slot name="body"></slot>
            </tbody>
        </table>
    </div>
</template>
