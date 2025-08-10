// utils/swal.js
import Swal from "sweetalert2";

// 1. Unsaved Changes Dialog
const MinimalSwal = Swal.mixin({
    buttonsStyling: false,
    background: "transparent",
    backdrop: "rgba(0, 0, 0, 0.4)",
    showClass: { popup: "" },
    hideClass: { popup: "" },
    customClass: {
        popup: "!bg-white dark:!bg-dark-800 rounded-md shadow-lg border !border-gray-200 dark:!border-gray-800 p-6 max-w-xs",
        title: "text-base font-medium !text-gray-900 dark:!text-white mb-2",
        htmlContainer: "text-sm !text-gray-600 dark:!text-gray-400 mb-4",
        actions: "flex gap-2 w-full",
        confirmButton:
            "font-medium bg-red-600 hover:bg-red-700 text-white text-sm px-3 py-1.5 rounded transition-colors",
        cancelButton:
            "!bg-gray-200 hover:!bg-gray-300 dark:!bg-gray-700 dark:!hover:!bg-gray-600 !text-gray-700 dark:!text-white text-sm px-3 py-1.5 rounded transition-colors",
    },
});

export const unsavedChangesDialog = () => {
    return MinimalSwal.fire({
        title: "Discard changes?",
        text: "Unsaved changes will be lost.",
        showCancelButton: true,
        confirmButtonText: "Discard",
        cancelButtonText: "Cancel",
        allowOutsideClick: true,
        focusCancel: true,
        reverseButtons: true,
    });
};

// 2. Toast Notifications
const getToastBase = () => {
    const isDark = document.documentElement.classList.contains("dark");
    return Swal.mixin({
        toast: true,
        position: "top-end",
        showConfirmButton: false,
        timerProgressBar: true,
        background: isDark ? "#1f2937" : "#ffffff",
        color: isDark ? "#f3f4f6" : "#1f2937",
    });
};

export const successToast = (title = "Success!", timer = 3000) => {
    const Toast = getToastBase();
    return Toast.fire({
        icon: "success",
        title,
        toast: true,
        position: "bottom-end",
        showConfirmButton: false,
        timer,
        timerProgressBar: true,
        background:
            localStorage.getItem("theme") === "dark" ? "#1c1d21" : "#ffffff",
        color: localStorage.getItem("theme") === "dark" ? "#f9fafb" : "#1f2937",
        iconColor:
            localStorage.getItem("theme") === "dark" ? "#10b981" : "#34d399",
    });
};

export const errorToast = (title = "Something went wrong!", timer = 4000) => {
    const Toast = getToastBase();
    return Toast.fire({
        icon: "error",
        title,
        timer,
    });
};

export const infoToast = (title = "Information", timer = 3000) => {
    const Toast = getToastBase();
    return Toast.fire({
        icon: "info",
        title,
        timer,
    });
};

// 3. Edit Confirmation
const EditSwal = Swal.mixin({
    buttonsStyling: false,
    background: "transparent",
    backdrop: "rgba(0, 0, 0, 0.4)",
    showClass: { popup: "" },
    hideClass: { popup: "" },
    customClass: {
        popup: "!bg-white dark:!bg-dark-800 rounded-md shadow-lg border !border-gray-200 dark:!border-gray-800 p-6 max-w-sm",
        title: "text-base font-medium !text-gray-900 dark:!text-white mb-2",
        htmlContainer: "text-sm !text-gray-600 dark:!text-gray-400 mb-4",
        actions: "flex gap-2 w-full",
        confirmButton:
            "font-medium bg-blue-600 hover:bg-blue-700 text-white text-sm px-3 py-1.5 rounded transition-colors",
        cancelButton:
            "!bg-gray-200 hover:!bg-gray-300 dark:!bg-gray-700 dark:hover:!bg-gray-600 !text-gray-700 dark:!text-white text-sm px-3 py-1.5 rounded transition-colors",
    },
});

export const editConfirmDialog = (itemName = "this item") => {
    return EditSwal.fire({
        title: "Edit item?",
        text: `Are you sure you want to edit ${itemName}?`,
        showCancelButton: true,
        confirmButtonText: "Edit",
        cancelButtonText: "Cancel",
        allowOutsideClick: true,
        focusCancel: true,
        reverseButtons: true,
    });
};

// 4. Delete Confirmation
const DeleteSwal = Swal.mixin({
    buttonsStyling: false,
    background: "transparent",
    backdrop: "rgba(0, 0, 0, 0.4)",
    showClass: { popup: "" },
    hideClass: { popup: "" },
    customClass: {
        popup: "bg-white dark:!bg-dark-800 rounded-md shadow-lg border border-gray-200 dark:!border-gray-800 p-6 max-w-sm",
        title: "text-base font-medium text-gray-900 dark:!text-white mb-2",
        htmlContainer: "text-sm text-gray-600 dark:!text-gray-400 mb-4",
        actions: "flex gap-2 w-full",
        confirmButton:
            "font-medium bg-red-600 hover:bg-red-700 text-white text-sm px-3 py-1.5 rounded transition-colors",
        cancelButton:
            "bg-gray-200 hover:bg-gray-300 dark:!bg-gray-700 dark:hover:!bg-gray-600 text-gray-700 dark:!text-white text-sm px-3 py-1.5 rounded transition-colors",
    },
});

export const deleteConfirmDialog = (itemName = "this item") => {
    return DeleteSwal.fire({
        title: "Delete item?",
        text: `Are you sure you want to delete ${itemName}? This action cannot be undone.`,
        showCancelButton: true,
        confirmButtonText: "Delete",
        cancelButtonText: "Cancel",
        allowOutsideClick: true,
        focusCancel: true,
        reverseButtons: true,
    });
};

// 5. Create Confirmation
const CreateSwal = Swal.mixin({
    buttonsStyling: false,
    background: "transparent",
    backdrop: "rgba(0, 0, 0, 0.4)",
    showClass: { popup: "" },
    hideClass: { popup: "" },
    customClass: {
        popup: "bg-white dark:!bg-dark-800 rounded-md shadow-lg border border-gray-200 dark:!border-gray-800 p-6 max-w-sm",
        title: "text-base font-medium text-gray-900 dark:!text-white mb-2",
        htmlContainer: "text-sm text-gray-600 dark:!text-gray-400 mb-4",
        actions: "flex gap-2 w-full",
        confirmButton:
            "font-medium bg-green-600 hover:bg-green-700 text-white text-sm px-3 py-1.5 rounded transition-colors",
        cancelButton:
            "bg-gray-200 hover:bg-gray-300 dark:!bg-gray-700 dark:hover:!bg-gray-600 text-gray-700 dark:!text-white text-sm px-3 py-1.5 rounded transition-colors",
    },
});

export const createConfirmDialog = (itemName = "this item") => {
    return CreateSwal.fire({
        title: "Create item?",
        text: `Are you sure you want to create ${itemName}?`,
        showCancelButton: true,
        confirmButtonText: "Create",
        cancelButtonText: "Cancel",
        allowOutsideClick: true,
        focusCancel: true,
        reverseButtons: true,
    });
};

export default Swal;
