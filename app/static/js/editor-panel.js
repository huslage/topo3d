import {
    DeleteKeysCommand,
    HideKeysCommand,
    RestoreAllCommand,
    SetColorCommand
} from "./history.js";

export function initEditorPanel({
    state,
    sceneController,
    history,
    showStatus,
    saveEditIntents
}) {
    const objectSearch = document.getElementById("object-search");
    const objectList = document.getElementById("object-list");
    const resultsCount = document.getElementById("results-count");
    const selectionCount = document.getElementById("selection-count");
    const chipContainer = document.getElementById("filter-chips");

    const selectVisibleBtn = document.getElementById("select-visible-btn");
    const clearSelectionBtn = document.getElementById("clear-selection-btn");
    const deleteSelectedBtn = document.getElementById("delete-selected-btn");
    const hideSelectedBtn = document.getElementById("hide-selected-btn");
    const showAllBtn = document.getElementById("show-all-btn");
    const undoBtn = document.getElementById("undo-btn");
    const redoBtn = document.getElementById("redo-btn");
    const toggleBoxSelectBtn = document.getElementById("toggle-box-select-btn");
    const addLabelBtn = document.getElementById("add-label-btn");

    objectSearch.value = state.editIntents.searchQuery;
    objectSearch.addEventListener("input", () => {
        state.editIntents.searchQuery = objectSearch.value.trim().toLowerCase();
        saveEditIntents();
        renderObjectList();
    });

    chipContainer.querySelectorAll(".chip").forEach((chip) => {
        const filter = chip.dataset.filter;
        const active = state.editIntents.activeFilters.has(filter);
        chip.classList.toggle("active", active);

        chip.addEventListener("click", () => {
            if (state.editIntents.activeFilters.has(filter)) {
                state.editIntents.activeFilters.delete(filter);
            } else {
                state.editIntents.activeFilters.add(filter);
            }
            chip.classList.toggle("active", state.editIntents.activeFilters.has(filter));
            saveEditIntents();
            renderObjectList();
        });
    });

    addLabelBtn.addEventListener("click", () => {
        const input = document.getElementById("label-text");
        const text = input.value.trim();
        if (!text) {
            showStatus("Enter text for label", "error");
            return;
        }
        sceneController.addLabel(text);
        input.value = "";
        saveEditIntents();
        renderObjectList();
    });

    selectVisibleBtn.addEventListener("click", () => {
        const keys = getVisibleRecords().map((record) => record.key);
        sceneController.setSelection(keys, { replace: true });
        renderObjectList();
    });

    clearSelectionBtn.addEventListener("click", () => {
        sceneController.clearSelection();
        renderObjectList();
    });

    deleteSelectedBtn.addEventListener("click", () => {
        const selected = sceneController.getSelectedKeys();
        if (selected.length === 0) {
            showStatus("No objects selected", "error");
            return;
        }

        const confirmed = confirm(`Delete ${selected.length} selected object(s)?`);
        if (!confirmed) {
            return;
        }

        history.execute(new DeleteKeysCommand(commandContext, selected));
        showStatus(`Deleted ${selected.length} objects`, "success");
    });

    hideSelectedBtn.addEventListener("click", () => {
        const selected = sceneController.getSelectedKeys();
        if (selected.length === 0) {
            showStatus("No objects selected", "error");
            return;
        }

        history.execute(new HideKeysCommand(commandContext, selected));
        showStatus(`Hidden ${selected.length} objects`, "info");
    });

    showAllBtn.addEventListener("click", () => {
        history.execute(new RestoreAllCommand(commandContext));
        showStatus("Hidden objects restored", "success");
    });

    undoBtn.addEventListener("click", () => {
        if (!history.undo()) {
            showStatus("Nothing to undo", "info");
        }
    });

    redoBtn.addEventListener("click", () => {
        if (!history.redo()) {
            showStatus("Nothing to redo", "info");
        }
    });

    toggleBoxSelectBtn.addEventListener("click", () => {
        state.boxSelectEnabled = !state.boxSelectEnabled;
        sceneController.setBoxSelectEnabled(state.boxSelectEnabled);
        toggleBoxSelectBtn.textContent = `Box Select: ${state.boxSelectEnabled ? "On" : "Off"}`;
        toggleBoxSelectBtn.classList.toggle("active", state.boxSelectEnabled);
    });

    document.addEventListener("keydown", (event) => {
        if (shouldIgnoreShortcuts(event.target)) {
            return;
        }

        const key = event.key.toLowerCase();

        if ((event.metaKey || event.ctrlKey) && key === "z") {
            event.preventDefault();
            if (event.shiftKey) {
                history.redo();
            } else {
                history.undo();
            }
            return;
        }

        if (key === "delete" || key === "backspace") {
            event.preventDefault();
            deleteSelectedBtn.click();
            return;
        }

        if (key === "a" && !event.metaKey && !event.ctrlKey) {
            event.preventDefault();
            selectVisibleBtn.click();
        }
    });

    history.onChange = ({ canUndo, canRedo }) => {
        undoBtn.disabled = !canUndo;
        redoBtn.disabled = !canRedo;
    };
    history.onChange({ canUndo: history.canUndo(), canRedo: history.canRedo() });

    function shouldIgnoreShortcuts(target) {
        if (!target) {
            return false;
        }
        const tag = target.tagName?.toLowerCase();
        return tag === "input" || tag === "textarea" || target.isContentEditable;
    }

    const commandContext = {
        setDeleted(keys, value) {
            sceneController.setDeleted(keys, value);
            saveEditIntents();
            renderObjectList();
        },
        setHidden(keys, value) {
            sceneController.setHidden(keys, value);
            saveEditIntents();
            renderObjectList();
        },
        setColor(key, color) {
            sceneController.setColor(key, color);
            if (!color) {
                delete state.editIntents.colorOverrides[key];
            } else {
                state.editIntents.colorOverrides[key] = color;
            }
            saveEditIntents();
            renderObjectList();
        },
        getColor(key) {
            return state.editIntents.colorOverrides[key] || null;
        },
        getHiddenSnapshot() {
            return sceneController.getHiddenSnapshot();
        },
        replaceHidden(snapshot) {
            sceneController.replaceHidden(snapshot);
            saveEditIntents();
            renderObjectList();
        },
        restoreAllHidden() {
            sceneController.restoreAllHidden();
            saveEditIntents();
            renderObjectList();
        }
    };

    function getFilteredRecords() {
        const query = state.editIntents.searchQuery || "";
        const activeFilters = state.editIntents.activeFilters;

        return sceneController.getObjectRecords().filter((record) => {
            if (record.deleted) {
                return false;
            }
            if (activeFilters.size > 0 && !activeFilters.has(record.type)) {
                return false;
            }
            if (!query) {
                return true;
            }
            const haystack = `${record.name} ${record.type} ${record.id}`.toLowerCase();
            return haystack.includes(query);
        });
    }

    function getVisibleRecords() {
        return getFilteredRecords().filter((record) => !record.hidden);
    }

    function typeColor(type) {
        const map = {
            terrain: "#8b7355",
            building: "#8f8f8f",
            road: "#3e3e3e",
            water: "#327fc9",
            gpx_track: "#d02e2e",
            marker: "#e84f3f",
            label: "#5468aa"
        };
        return map[type] || "#7a7a7a";
    }

    function displayType(type) {
        const map = {
            gpx_track: "gpx"
        };
        return (map[type] || type).replace(/_/g, " ");
    }

    function renderObjectList() {
        const records = getFilteredRecords();
        const selectedKeys = new Set(sceneController.getSelectedKeys());

        if (records.length === 0) {
            objectList.innerHTML = "<div class=\"object-row\"><span class=\"object-name\">No objects match current filters</span></div>";
        } else {
            objectList.innerHTML = "";
            records.forEach((record) => {
                const row = document.createElement("div");
                row.className = "object-row";
                if (selectedKeys.has(record.key)) {
                    row.classList.add("selected");
                }
                if (record.deleted) {
                    row.classList.add("deleted");
                }

                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.checked = selectedKeys.has(record.key);
                checkbox.addEventListener("change", () => {
                    sceneController.toggleSelection(record.key);
                    renderObjectList();
                });

                const typeBadge = document.createElement("span");
                typeBadge.className = "object-type";
                typeBadge.style.background = typeColor(record.type);
                typeBadge.textContent = displayType(record.type);

                const name = document.createElement("span");
                name.className = "object-name";
                name.textContent = record.name;

                row.appendChild(checkbox);
                row.appendChild(typeBadge);
                row.appendChild(name);

                if (record.type === "building") {
                    const colorInput = document.createElement("input");
                    colorInput.className = "color-inline";
                    colorInput.type = "color";
                    colorInput.value = state.editIntents.colorOverrides[record.key] || "#aaaaaa";
                    colorInput.title = "Building color";
                    colorInput.addEventListener("input", (event) => {
                        history.execute(new SetColorCommand(commandContext, record.key, event.target.value));
                        showStatus("Building color updated", "success");
                    });
                    row.appendChild(colorInput);
                }

                row.addEventListener("click", (event) => {
                    if (event.target.tagName?.toLowerCase() === "input") {
                        return;
                    }
                    sceneController.toggleSelection(record.key);
                    renderObjectList();
                });

                objectList.appendChild(row);
            });
        }

        resultsCount.textContent = `${records.length} results`;
        selectionCount.textContent = `${sceneController.getSelectedKeys().length} selected`;
    }

    function handleSelectionChanged() {
        renderObjectList();
    }

    function handleObjectsChanged() {
        renderObjectList();
    }

    renderObjectList();

    return {
        renderObjectList,
        handleSelectionChanged,
        handleObjectsChanged
    };
}
