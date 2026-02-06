import { createAppState, saveEditIntents } from "./state.js";
import { createSceneController } from "./scene-controller.js";
import { HistoryStack } from "./history.js";
import { initEditorPanel } from "./editor-panel.js";
import { initSetupPanel } from "./setup-panel.js";

const state = createAppState();
let statusHideTimer = null;

function showStatus(message, type = "info") {
    const statusEl = document.getElementById("status-message");
    statusEl.textContent = message;
    statusEl.className = `status-floating ${type}`;
    statusEl.style.display = "block";

    if (statusHideTimer) {
        clearTimeout(statusHideTimer);
    }
    statusHideTimer = setTimeout(() => {
        statusEl.style.display = "none";
    }, 9000);
}

function showLoading(show) {
    document.getElementById("loading").classList.toggle("active", show);
}

function saveCurrentIntents() {
    saveEditIntents(state.editIntents);
}

function setMode(mode) {
    state.mode = mode;
    document.body.dataset.mode = mode;

    const setupPanel = document.getElementById("setup-mode-panel");
    const exportPanel = document.getElementById("export-mode-panel");
    const editPanel = document.getElementById("edit-mode-panel");

    setupPanel.classList.toggle("active", mode !== "export");
    exportPanel.classList.toggle("active", mode === "export");
    editPanel.classList.toggle("active", mode === "edit");

    document.querySelectorAll(".mode-tab").forEach((tab) => {
        const active = tab.dataset.mode === mode;
        tab.classList.toggle("active", active);
    });

    if (sceneController?.refreshViewport) {
        sceneController.refreshViewport();
    }
}

const history = new HistoryStack(100);
let editorApi = null;

const sceneController = createSceneController({
    state,
    showStatus,
    onSelectionChanged: () => {
        if (editorApi) {
            editorApi.handleSelectionChanged();
        }
    },
    onObjectsChanged: () => {
        if (editorApi) {
            editorApi.handleObjectsChanged();
        }
    }
});

window.addEventListener("DOMContentLoaded", () => {
    sceneController.init();

    editorApi = initEditorPanel({
        state,
        sceneController,
        history,
        showStatus,
        saveEditIntents: saveCurrentIntents
    });

    initSetupPanel({
        state,
        sceneController,
        showStatus,
        showLoading,
        saveEditIntents: saveCurrentIntents,
        switchMode: setMode,
        onMeshUpdated: () => {
            if (editorApi) {
                editorApi.renderObjectList();
            }
        }
    });

    document.querySelectorAll(".mode-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            if (tab.dataset.mode === "edit" && !state.currentMesh) {
                showStatus("Generate a model before entering Edit mode", "info");
                return;
            }
            setMode(tab.dataset.mode);
        });
    });

    setMode("setup");
    saveCurrentIntents();
});
