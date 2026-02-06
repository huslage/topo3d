export const STORAGE_KEY = "topo3d.session.v1";

export const DEFAULT_FILTERS = [
    "terrain",
    "building",
    "road",
    "water",
    "gpx_track",
    "marker",
    "label"
];

export function makeFeatureKey(type, id) {
    return `${type}:${String(id)}`;
}

function toArray(value) {
    return Array.isArray(value) ? value : [];
}

export function loadEditIntents() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            return {
                deletedKeys: new Set(),
                hiddenKeys: new Set(),
                colorOverrides: {},
                activeFilters: new Set(DEFAULT_FILTERS),
                searchQuery: ""
            };
        }

        const parsed = JSON.parse(raw);
        const activeFilters = new Set(toArray(parsed.active_filters));
        if (activeFilters.size === 0) {
            DEFAULT_FILTERS.forEach((f) => activeFilters.add(f));
        }

        return {
            deletedKeys: new Set(toArray(parsed.deleted_keys)),
            hiddenKeys: new Set(toArray(parsed.hidden_keys)),
            colorOverrides: parsed.color_overrides && typeof parsed.color_overrides === "object"
                ? parsed.color_overrides
                : {},
            activeFilters,
            searchQuery: typeof parsed.search_query === "string" ? parsed.search_query : ""
        };
    } catch (err) {
        console.warn("Failed to load edit intents", err);
        return {
            deletedKeys: new Set(),
            hiddenKeys: new Set(),
            colorOverrides: {},
            activeFilters: new Set(DEFAULT_FILTERS),
            searchQuery: ""
        };
    }
}

export function saveEditIntents(editIntents) {
    const payload = {
        deleted_keys: Array.from(editIntents.deletedKeys),
        hidden_keys: Array.from(editIntents.hiddenKeys),
        color_overrides: editIntents.colorOverrides,
        active_filters: Array.from(editIntents.activeFilters),
        search_query: editIntents.searchQuery
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

export function resetEditIntents() {
    localStorage.removeItem(STORAGE_KEY);
}

export function buildBuildingColorMap(editIntents) {
    const customBuildingColors = {};
    Object.entries(editIntents.colorOverrides).forEach(([featureKey, color]) => {
        if (!featureKey.startsWith("building:")) {
            return;
        }
        const buildingId = featureKey.split(":")[1];
        customBuildingColors[buildingId] = color;
    });
    return customBuildingColors;
}

export function createAppState() {
    return {
        mode: "setup",
        gpxData: null,
        originalBounds: null,
        elevationData: null,
        elevationFallbackReason: null,
        osmFeatures: null,
        geocodedLocation: null,
        currentMesh: null,
        lastFetchShape: "square",
        lastFetchMode: "preview",
        needsRefetch: true,
        editIntents: loadEditIntents(),
        selectedKeys: new Set(),
        objectsByKey: new Map(),
        objectOrder: [],
        boxSelectEnabled: false,
        sceneOrigin: { x: 0, y: 0, z: 0 },
        lastRenderedMeshData: null
    };
}
