import { buildBuildingColorMap } from "./state.js";

export function initSetupPanel({
    state,
    sceneController,
    showStatus,
    showLoading,
    saveEditIntents,
    switchMode,
    onMeshUpdated
}) {
    bindSliderValue("vertical-scale");
    bindSliderValue("model-width");
    bindSliderValue("base-height");
    bindSliderValue("boundary-padding");

    const gpxInput = document.getElementById("gpx-file");
    const geocodeBtn = document.getElementById("geocode-btn");
    const previewBtn = document.getElementById("generate-preview-btn");
    const finalBtn = document.getElementById("generate-final-btn");
    const quickPreviewBtn = document.getElementById("quick-preview-btn");
    const exportBtn = document.getElementById("export-3mf-btn");
    const quickExportBtn = document.getElementById("quick-export-btn");

    gpxInput.addEventListener("change", onGpxUpload);
    geocodeBtn.addEventListener("click", geocodeAddress);
    previewBtn.addEventListener("click", () => generateModel("preview"));
    finalBtn.addEventListener("click", () => generateModel("final"));
    quickPreviewBtn.addEventListener("click", () => generateModel("preview"));
    exportBtn.addEventListener("click", export3MF);
    quickExportBtn.addEventListener("click", export3MF);

    document.getElementById("boundary-padding").addEventListener("input", () => {
        if (state.originalBounds && state.gpxData) {
            state.gpxData.bounds = getPaddedBounds(state.originalBounds);
            state.needsRefetch = true;
        }
    });

    document.getElementById("show-only-address-building").addEventListener("change", () => {
        state.needsRefetch = true;
    });

    async function onGpxUpload(event) {
        const file = event.target.files?.[0];
        if (!file) {
            return;
        }

        document.getElementById("file-name").textContent = file.name;

        const formData = new FormData();
        formData.append("file", file);

        try {
            showLoading(true);
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || "Upload failed");
            }

            state.gpxData = result.data;
            state.originalBounds = state.gpxData.bounds ? { ...state.gpxData.bounds } : null;
            if (state.gpxData.bounds) {
                state.gpxData.bounds = getPaddedBounds(state.gpxData.bounds);
            }
            state.needsRefetch = true;

            showStatus("GPX loaded. Generate preview to enter edit mode.", "success");
        } catch (error) {
            showStatus(`Error uploading file: ${error.message}`, "error");
        } finally {
            showLoading(false);
        }
    }

    function extractCoordsFromGoogleMapsUrl(url) {
        const dataMatch = url.match(/!3d(-?\d+\.?\d*)!4d(-?\d+\.?\d*)/);
        if (dataMatch) {
            return {
                lat: Number.parseFloat(dataMatch[1]),
                lon: Number.parseFloat(dataMatch[2])
            };
        }

        const atMatch = url.match(/@(-?\d+\.?\d*),(-?\d+\.?\d*)/);
        if (atMatch) {
            return {
                lat: Number.parseFloat(atMatch[1]),
                lon: Number.parseFloat(atMatch[2])
            };
        }

        const qMatch = url.match(/[?&]q=(-?\d+\.?\d*),(-?\d+\.?\d*)/);
        if (qMatch) {
            return {
                lat: Number.parseFloat(qMatch[1]),
                lon: Number.parseFloat(qMatch[2])
            };
        }

        return null;
    }

    async function geocodeAddress() {
        const input = document.getElementById("address-input").value.trim();
        const button = document.getElementById("geocode-btn");
        const addressStatus = document.getElementById("address-status");

        if (!input) {
            showStatus("Enter an address or Google Maps URL", "error");
            return;
        }

        const originalText = button.textContent;
        button.disabled = true;
        button.textContent = "Searching...";

        try {
            let lat;
            let lon;
            let label;

            if (input.includes("google.com/maps") || input.includes("goo.gl/maps")) {
                const coords = extractCoordsFromGoogleMapsUrl(input);
                if (!coords) {
                    throw new Error("Could not extract coordinates from Google Maps URL");
                }
                lat = coords.lat;
                lon = coords.lon;
                label = `Coordinates from Google Maps: ${lat.toFixed(6)}, ${lon.toFixed(6)}`;
            } else {
                const response = await fetch("/api/geocode", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ address: input })
                });
                const result = await response.json();
                if (!result.success) {
                    throw new Error(result.error || "Geocoding failed");
                }
                lat = result.location.lat;
                lon = result.location.lon;
                label = `Address: ${result.location.address}`;
            }

            if (state.originalBounds && state.gpxData) {
                const maybeExpanded = expandBoundsForAddress(state.originalBounds, lat, lon);
                if (!maybeExpanded.success) {
                    throw new Error(maybeExpanded.error);
                }
                state.originalBounds = maybeExpanded.bounds;
                state.gpxData.bounds = getPaddedBounds(state.originalBounds);
                state.needsRefetch = true;
            }

            state.geocodedLocation = { lat, lon };
            sceneController.createAddressMarker(lat, lon);
            document.getElementById("show-only-address-building").disabled = false;
            addressStatus.textContent = label;
            showStatus("Location saved", "success");
        } catch (error) {
            showStatus(`Geocoding error: ${error.message}`, "error");
        } finally {
            button.disabled = false;
            button.textContent = originalText;
        }
    }

    function getPaddedBounds(bounds) {
        if (!bounds) {
            return null;
        }

        const paddingMeters = Number.parseFloat(document.getElementById("boundary-padding").value || "0");
        const avgLat = (bounds.north + bounds.south) / 2;
        const latPadding = paddingMeters / 111000;
        const lonPadding = paddingMeters / (111000 * Math.cos(avgLat * Math.PI / 180));

        return {
            north: bounds.north + latPadding,
            south: bounds.south - latPadding,
            east: bounds.east + lonPadding,
            west: bounds.west - lonPadding
        };
    }

    function getSquareBoundsForCircle(bounds) {
        if (!bounds) {
            return null;
        }

        const avgLat = (bounds.north + bounds.south) / 2;
        const lonScale = Math.cos(avgLat * Math.PI / 180);

        const latRangeMeters = (bounds.north - bounds.south) * 111000;
        const lonRangeMeters = (bounds.east - bounds.west) * 111000 * lonScale;
        const diagonal = Math.sqrt((latRangeMeters ** 2) + (lonRangeMeters ** 2));

        const centerLat = (bounds.north + bounds.south) / 2;
        const centerLon = (bounds.east + bounds.west) / 2;

        const halfSideLat = (diagonal / 2) / 111000;
        const halfSideLon = (diagonal / 2) / (111000 * lonScale);

        return {
            north: centerLat + halfSideLat,
            south: centerLat - halfSideLat,
            east: centerLon + halfSideLon,
            west: centerLon - halfSideLon
        };
    }

    function expandBoundsForAddress(originalBounds, lat, lon) {
        const maxExpansion = 0.1;

        const latDiff = Math.min(
            Math.abs(lat - originalBounds.south),
            Math.abs(lat - originalBounds.north)
        );
        const lonDiff = Math.min(
            Math.abs(lon - originalBounds.west),
            Math.abs(lon - originalBounds.east)
        );

        const outside = lat < originalBounds.south || lat > originalBounds.north ||
            lon < originalBounds.west || lon > originalBounds.east;

        if (outside && (latDiff > maxExpansion || lonDiff > maxExpansion)) {
            return {
                success: false,
                error: `Location too far from GPX area (${(Math.max(latDiff, lonDiff) * 111).toFixed(1)}km). Max expansion is ~11km.`
            };
        }

        const margin = 0.001;
        const expanded = {
            north: Math.max(originalBounds.north, lat + margin),
            south: Math.min(originalBounds.south, lat - margin),
            east: Math.max(originalBounds.east, lon + margin),
            west: Math.min(originalBounds.west, lon - margin)
        };

        return { success: true, bounds: expanded };
    }

    async function fetchElevationData(bounds, previewMode) {
        const resolution = previewMode ? 220 : 500;
        const response = await fetch("/api/elevation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                bounds,
                resolution,
                source_mode: "cesium",
                preview_mode: previewMode
            })
        });

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || "Failed elevation fetch");
        }

        state.elevationData = result.elevation;
        state.elevationFallbackReason = result.fallback_reason || null;
    }

    async function fetchOSMFeatures(bounds, previewMode) {
        const response = await fetch("/api/osm-features", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                bounds,
                features: ["roads", "water", "buildings", "railways", "landuse"]
            })
        });

        const result = await response.json();
        if (!result.success) {
            throw new Error(result.error || "Failed map feature fetch");
        }

        let features = result.features || {};
        if (previewMode) {
            features = {
                roads: (features.roads || []).slice(0, 120),
                water: (features.water || []).slice(0, 30),
                buildings: (features.buildings || []).slice(0, 120),
                railways: (features.railways || []).slice(0, 60),
                landuse: (features.landuse || []).slice(0, 30)
            };
        }

        state.osmFeatures = features;
    }

    async function generateModel(mode = "preview") {
        if (!state.gpxData?.bounds) {
            showStatus("Upload a GPX file first", "error");
            return;
        }

        const previewMode = mode !== "final";
        const modelShape = document.getElementById("model-shape").value;
        const modeLabel = previewMode ? "preview" : "final";

        try {
            showLoading(true);

            const shapeChanged = modelShape !== state.lastFetchShape;
            const modeChanged = modeLabel !== state.lastFetchMode;
            const shouldRefetch = state.needsRefetch || !state.elevationData || shapeChanged || modeChanged;

            if (shouldRefetch) {
                showStatus("Fetching terrain and feature data...", "info");
                let fetchBounds = state.gpxData.bounds;
                if (["square", "circle", "hexagon"].includes(modelShape)) {
                    fetchBounds = getSquareBoundsForCircle(state.gpxData.bounds);
                }

                await Promise.all([
                    fetchElevationData(fetchBounds, previewMode),
                    fetchOSMFeatures(fetchBounds, previewMode)
                ]);

                state.needsRefetch = false;
                state.lastFetchShape = modelShape;
                state.lastFetchMode = modeLabel;
            }

            const options = {
                vertical_scale: Number.parseFloat(document.getElementById("vertical-scale").value),
                model_width: Number.parseFloat(document.getElementById("model-width").value),
                base_height: Number.parseFloat(document.getElementById("base-height").value),
                include_base: document.getElementById("include-base").checked,
                model_shape: modelShape,
                building_mode: "hybrid",
                terrain_source_mode: "cesium",
                building_mesh_simplify: true,
                building_mesh_target_ratio_preview: 0.2,
                building_mesh_target_ratio_final: 0.4,
                gpx_tracks: state.gpxData?.tracks || [],
                address_location: state.geocodedLocation,
                show_only_address_building: document.getElementById("show-only-address-building").checked,
                custom_building_colors: buildBuildingColorMap(state.editIntents),
                excluded_feature_keys: Array.from(state.editIntents.deletedKeys),
                preview_mode: previewMode,
                generation_mode: modeLabel,
                _terrain_fallback_reason: state.elevationFallbackReason
            };

            showStatus("Generating 3D model...", "info");
            const response = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    elevation: state.elevationData,
                    features: state.osmFeatures || {},
                    options
                })
            });

            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || "Generate failed");
            }

            state.currentMesh = result.mesh;
            sceneController.renderMesh(result.mesh);
            const unmatched = sceneController.applyIntentsAndReport();

            saveEditIntents();
            if (typeof onMeshUpdated === "function") {
                onMeshUpdated(result.mesh);
            }

            if (unmatched.unmatchedDeleted || unmatched.unmatchedHidden || unmatched.unmatchedColors) {
                showStatus(
                    `Model generated. Unmatched saved edits: deleted=${unmatched.unmatchedDeleted}, hidden=${unmatched.unmatchedHidden}, colors=${unmatched.unmatchedColors}`,
                    "info"
                );
            } else {
                showStatus("Model generated successfully", "success");
            }

            switchMode("edit");
        } catch (error) {
            showStatus(`Generation error: ${error.message}`, "error");
        } finally {
            showLoading(false);
        }
    }

    async function export3MF() {
        if (!state.currentMesh || state.objectsByKey.size === 0) {
            showStatus("Generate a model before exporting", "error");
            return;
        }

        try {
            showLoading(true);
            showStatus("Exporting 3MF...", "info");

            const exportMesh = sceneController.buildExportMesh();
            const response = await fetch("/api/export/3mf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mesh: exportMesh,
                    filename: "topo_model.3mf"
                })
            });

            if (!response.ok) {
                const payload = await response.json();
                throw new Error(payload.error || "Export failed");
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement("a");
            anchor.href = url;
            anchor.download = "topo_model.3mf";
            anchor.click();
            URL.revokeObjectURL(url);

            showStatus("3MF exported", "success");
        } catch (error) {
            showStatus(`Export error: ${error.message}`, "error");
        } finally {
            showLoading(false);
        }
    }

    function bindSliderValue(id) {
        const slider = document.getElementById(id);
        const valueDisplay = document.getElementById(`${id}-value`);
        const update = () => {
            valueDisplay.textContent = slider.value;
        };
        slider.addEventListener("input", update);
        update();
    }

    return {
        generateModel,
        export3MF
    };
}
