import { makeFeatureKey } from "./state.js";

export function createSceneController({ state, showStatus, onSelectionChanged, onObjectsChanged }) {
    let scene;
    let camera;
    let renderer;
    let controls;
    let canvasContainer;
    let selectionBox;
    let resizeObserver = null;
    let viewportSyncFrame = 0;
    const viewportSize = { width: 0, height: 0 };

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const boxSelectState = {
        active: false,
        startX: 0,
        startY: 0
    };

    function init() {
        canvasContainer = document.getElementById("canvas-container");
        selectionBox = document.getElementById("selection-box");

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111715);
        scene.fog = new THREE.Fog(0x111715, 450, 2200);

        const initialWidth = Math.max(1, Math.floor(canvasContainer.clientWidth || 1));
        const initialHeight = Math.max(1, Math.floor(canvasContainer.clientHeight || 1));
        camera = new THREE.PerspectiveCamera(58, initialWidth / initialHeight, 0.1, 6000);
        camera.position.set(210, 210, 210);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.setSize(initialWidth, initialHeight, false);
        renderer.shadowMap.enabled = true;
        renderer.domElement.style.display = "block";
        renderer.domElement.style.width = "100%";
        renderer.domElement.style.height = "100%";
        canvasContainer.appendChild(renderer.domElement);

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.62);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.84);
        directionalLight.position.set(250, 530, 130);
        directionalLight.castShadow = true;
        scene.add(directionalLight);

        const gridHelper = new THREE.GridHelper(1200, 52, 0x334437, 0x1d2721);
        scene.add(gridHelper);

        window.addEventListener("resize", onWindowResize);
        if (typeof ResizeObserver === "function") {
            resizeObserver = new ResizeObserver(() => {
                requestViewportSync();
            });
            resizeObserver.observe(canvasContainer);
        }
        renderer.domElement.addEventListener("pointerdown", onPointerDown, true);
        document.addEventListener("pointermove", onPointerMove, true);
        document.addEventListener("pointerup", onPointerUp, true);

        requestViewportSync();
        animate();
    }

    function animate() {
        requestAnimationFrame(animate);
        if (controls) {
            controls.update();
        }
        if (renderer && scene && camera) {
            renderer.render(scene, camera);
        }
        updateDiagnostics();
    }

    function onWindowResize() {
        if (renderer) {
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        }
        requestViewportSync();
    }

    function syncViewportSize() {
        if (!canvasContainer || !camera || !renderer) {
            return false;
        }

        const width = Math.max(1, Math.floor(canvasContainer.clientWidth || 1));
        const height = Math.max(1, Math.floor(canvasContainer.clientHeight || 1));
        if (width === viewportSize.width && height === viewportSize.height) {
            return false;
        }

        viewportSize.width = width;
        viewportSize.height = height;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
        return true;
    }

    function requestViewportSync() {
        if (viewportSyncFrame) {
            cancelAnimationFrame(viewportSyncFrame);
        }

        viewportSyncFrame = requestAnimationFrame(() => {
            viewportSyncFrame = requestAnimationFrame(() => {
                viewportSyncFrame = 0;
                const didResize = syncViewportSize();
                if (didResize) {
                    forceRenderRefresh();
                }
            });
        });
    }

    function clearSceneObjects() {
        while (scene.children.length > 3) {
            scene.remove(scene.children[3]);
        }

        state.objectsByKey = new Map();
        state.objectOrder = [];
        state.selectedKeys.clear();
        emitSelectionChange();
        emitObjectsChange();
    }

    function parseColorValue(value, fallback = 0x888888) {
        if (typeof value === "number" && Number.isFinite(value)) {
            return value;
        }
        if (typeof value === "string") {
            const hex = value.startsWith("#") ? value.slice(1) : value;
            const parsed = Number.parseInt(hex, 16);
            if (Number.isFinite(parsed)) {
                return parsed;
            }
        }
        return fallback;
    }

    function createMeshFromData(meshData, colorValue) {
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(meshData.vertices.flat());
        const faces = new Uint32Array(meshData.faces.flat());

        geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(faces, 1));
        geometry.computeVertexNormals();

        const material = new THREE.MeshPhongMaterial({
            color: colorValue,
            side: THREE.DoubleSide,
            shininess: 16
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        return mesh;
    }

    function computeTerrainCenter(vertices) {
        if (!Array.isArray(vertices) || vertices.length === 0) {
            return { x: 0, y: 0, z: 0 };
        }

        let minX = Infinity;
        let minY = Infinity;
        let minZ = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        let maxZ = -Infinity;

        vertices.forEach((vertex) => {
            if (!vertex || vertex.length < 3) {
                return;
            }
            const [x, y, z] = vertex;
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
                return;
            }
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            minZ = Math.min(minZ, z);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
            maxZ = Math.max(maxZ, z);
        });

        if (![minX, minY, minZ, maxX, maxY, maxZ].every(Number.isFinite)) {
            return { x: 0, y: 0, z: 0 };
        }

        return {
            x: (minX + maxX) / 2,
            y: (minY + maxY) / 2,
            z: (minZ + maxZ) / 2
        };
    }

    function registerObject(record) {
        state.objectsByKey.set(record.key, record);
        state.objectOrder.push(record.key);
    }

    function featureTypeLabel(type) {
        const map = {
            terrain: "Terrain",
            building: "Building",
            road: "Road",
            water: "Water",
            gpx_track: "GPX",
            marker: "Marker",
            label: "Label"
        };
        return map[type] || type;
    }

    function resolveFeatureColor(feature) {
        if (feature.type === "building") {
            if (feature.custom_color) {
                return parseColorValue(feature.custom_color, 0xaaaaaa);
            }
            if (feature.is_address_building) {
                return 0xffc933;
            }
            return 0xaaaaaa;
        }
        if (feature.type === "road") {
            return 0x444444;
        }
        if (feature.type === "water") {
            return 0x4a90e2;
        }
        return 0x8a8a8a;
    }

    function deriveFeatureName(feature) {
        if (feature.is_address_building) {
            return "Address Building";
        }
        if (feature.name && String(feature.name).trim().length > 0) {
            return feature.name;
        }
        return featureTypeLabel(feature.type);
    }

    function renderMesh(meshData) {
        if (!meshData?.terrain?.vertices?.length) {
            showStatus("No terrain received from server", "error");
            return;
        }

        clearSceneObjects();
        state.currentMesh = meshData;
        state.lastRenderedMeshData = meshData;

        const terrainColor = 0x8b7355;
        const terrainObj = createMeshFromData(meshData.terrain, terrainColor);
        const terrainKey = makeFeatureKey("terrain", "terrain");
        terrainObj.userData = {
            key: terrainKey,
            type: "terrain",
            id: "terrain",
            originalColor: terrainColor
        };

        scene.add(terrainObj);
        registerObject({
            key: terrainKey,
            type: "terrain",
            id: "terrain",
            name: "Terrain",
            object: terrainObj,
            originalColor: terrainColor,
            sourceMesh: meshData.terrain,
            selectable: true
        });

        (meshData.features || []).forEach((feature) => {
            if (!feature?.vertices?.length || !feature?.faces?.length) {
                return;
            }

            const featureColor = resolveFeatureColor(feature);
            const featureObj = createMeshFromData(feature, featureColor);
            const key = feature.feature_key || makeFeatureKey(feature.type || "feature", feature.id);

            featureObj.userData = {
                key,
                type: feature.type,
                id: feature.id,
                originalColor: featureColor,
                isAddressBuilding: feature.is_address_building === true
            };

            if (feature.is_address_building) {
                featureObj.material.emissive = new THREE.Color(0x664900);
            }

            scene.add(featureObj);
            registerObject({
                key,
                type: feature.type,
                id: feature.id,
                name: deriveFeatureName(feature),
                object: featureObj,
                originalColor: featureColor,
                sourceMesh: feature,
                selectable: true
            });
        });

        if (meshData.gpx_track?.vertices?.length) {
            const trackColor = 0xff2020;
            const trackObj = createMeshFromData(meshData.gpx_track, trackColor);
            const key = makeFeatureKey("gpx_track", "gpx_track");
            trackObj.userData = {
                key,
                type: "gpx_track",
                id: "gpx_track",
                originalColor: trackColor
            };
            trackObj.material.emissive = new THREE.Color(0x661010);

            scene.add(trackObj);
            registerObject({
                key,
                type: "gpx_track",
                id: "gpx_track",
                name: "GPX Track",
                object: trackObj,
                originalColor: trackColor,
                sourceMesh: meshData.gpx_track,
                selectable: true
            });
        }

        state.sceneOrigin = computeTerrainCenter(meshData.terrain.vertices);
        centerCamera();
        applyIntentState();
        emitObjectsChange();
        forceRenderRefresh();
    }

    function forceRenderRefresh() {
        if (!renderer || !scene || !camera) {
            return;
        }
        scene.updateMatrixWorld(true);
        controls.update();
        renderer.render(scene, camera);
    }

    function centerCamera() {
        const origin = state.sceneOrigin || { x: 0, y: 0, z: 0 };
        const distance = 260;
        camera.position.set(origin.x + distance, origin.y + distance * 0.6, origin.z + distance);
        controls.target.set(origin.x, origin.y, origin.z);
        controls.update();
    }

    function setSelection(keys, { replace = true } = {}) {
        if (replace) {
            state.selectedKeys.clear();
        }

        keys.forEach((key) => {
            const record = state.objectsByKey.get(key);
            if (!record || record.deleted) {
                return;
            }
            state.selectedKeys.add(key);
        });

        refreshSelectionVisuals();
        emitSelectionChange();
    }

    function toggleSelection(key) {
        const record = state.objectsByKey.get(key);
        if (!record || record.deleted) {
            return;
        }

        if (state.selectedKeys.has(key)) {
            state.selectedKeys.delete(key);
        } else {
            state.selectedKeys.add(key);
        }

        refreshSelectionVisuals();
        emitSelectionChange();
    }

    function clearSelection() {
        state.selectedKeys.clear();
        refreshSelectionVisuals();
        emitSelectionChange();
    }

    function refreshSelectionVisuals() {
        state.objectsByKey.forEach((record) => {
            const mesh = record.object;
            if (!mesh?.material) {
                return;
            }

            const key = record.key;
            const selected = state.selectedKeys.has(key);
            if (selected) {
                mesh.material.emissive.setHex(0x6e6600);
                mesh.material.color.setHex(0xffdd33);
                return;
            }

            const colorOverride = state.editIntents.colorOverrides[key];
            if (colorOverride) {
                mesh.material.color.setHex(parseColorValue(colorOverride, record.originalColor));
            } else {
                mesh.material.color.setHex(record.originalColor);
            }

            if (record.type === "gpx_track") {
                mesh.material.emissive.setHex(0x661010);
            } else if (mesh.userData?.isAddressBuilding) {
                mesh.material.emissive.setHex(0x664900);
            } else {
                mesh.material.emissive.setHex(0x000000);
            }
        });
    }

    function setDeleted(keys, value) {
        keys.forEach((key) => {
            if (!state.objectsByKey.has(key)) {
                return;
            }
            if (value) {
                state.editIntents.deletedKeys.add(key);
            } else {
                state.editIntents.deletedKeys.delete(key);
            }
        });

        if (value) {
            keys.forEach((key) => state.selectedKeys.delete(key));
        }

        applyIntentState();
        emitSelectionChange();
        emitObjectsChange();
    }

    function setHidden(keys, value) {
        keys.forEach((key) => {
            if (!state.objectsByKey.has(key)) {
                return;
            }
            if (value) {
                state.editIntents.hiddenKeys.add(key);
            } else {
                state.editIntents.hiddenKeys.delete(key);
            }
        });

        applyIntentState();
        emitObjectsChange();
    }

    function setColor(key, colorValue) {
        if (!state.objectsByKey.has(key)) {
            return;
        }

        if (!colorValue) {
            delete state.editIntents.colorOverrides[key];
        } else {
            state.editIntents.colorOverrides[key] = colorValue;
        }

        applyIntentState();
        emitObjectsChange();
    }

    function getColor(key) {
        return state.editIntents.colorOverrides[key] || null;
    }

    function getHiddenSnapshot() {
        return new Set(state.editIntents.hiddenKeys);
    }

    function replaceHidden(snapshot) {
        state.editIntents.hiddenKeys = new Set(snapshot);
        applyIntentState();
        emitObjectsChange();
    }

    function restoreAllHidden() {
        state.editIntents.hiddenKeys.clear();
        applyIntentState();
        emitObjectsChange();
    }

    function applyIntentState() {
        state.objectsByKey.forEach((record, key) => {
            const deleted = state.editIntents.deletedKeys.has(key);
            const hidden = state.editIntents.hiddenKeys.has(key);
            record.deleted = deleted;
            record.hidden = hidden;
            record.object.visible = !deleted && !hidden;

            const colorOverride = state.editIntents.colorOverrides[key];
            if (colorOverride && record.object?.material) {
                record.object.material.color.setHex(parseColorValue(colorOverride, record.originalColor));
            }
        });

        refreshSelectionVisuals();
    }

    function applyIntentsAndReport() {
        applyIntentState();

        let unmatchedDeleted = 0;
        let unmatchedHidden = 0;
        let unmatchedColors = 0;

        state.editIntents.deletedKeys.forEach((key) => {
            if (!state.objectsByKey.has(key)) {
                unmatchedDeleted += 1;
            }
        });
        state.editIntents.hiddenKeys.forEach((key) => {
            if (!state.objectsByKey.has(key)) {
                unmatchedHidden += 1;
            }
        });
        Object.keys(state.editIntents.colorOverrides).forEach((key) => {
            if (!state.objectsByKey.has(key)) {
                unmatchedColors += 1;
            }
        });

        return { unmatchedDeleted, unmatchedHidden, unmatchedColors };
    }

    function getObjectRecords() {
        return state.objectOrder
            .map((key) => state.objectsByKey.get(key))
            .filter(Boolean);
    }

    function getSelectedKeys() {
        return Array.from(state.selectedKeys);
    }

    function addLabel(text) {
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");
        canvas.width = 640;
        canvas.height = 160;

        context.fillStyle = "rgba(0, 0, 0, 0.72)";
        context.fillRect(0, 0, canvas.width, canvas.height);

        context.font = "48px 'IBM Plex Sans'";
        context.fillStyle = "#ffffff";
        context.textAlign = "center";
        context.fillText(text, canvas.width / 2, canvas.height / 2 + 16);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(64, 16, 1);

        const origin = state.sceneOrigin || { x: 0, y: 0, z: 0 };
        sprite.position.set(origin.x, origin.y + 45, origin.z);

        const labelId = `label_${Date.now()}`;
        const key = makeFeatureKey("label", labelId);
        sprite.userData = {
            key,
            type: "label",
            id: labelId,
            originalColor: 0x6677aa
        };

        scene.add(sprite);
        registerObject({
            key,
            type: "label",
            id: labelId,
            name: text,
            object: sprite,
            originalColor: 0x6677aa,
            sourceMesh: null,
            selectable: false
        });

        applyIntentState();
        emitObjectsChange();
        showStatus("Label added", "success");
    }

    function createAddressMarker(lat, lon) {
        let existingKey = null;
        state.objectsByKey.forEach((record) => {
            if (record.type === "marker" && record.id === "address_marker") {
                existingKey = record.key;
            }
        });

        if (existingKey) {
            const existing = state.objectsByKey.get(existingKey);
            if (existing) {
                scene.remove(existing.object);
                state.objectsByKey.delete(existingKey);
                state.objectOrder = state.objectOrder.filter((key) => key !== existingKey);
                state.selectedKeys.delete(existingKey);
            }
        }

        const geometry = new THREE.ConeGeometry(3, 8, 8);
        const material = new THREE.MeshPhongMaterial({ color: 0xff0000, emissive: 0x660000 });
        const marker = new THREE.Mesh(geometry, material);
        marker.rotation.x = Math.PI;

        if (state.elevationData?.bounds && state.currentMesh) {
            const bounds = state.elevationData.bounds;
            const modelWidth = Number.parseFloat(document.getElementById("model-width")?.value || "200");
            const latRange = bounds.north - bounds.south;
            const lonRange = bounds.east - bounds.west;
            const avgLat = (bounds.north + bounds.south) / 2;
            const lonScale = Math.cos(avgLat * Math.PI / 180);
            const scaleFactor = modelWidth / Math.max(latRange, lonRange * lonScale);

            const x = (lon - bounds.west) * scaleFactor * lonScale;
            const z = (bounds.north - lat) * scaleFactor;
            marker.position.set(x, 32, z);
        } else {
            const origin = state.sceneOrigin || { x: 0, y: 0, z: 0 };
            marker.position.set(origin.x, origin.y + 22, origin.z);
        }

        const key = makeFeatureKey("marker", "address_marker");
        marker.userData = {
            key,
            type: "marker",
            id: "address_marker",
            originalColor: 0xff0000
        };

        scene.add(marker);
        registerObject({
            key,
            type: "marker",
            id: "address_marker",
            name: "Address Marker",
            object: marker,
            originalColor: 0xff0000,
            sourceMesh: null,
            selectable: true
        });

        applyIntentState();
        emitObjectsChange();
    }

    function onPointerDown(event) {
        if (!renderer || !renderer.domElement) {
            return;
        }
        const rect = renderer.domElement.getBoundingClientRect();
        if (!isInsideRect(event.clientX, event.clientY, rect)) {
            return;
        }

        if (!state.boxSelectEnabled) {
            boxSelectState.active = false;
            boxSelectState.startX = event.clientX - rect.left;
            boxSelectState.startY = event.clientY - rect.top;
            return;
        }

        controls.enabled = false;
        event.preventDefault();
        event.stopPropagation();

        boxSelectState.active = true;
        boxSelectState.startX = event.clientX - rect.left;
        boxSelectState.startY = event.clientY - rect.top;

        const viewerRect = document.querySelector(".viewer-container").getBoundingClientRect();
        selectionBox.style.left = `${event.clientX - viewerRect.left}px`;
        selectionBox.style.top = `${event.clientY - viewerRect.top}px`;
        selectionBox.style.width = "0px";
        selectionBox.style.height = "0px";
        selectionBox.style.display = "block";
    }

    function onPointerMove(event) {
        if (!boxSelectState.active || !state.boxSelectEnabled) {
            return;
        }
        event.preventDefault();
        event.stopPropagation();

        const rect = renderer.domElement.getBoundingClientRect();
        const viewerRect = document.querySelector(".viewer-container").getBoundingClientRect();

        const currentX = event.clientX - rect.left;
        const currentY = event.clientY - rect.top;

        const startViewerX = boxSelectState.startX + (rect.left - viewerRect.left);
        const startViewerY = boxSelectState.startY + (rect.top - viewerRect.top);
        const currentViewerX = currentX + (rect.left - viewerRect.left);
        const currentViewerY = currentY + (rect.top - viewerRect.top);

        const left = Math.min(startViewerX, currentViewerX);
        const top = Math.min(startViewerY, currentViewerY);
        const width = Math.abs(currentViewerX - startViewerX);
        const height = Math.abs(currentViewerY - startViewerY);

        selectionBox.style.left = `${left}px`;
        selectionBox.style.top = `${top}px`;
        selectionBox.style.width = `${width}px`;
        selectionBox.style.height = `${height}px`;
    }

    function onPointerUp(event) {
        if (!renderer || !renderer.domElement) {
            return;
        }

        const rect = renderer.domElement.getBoundingClientRect();
        const endX = event.clientX - rect.left;
        const endY = event.clientY - rect.top;

        const dragWidth = Math.abs(endX - boxSelectState.startX);
        const dragHeight = Math.abs(endY - boxSelectState.startY);

        if (boxSelectState.active && state.boxSelectEnabled) {
            event.preventDefault();
            event.stopPropagation();
            controls.enabled = true;
            selectionBox.style.display = "none";
            boxSelectState.active = false;

            const minX = Math.min(boxSelectState.startX, endX);
            const maxX = Math.max(boxSelectState.startX, endX);
            const minY = Math.min(boxSelectState.startY, endY);
            const maxY = Math.max(boxSelectState.startY, endY);

            if (dragWidth < 5 && dragHeight < 5) {
                selectAtPoint(boxSelectState.startX, boxSelectState.startY, rect, false);
            } else {
                selectInBox(minX, minY, maxX, maxY, rect);
            }
            return;
        }

        if (dragWidth < 4 && dragHeight < 4) {
            selectAtPoint(endX, endY, rect, event.shiftKey);
        }
    }

    function selectAtPoint(x, y, rect, additive) {
        mouse.x = (x / rect.width) * 2 - 1;
        mouse.y = -(y / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);

        const meshes = getObjectRecords()
            .filter((record) => record.selectable && !record.deleted)
            .map((record) => record.object)
            .filter((obj) => obj && obj.isMesh);

        const intersects = raycaster.intersectObjects(meshes);
        if (intersects.length === 0) {
            if (!additive) {
                clearSelection();
            }
            return;
        }

        const clicked = intersects[0].object;
        const key = clicked.userData?.key;
        if (!key) {
            return;
        }

        if (additive) {
            toggleSelection(key);
            return;
        }

        setSelection([key], { replace: true });
    }

    function selectInBox(minX, minY, maxX, maxY, rect) {
        const selectedKeys = [];

        getObjectRecords().forEach((record) => {
            if (!record.selectable || record.deleted || !record.object?.geometry) {
                return;
            }

            record.object.geometry.computeBoundingBox();
            const center = new THREE.Vector3();
            record.object.geometry.boundingBox.getCenter(center);
            center.applyMatrix4(record.object.matrixWorld);

            const screenPos = center.clone().project(camera);
            const screenX = ((screenPos.x + 1) / 2) * rect.width;
            const screenY = ((-screenPos.y + 1) / 2) * rect.height;

            if (screenX >= minX && screenX <= maxX && screenY >= minY && screenY <= maxY && screenPos.z < 1) {
                selectedKeys.push(record.key);
            }
        });

        if (selectedKeys.length > 0) {
            setSelection(selectedKeys, { replace: false });
            showStatus(`Selected ${selectedKeys.length} objects`, "success");
        }
    }

    function isInsideRect(x, y, rect) {
        return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
    }

    function updateDiagnostics(note = "") {
        const debugEl = document.getElementById("render-debug");
        if (!debugEl || !camera || !scene) {
            return;
        }

        const meshData = state.lastRenderedMeshData;
        const terrainVerts = meshData?.terrain?.vertices?.length || 0;
        const terrainFaces = meshData?.terrain?.faces?.length || 0;
        const featureCount = meshData?.features?.length || 0;
        const gpxVerts = meshData?.gpx_track?.vertices?.length || 0;
        const origin = state.sceneOrigin || { x: 0, y: 0, z: 0 };
        const visibleObjects = getObjectRecords().filter((record) => !record.deleted && !record.hidden).length;

        const lines = [
            `meshes=${scene.children.filter((child) => child.isMesh).length} objects=${state.objectsByKey.size} visible=${visibleObjects}`,
            `terrain(v/f)=${terrainVerts}/${terrainFaces} features=${featureCount} gpxVerts=${gpxVerts}`,
            `cam=(${camera.position.x.toFixed(1)}, ${camera.position.y.toFixed(1)}, ${camera.position.z.toFixed(1)}) target=(${controls.target.x.toFixed(1)}, ${controls.target.y.toFixed(1)}, ${controls.target.z.toFixed(1)})`,
            `origin=(${origin.x.toFixed(1)}, ${origin.y.toFixed(1)}, ${origin.z.toFixed(1)})`
        ];
        if (note) {
            lines.push(`note=${note}`);
        }

        debugEl.textContent = lines.join("\n");
    }

    function emitSelectionChange() {
        if (typeof onSelectionChanged === "function") {
            onSelectionChanged(getSelectedKeys());
        }
    }

    function emitObjectsChange() {
        if (typeof onObjectsChanged === "function") {
            onObjectsChanged(getObjectRecords());
        }
    }

    function buildExportMesh() {
        const exportData = {
            terrain: null,
            features: [],
            gpx_track: null
        };

        getObjectRecords().forEach((record) => {
            if (record.deleted) {
                return;
            }

            if (record.type === "label" || record.type === "marker") {
                return;
            }

            const geometry = record.object?.geometry;
            if (!geometry) {
                return;
            }

            const position = geometry.attributes.position;
            const index = geometry.index;
            const vertices = [];
            for (let i = 0; i < position.count; i += 1) {
                vertices.push([
                    position.getX(i),
                    position.getY(i),
                    position.getZ(i)
                ]);
            }

            const faces = [];
            if (index) {
                for (let i = 0; i < index.count; i += 3) {
                    faces.push([
                        index.getX(i),
                        index.getX(i + 1),
                        index.getX(i + 2)
                    ]);
                }
            }

            const meshData = {
                vertices,
                faces,
                type: record.type,
                id: record.id,
                name: record.name,
                feature_key: record.key,
                is_address_building: record.object?.userData?.isAddressBuilding || false
            };

            if (record.type === "terrain") {
                exportData.terrain = meshData;
            } else if (record.type === "gpx_track") {
                exportData.gpx_track = meshData;
            } else {
                exportData.features.push(meshData);
            }
        });

        return exportData;
    }

    return {
        init,
        renderMesh,
        centerCamera,
        refreshViewport: requestViewportSync,
        setSelection,
        toggleSelection,
        clearSelection,
        setDeleted,
        setHidden,
        setColor,
        getColor,
        getHiddenSnapshot,
        replaceHidden,
        restoreAllHidden,
        applyIntentState,
        applyIntentsAndReport,
        getObjectRecords,
        getSelectedKeys,
        addLabel,
        createAddressMarker,
        buildExportMesh,
        forceRenderRefresh,
        setBoxSelectEnabled(enabled) {
            state.boxSelectEnabled = Boolean(enabled);
            if (!enabled) {
                boxSelectState.active = false;
                if (selectionBox) {
                    selectionBox.style.display = "none";
                }
                controls.enabled = true;
            }
        }
    };
}
