// Face Attendance System — frontend logic.
// Talks to the team's FastAPI backend over HTTP.

const cfg = window.APP_CONFIG;
const API = cfg.API_BASE;

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const nameInput = $('nameInput');
const registerBtn = $('registerBtn');
const userList = $('userList');
const userCount = $('userCount');
const toast = $('toast');

const infoName = $('infoName');
const infoEmotion = $('infoEmotion');
const infoLiveness = $('infoLiveness');
const infoTime = $('infoTime');

const dotCam = $('dotCam');
const dotFace = $('dotFace');
const dotBackend = $('dotBackend');
const clockEl = $('clock');
const apiBaseEl = $('apiBase');

const videoEl = $('videoStream');
const camFallback = $('camFallback');

// Lasith's Grad-CAM spoof explainability DOM elements
const explainSpoofBtn = $('explainSpoofBtn');
const spoofPrediction = $('spoofPrediction');
const spoofScore = $('spoofScore');
const spoofConfidence = $('spoofConfidence');
const spoofHeatmapImage = $('spoofHeatmapImage');

apiBaseEl.textContent = API.replace(/^https?:\/\//, '');

// ---- Clock ----
function tickClock() {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    clockEl.textContent = `${hh}:${mm}:${ss}`;
}
setInterval(tickClock, 1000);
tickClock();

// ---- Toast ----
let toastTimer = null;

function showToast(text, kind = 'info') {
    toast.textContent = text;
    toast.className = 'toast show ' + kind;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.className = 'toast';
    }, 2500);
}

// ---- API ----
async function fetchJson(path, opts = {}) {
    const res = await fetch(API + path, opts);

    if (!res.ok) {
        let detail = `HTTP ${res.status}`;

        try {
            const data = await res.json();
            detail = data.detail || JSON.stringify(data);
        } catch {}

        throw new Error(detail);
    }

    return res.json();
}

async function loadUsers() {
    try {
        const names = await fetchJson('/identities');
        renderUsers(Array.isArray(names) ? names : []);
        dotBackend.className = 'dot dot-green';
    } catch {
        dotBackend.className = 'dot dot-red';
        renderUsers([]);
    }
}

async function registerName() {
    const name = nameInput.value.trim();

    if (!name) {
        showToast('Please enter a name first', 'warn');
        nameInput.focus();
        return;
    }

    registerBtn.disabled = true;
    registerBtn.textContent = 'Registering…';

    try {
        const result = await fetchJson('/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });

        showToast(`Registered: ${result.registered}`, 'ok');
        nameInput.value = '';
        await loadUsers();

    } catch (err) {
        showToast(err.message || 'Registration failed', 'error');

    } finally {
        registerBtn.disabled = false;
        registerBtn.textContent = 'Register current face';
    }
}

async function deleteName(name) {
    if (!confirm(`Delete user "${name}"?`)) return;

    try {
        await fetchJson(`/identities/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        });

        showToast(`Deleted: ${name}`, 'info');
        await loadUsers();

    } catch (err) {
        showToast(err.message || 'Delete failed', 'error');
    }
}

// ---- Rendering ----
function renderUsers(names) {
    userCount.textContent = names.length;

    if (!names.length) {
        userList.innerHTML = '<li class="empty">No users registered yet</li>';
        return;
    }

    userList.innerHTML = '';

    for (const name of names) {
        const li = document.createElement('li');

        const span = document.createElement('span');
        span.textContent = name;

        const btn = document.createElement('button');
        btn.className = 'btn btn-danger';
        btn.textContent = 'Delete';
        btn.onclick = () => deleteName(name);

        li.appendChild(span);
        li.appendChild(btn);
        userList.appendChild(li);
    }
}

// ---- Video stream ----
function startVideo() {
    // Bust cache so the MJPEG stream reconnects cleanly on reload.
    videoEl.src = `${API}/video?t=${Date.now()}`;
}

videoEl.addEventListener('load', () => {
    dotCam.className = 'dot dot-green';
    camFallback.classList.remove('show');
    videoEl.classList.remove('hidden');
});

videoEl.addEventListener('error', () => {
    dotCam.className = 'dot dot-red';
    camFallback.classList.add('show');
    videoEl.classList.add('hidden');
});

// ---- Status polling ----
// Tracks the current primary identity so 'Detected at' updates once when the
// identity changes, instead of refreshing every poll (which is meaningless).
let currentIdentity = null;

async function pollStatus() {
    try {
        const s = await fetchJson('/status');

        dotFace.className = 'dot ' + (s.face_detected ? 'dot-green' : 'dot-grey');

        if (s.face_detected) {
            const name = s.name || 'Unknown';
            infoName.textContent = name;
            infoEmotion.textContent = s.emotion || '—';
            infoLiveness.textContent = s.liveness || '—';

            // Only stamp the time when the identity actually changes
            if (name !== currentIdentity) {
                currentIdentity = name;
                infoTime.textContent = new Date().toLocaleTimeString();
            }
        } else {
            infoName.textContent = '—';
            infoEmotion.textContent = '—';
            infoLiveness.textContent = '—';
            infoTime.textContent = '—';
            currentIdentity = null;
        }

    } catch {
        // /status unavailable — leave panel as-is until next poll
    }
}

// ------------------------------------------------------------
// Lasith's Innovative Feature:
// Grad-CAM Spoof Explainability Integration
// ------------------------------------------------------------
async function generateSpoofHeatmap() {
    if (!explainSpoofBtn) return;

    explainSpoofBtn.disabled = true;
    explainSpoofBtn.textContent = 'Generating…';

    try {
        const data = await fetchJson('/spoof-explainability');

        spoofPrediction.textContent = data.prediction || '—';
        spoofScore.textContent = Number(data.spoof_score).toFixed(4);
        spoofConfidence.textContent = Number(data.confidence).toFixed(4);

        spoofHeatmapImage.src = `${API}${data.heatmap_url}?t=${Date.now()}`;
        spoofHeatmapImage.style.display = 'block';

        showToast('Spoof Grad-CAM heatmap generated', 'ok');

    } catch (err) {
        showToast(err.message || 'Failed to generate spoof heatmap', 'error');

    } finally {
        explainSpoofBtn.disabled = false;
        explainSpoofBtn.textContent = 'Generate spoof heatmap';
    }
}

// ---- Wiring ----
registerBtn.addEventListener('click', registerName);

nameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') registerName();
});

if (explainSpoofBtn) {
    explainSpoofBtn.addEventListener('click', generateSpoofHeatmap);
}

// ---- Boot ----
startVideo();
loadUsers();
pollStatus();

setInterval(loadUsers, cfg.POLL_USERS_MS);
setInterval(pollStatus, cfg.POLL_STATUS_MS);
// Face Attendance System — frontend logic.
// Talks to the team's FastAPI backend over HTTP.

const cfg = window.APP_CONFIG;
const API = cfg.API_BASE;

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const nameInput = $('nameInput');
const registerBtn = $('registerBtn');
const userList = $('userList');
const userCount = $('userCount');
const toast = $('toast');

const infoName = $('infoName');
const infoEmotion = $('infoEmotion');
const infoLiveness = $('infoLiveness');
const infoTime = $('infoTime');

const dotCam = $('dotCam');
const dotFace = $('dotFace');
const dotBackend = $('dotBackend');
const fpsValue = $('fpsValue');
const clockEl = $('clock');
const apiBaseEl = $('apiBase');

const videoEl = $('videoStream');
const camFallback = $('camFallback');

// Lasith's Grad-CAM spoof explainability DOM elements
const explainSpoofBtn = $('explainSpoofBtn');
const spoofPrediction = $('spoofPrediction');
const spoofScore = $('spoofScore');
const spoofConfidence = $('spoofConfidence');
const spoofHeatmapImage = $('spoofHeatmapImage');

apiBaseEl.textContent = API.replace(/^https?:\/\//, '');

// ---- Clock ----
function tickClock() {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    clockEl.textContent = `${hh}:${mm}:${ss}`;
}
setInterval(tickClock, 1000);
tickClock();

// ---- Toast ----
let toastTimer = null;

function showToast(text, kind = 'info') {
    toast.textContent = text;
    toast.className = 'toast show ' + kind;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.className = 'toast';
    }, 2500);
}

// ---- API ----
async function fetchJson(path, opts = {}) {
    const res = await fetch(API + path, opts);

    if (!res.ok) {
        let detail = `HTTP ${res.status}`;

        try {
            const data = await res.json();
            detail = data.detail || JSON.stringify(data);
        } catch {}

        throw new Error(detail);
    }

    return res.json();
}

async function loadUsers() {
    try {
        const names = await fetchJson('/identities');
        renderUsers(Array.isArray(names) ? names : []);
        dotBackend.className = 'dot dot-green';
    } catch {
        dotBackend.className = 'dot dot-red';
        renderUsers([]);
    }
}

async function registerName() {
    const name = nameInput.value.trim();

    if (!name) {
        showToast('Please enter a name first', 'warn');
        nameInput.focus();
        return;
    }

    registerBtn.disabled = true;
    registerBtn.textContent = 'Registering…';

    try {
        const result = await fetchJson('/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });

        showToast(`Registered: ${result.registered}`, 'ok');
        nameInput.value = '';
        await loadUsers();

    } catch (err) {
        showToast(err.message || 'Registration failed', 'error');

    } finally {
        registerBtn.disabled = false;
        registerBtn.textContent = 'Register current face';
    }
}

async function deleteName(name) {
    if (!confirm(`Delete user "${name}"?`)) return;

    try {
        await fetchJson(`/identities/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        });

        showToast(`Deleted: ${name}`, 'info');
        await loadUsers();

    } catch (err) {
        showToast(err.message || 'Delete failed', 'error');
    }
}

// ---- Rendering ----
function renderUsers(names) {
    userCount.textContent = names.length;

    if (!names.length) {
        userList.innerHTML = '<li class="empty">No users registered yet</li>';
        return;
    }

    userList.innerHTML = '';

    for (const name of names) {
        const li = document.createElement('li');

        const span = document.createElement('span');
        span.textContent = name;

        const btn = document.createElement('button');
        btn.className = 'btn btn-danger';
        btn.textContent = 'Delete';
        btn.onclick = () => deleteName(name);

        li.appendChild(span);
        li.appendChild(btn);
        userList.appendChild(li);
    }
}

// ---- Video stream ----
function startVideo() {
    // Bust cache so the MJPEG stream reconnects cleanly on reload.
    videoEl.src = `${API}/video?t=${Date.now()}`;
}

videoEl.addEventListener('load', () => {
    dotCam.className = 'dot dot-green';
    camFallback.classList.remove('show');
    videoEl.classList.remove('hidden');
});

videoEl.addEventListener('error', () => {
    dotCam.className = 'dot dot-red';
    camFallback.classList.add('show');
    videoEl.classList.add('hidden');
});

// ---- Status polling ----
async function pollStatus() {
    try {
        const s = await fetchJson('/status');

        dotFace.className = 'dot ' + (s.face_detected ? 'dot-green' : 'dot-grey');

        if (s.face_detected) {
            infoName.textContent = s.name || 'Unknown';
            infoEmotion.textContent = s.emotion || '—';
            infoLiveness.textContent = s.liveness || '—';
            infoTime.textContent = new Date().toLocaleTimeString();
        } else {
            infoName.textContent = '—';
            infoEmotion.textContent = '—';
            infoLiveness.textContent = '—';
        }

        if (typeof s.fps === 'number') {
            fpsValue.textContent = s.fps.toFixed(1);
        }

    } catch {
        fpsValue.textContent = '—';
    }
}

// ------------------------------------------------------------
// Lasith's Innovative Feature:
// Grad-CAM Spoof Explainability Integration
// ------------------------------------------------------------
async function generateSpoofHeatmap() {
    if (!explainSpoofBtn) return;

    explainSpoofBtn.disabled = true;
    explainSpoofBtn.textContent = 'Generating…';

    try {
        const data = await fetchJson('/spoof-explainability');

        spoofPrediction.textContent = data.prediction || '—';
        spoofScore.textContent = Number(data.spoof_score).toFixed(4);
        spoofConfidence.textContent = Number(data.confidence).toFixed(4);

        spoofHeatmapImage.src = `${API}${data.heatmap_url}?t=${Date.now()}`;
        spoofHeatmapImage.style.display = 'block';

        showToast('Spoof Grad-CAM heatmap generated', 'ok');

    } catch (err) {
        showToast(err.message || 'Failed to generate spoof heatmap', 'error');

    } finally {
        explainSpoofBtn.disabled = false;
        explainSpoofBtn.textContent = 'Generate spoof heatmap';
    }
}

// ---- Wiring ----
registerBtn.addEventListener('click', registerName);

nameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') registerName();
});

if (explainSpoofBtn) {
    explainSpoofBtn.addEventListener('click', generateSpoofHeatmap);
}

// ---- Boot ----
startVideo();
loadUsers();
pollStatus();

setInterval(loadUsers, cfg.POLL_USERS_MS);
setInterval(pollStatus, cfg.POLL_STATUS_MS);