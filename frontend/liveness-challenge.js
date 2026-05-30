// ============================================================
// Innovative Feature: Active Liveness Challenge
// ------------------------------------------------------------
// Adds an interactive anti-replay check on top of the passive
// CNN liveness model. The system asks the user to perform a
// random action (move / turn head / nod) and confirms real
// motion before granting check-in.
//
// A printed photo or a still image on a screen cannot respond
// to a random prompt, so this defends against simple replay
// attacks that a passive model might accept.
//
// Runs ALONGSIDE the backend (no camera conflict): it fetches
// the backend's MJPEG stream, parses out each JPEG frame, and
// decodes them itself. This guarantees fresh frames (an <img>
// freezes on the first frame for canvas reads) and avoids the
// cross-origin pixel-read problem.
//
// Self-contained: this whole feature lives in this one file.
// To remove it, delete this file and the matching <script> line
// in index.html.
// ============================================================

(function () {
    const API = window.APP_CONFIG.API_BASE;

    const PROMPTS = [
        'Turn your head slowly',
        'Nod your head',
        'Move a little',
        'Lean left, then right',
    ];

    const CALIB_MS = 1800;          // "hold still" baseline window
    const CHALLENGE_SECONDS = 6;    // time allowed to perform the action
    const SAMPLE_MS = 150;          // how often we analyse a frame
    const PIXEL_DELTA = 30;         // per-pixel change that counts as "moved" (ignores mild noise)
    const SPIKE_MULTIPLIER = 1.8;   // motion must clearly beat the PEAK still noise
    const SPIKE_MARGIN = 40;        // ...and at least this many changed pixels above it
    const MIN_THRESHOLD = 60;       // absolute floor for the spike threshold
    const REQUIRED_CONSEC = 3;      // need this many CONSECUTIVE frames of motion (~0.45s)

    // ---- Inject this feature's own styles (keeps it self-contained) ----
    const style = document.createElement('style');
    style.textContent = `
        #livenessOverlay {
            position: fixed; inset: 0;
            background: rgba(0, 0, 0, 0.82);
            display: none; align-items: center; justify-content: center;
            z-index: 2000;
        }
        #livenessOverlay.show { display: flex; }
        #livenessOverlay .lc-box {
            background: #1a1e26; border: 1px solid #252a34;
            border-radius: 16px; padding: 28px 36px; text-align: center;
            min-width: 360px;
        }
        #livenessOverlay canvas {
            width: 320px; height: 240px; border-radius: 12px; background: #000;
        }
        #livenessOverlay .lc-prompt {
            font-size: 22px; font-weight: 700; color: #e6e8ed; margin: 16px 0 8px;
        }
        #livenessOverlay .lc-timer {
            font-size: 48px; font-weight: 800; color: #4cc9f0; line-height: 1;
        }
        #livenessOverlay .lc-status {
            margin-top: 12px; font-size: 14px; color: #aab1bd;
        }
        #livenessOverlay .lc-meter {
            --lc-fill: 0%;
            margin-top: 12px; height: 22px; border-radius: 6px;
            background: linear-gradient(#22c55e, #22c55e) no-repeat, #0f1115;
            background-size: var(--lc-fill) 100%;
            border: 1px solid #252a34;
            font-size: 11px; color: #e6e8ed; line-height: 22px;
            font-family: 'Consolas', monospace;
            transition: background-size 0.1s linear;
        }
        #livenessOverlay .lc-box.lc-pass { border-color: #2a9d8f; }
        #livenessOverlay .lc-box.lc-pass .lc-timer { color: #2a9d8f; }
        #livenessOverlay .lc-box.lc-fail { border-color: #ef4444; }
        #livenessOverlay .lc-box.lc-fail .lc-timer { color: #ef4444; }
    `;
    document.head.appendChild(style);

    // ---- Side panel ----
    function buildPanel() {
        const sidePanels = document.querySelector('.side-panels');
        if (!sidePanels) return;
        const panel = document.createElement('div');
        panel.className = 'panel';
        panel.innerHTML = `
            <div class="panel-header">
                <h2>Liveness Check-In</h2>
                <span class="panel-pill">Active</span>
            </div>
            <button id="livenessBtn" class="btn btn-primary">Start check-in</button>
            <p class="hint">
                Asks you to perform a random action and confirms real movement
                before check-in. Defends against photo / screen replay attacks.
            </p>
        `;
        sidePanels.appendChild(panel);
        document.getElementById('livenessBtn')
            .addEventListener('click', startChallenge);
    }

    // ---- Overlay ----
    function buildOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'livenessOverlay';
        overlay.innerHTML = `
            <div class="lc-box">
                <canvas id="lcCanvas" width="240" height="180"></canvas>
                <div class="lc-prompt" id="lcPrompt">Get ready…</div>
                <div class="lc-timer" id="lcTimer">…</div>
                <div class="lc-status" id="lcStatus">Connecting to video…</div>
                <div class="lc-meter" id="lcMeter"></div>
            </div>
        `;
        document.body.appendChild(overlay);
        return overlay;
    }

    function toast(msg, kind) {
        if (window.showToast) window.showToast(msg, kind);
    }

    // ---- MJPEG byte-stream reader: pulls out individual JPEG frames ----
    // Each JPEG starts with FF D8 and ends with FF D9. We scan the buffered
    // bytes for those markers and emit complete frames.
    function makeFrameReader(onFrame, onError) {
        const controller = new AbortController();
        let buffer = new Uint8Array(0);

        fetch(`${API}/video?lc=${Date.now()}`, { signal: controller.signal })
            .then((res) => {
                const reader = res.body.getReader();
                function pump() {
                    return reader.read().then(({ done, value }) => {
                        if (done) return;
                        // append chunk
                        const merged = new Uint8Array(buffer.length + value.length);
                        merged.set(buffer);
                        merged.set(value, buffer.length);
                        buffer = merged;

                        // extract complete JPEGs
                        let start = -1, lastEnd = -1;
                        for (let i = 0; i < buffer.length - 1; i++) {
                            if (buffer[i] === 0xFF && buffer[i + 1] === 0xD8) {
                                start = i;
                            } else if (buffer[i] === 0xFF && buffer[i + 1] === 0xD9 && start >= 0) {
                                onFrame(buffer.slice(start, i + 2));
                                lastEnd = i + 2;
                                start = -1;
                            }
                        }
                        if (lastEnd >= 0) buffer = buffer.slice(lastEnd);
                        // safety: don't let buffer grow unbounded
                        if (buffer.length > 5_000_000) buffer = new Uint8Array(0);

                        return pump();
                    });
                }
                return pump();
            })
            .catch((err) => { if (err.name !== 'AbortError') onError(err); });

        return controller;
    }

    // ---- The challenge ----
    async function startChallenge() {
        const btn = document.getElementById('livenessBtn');
        btn.disabled = true;

        const overlay = buildOverlay();
        overlay.classList.add('show');

        const promptEl = document.getElementById('lcPrompt');
        const timerEl = document.getElementById('lcTimer');
        const statusEl = document.getElementById('lcStatus');
        const meterEl = document.getElementById('lcMeter');
        const viewCanvas = document.getElementById('lcCanvas');
        const viewCtx = viewCanvas.getContext('2d');

        const prompt = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];

        // Analysis canvas (small = fast)
        const aCanvas = document.createElement('canvas');
        aCanvas.width = 160; aCanvas.height = 120;
        const aCtx = aCanvas.getContext('2d', { willReadFrequently: true });
        const W = aCanvas.width, H = aCanvas.height;
        const RX0 = Math.floor(W * 0.20), RX1 = Math.floor(W * 0.80);
        const RY0 = Math.floor(H * 0.10), RY1 = Math.floor(H * 0.90);

        let latestJpeg = null;
        let decoding = false;
        let gotFrame = false;
        let prevFrame = null;
        let phase = 'connect';
        let baselineDiffs = [];
        let dynamicThreshold = Infinity;
        let consec = 0;          // consecutive frames currently over the line
        let passedFlag = false;  // sustained motion achieved

        const controller = makeFrameReader(
            (jpeg) => { latestJpeg = jpeg; gotFrame = true; },
            () => { statusEl.textContent = 'Could not read the video stream. Is the backend running?'; }
        );

        async function analyse() {
            if (!latestJpeg || decoding) return;
            decoding = true;
            try {
                const blob = new Blob([latestJpeg], { type: 'image/jpeg' });
                const bmp = await createImageBitmap(blob);
                // show preview
                viewCtx.drawImage(bmp, 0, 0, viewCanvas.width, viewCanvas.height);
                // analyse small copy
                aCtx.drawImage(bmp, 0, 0, W, H);
                bmp.close && bmp.close();
                const frame = aCtx.getImageData(0, 0, W, H).data;

                let changed = null;
                if (prevFrame) {
                    changed = 0;
                    for (let y = RY0; y < RY1; y += 2) {
                        for (let x = RX0; x < RX1; x += 2) {
                            const i = (y * W + x) * 4;
                            if (Math.abs(frame[i] - prevFrame[i]) > PIXEL_DELTA) changed++;
                        }
                    }
                }
                prevFrame = frame.slice(0);

                if (changed !== null) handleMotion(changed);
            } catch (e) {
                // decode error — skip this frame
            } finally {
                decoding = false;
            }
        }

        function handleMotion(changed) {
            if (phase === 'calibrate') {
                baselineDiffs.push(changed);
            } else if (phase === 'challenge') {
                const pct = Math.min(100, (changed / (dynamicThreshold * 2)) * 100);
                meterEl.style.setProperty('--lc-fill', pct + '%');
                meterEl.textContent = `moved px: ${changed} (need > ${dynamicThreshold.toFixed(0)})`;

                if (changed > dynamicThreshold) {
                    consec += 1;                       // sustained motion building up
                    statusEl.textContent = `Movement detected (${consec}/${REQUIRED_CONSEC})`;
                    if (consec >= REQUIRED_CONSEC) passedFlag = true;
                } else {
                    consec = 0;                        // broke the streak — must be continuous
                }
            }
        }

        const sampler = setInterval(analyse, SAMPLE_MS);

        // Wait until frames are flowing, then calibrate
        statusEl.textContent = 'Connecting to video…';
        timerEl.textContent = '…';

        const waitStart = Date.now();
        const waiter = setInterval(() => {
            if (gotFrame) {
                clearInterval(waiter);
                beginCalibration();
            } else if (Date.now() - waitStart > 8000) {
                clearInterval(waiter);
                statusEl.textContent = 'No video received. Is the backend running?';
                overlay.querySelector('.lc-box').classList.add('lc-fail');
                setTimeout(cleanup, 2600);
            }
        }, 100);

        function beginCalibration() {
            phase = 'calibrate';
            statusEl.textContent = 'Hold still…';
            setTimeout(() => {
                // Use the PEAK still-noise (not the average) so normal jitter
                // while holding still never crosses the line.
                const baselinePeak = baselineDiffs.length
                    ? Math.max(...baselineDiffs)
                    : 0;
                dynamicThreshold = Math.max(
                    baselinePeak * SPIKE_MULTIPLIER, baselinePeak + SPIKE_MARGIN, MIN_THRESHOLD);

                phase = 'challenge';
                promptEl.textContent = prompt;
                statusEl.textContent = 'Now — do the action!';

                let secondsLeft = CHALLENGE_SECONDS;
                timerEl.textContent = secondsLeft;
                const ticker = setInterval(() => {
                    secondsLeft -= 1;
                    timerEl.textContent = Math.max(secondsLeft, 0);
                    if (passedFlag || secondsLeft <= 0) {
                        clearInterval(ticker);
                        finish();
                    }
                }, 1000);
            }, CALIB_MS);
        }

        function finish() {
            const passed = passedFlag;
            if (passed) {
                statusEl.textContent = 'Liveness confirmed ✓';
                overlay.querySelector('.lc-box').classList.add('lc-pass');
                toast('Liveness confirmed — checked in', 'ok');
            } else {
                statusEl.textContent = 'No real movement — check-in failed ✗';
                overlay.querySelector('.lc-box').classList.add('lc-fail');
                toast('Liveness failed — possible spoof', 'error');
            }
            setTimeout(cleanup, 1800);
        }

        function cleanup() {
            clearInterval(sampler);
            try { controller.abort(); } catch {}
            overlay.remove();
            btn.disabled = false;
        }
    }

    // Boot
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', buildPanel);
    } else {
        buildPanel();
    }
})();
