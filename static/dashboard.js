/**
 * StintEngine — Dashboard Client-Side Logic
 * Uses Server-Sent Events (SSE) for real-time updates + fallback polling.
 */

// ─── State ───────────────────────────────────────────────────────────────────
let rewardChart = null;
let tyreChart = null;
let eventSource = null;
let raceData = null;
let rewardHistory = [];
let telemetryLaps = [];

const COMPOUND_COLORS = { SOFT: '#FF3333', MEDIUM: '#FFD700', HARD: '#cccccc' };
const ACTION_NAMES = ['STAY OUT', 'PIT SOFT', 'PIT MED', 'PIT HARD'];

// ─── Initialize ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    connectSSE();
    fetchStatus();     // initial state
    fetchTyreModel();  // load curves
});

// ─── SSE Connection ──────────────────────────────────────────────────────────
function connectSSE() {
    if (eventSource) eventSource.close();

    eventSource = new EventSource('/api/stream');

    eventSource.addEventListener('connected', (e) => {
        const data = JSON.parse(e.data);
        updateDeviceBadge(data.device);
        console.log('[SSE] Connected, device:', data.device);
    });

    // ── Training Events ──
    eventSource.addEventListener('training_start', (e) => {
        const data = JSON.parse(e.data);
        setStatus('training', 'Training...');
        setMetric('metricStatus', 'TRAINING', 'yellow');
        showToast(`Training started — ${formatNumber(data.timesteps)} steps`);
        rewardHistory = [];
        setButtons(true);
        document.getElementById('progressContainer').style.display = 'block';
    });

    eventSource.addEventListener('training_progress', (e) => {
        const data = JSON.parse(e.data);
        setMetric('metricTimesteps', formatNumber(data.timestep));
        setMetric('metricEpisodes', data.episodes || '—', 'blue');
        setMetric('metricReward', data.mean_reward?.toFixed(1) || '—', 'green');

        // Progress bar
        if (data.total > 0) {
            const pct = Math.min((data.timestep / data.total) * 100, 100);
            document.getElementById('progressFill').style.width = pct + '%';
        }

        // Append to reward chart
        if (data.reward_history) {
            for (const h of data.reward_history) {
                if (!rewardHistory.find(r => r.episode === h.episode)) {
                    rewardHistory.push(h);
                }
            }
            updateRewardChart();
        }
    });

    eventSource.addEventListener('training_complete', (e) => {
        const data = JSON.parse(e.data);
        setStatus('active', 'Ready');
        setMetric('metricStatus', 'COMPLETE', 'green');
        setButtons(false);
        document.getElementById('progressContainer').style.display = 'none';
        showToast(`Training complete! ${data.episodes} episodes, reward: ${data.mean_reward?.toFixed(1)}, ${data.elapsed?.toFixed(0)}s`);
        fetchStatus(); // refresh model availability
    });

    eventSource.addEventListener('training_error', (e) => {
        const data = JSON.parse(e.data);
        setStatus('', 'Error');
        setMetric('metricStatus', 'ERROR', 'red');
        setButtons(false);
        document.getElementById('progressContainer').style.display = 'none';
        showToast(`Training error: ${data.error}`, true);
    });

    // ── Inference Events ──
    eventSource.addEventListener('infer_start', () => {
        setStatus('training', 'Racing...');
        setMetric('metricStatus', 'RACING', 'yellow');
        telemetryLaps = [];
        clearTelemetry();
        clearStrategy();
        setButtons(true);
    });

    eventSource.addEventListener('infer_lap', (e) => {
        const lap = JSON.parse(e.data);
        telemetryLaps.push(lap);
        appendTelemetryRow(lap);
    });

    eventSource.addEventListener('infer_complete', (e) => {
        const data = JSON.parse(e.data);
        raceData = data;
        setStatus('active', 'Ready');
        setMetric('metricStatus', 'COMPLETE', 'green');
        setButtons(false);
        renderStrategyTimeline(data.laps);
        showToast(`Race complete! P${data.final_position} | ${data.pit_stops} stops | Reward: ${data.total_reward?.toFixed(1)}`);
    });

    eventSource.addEventListener('infer_error', (e) => {
        const data = JSON.parse(e.data);
        setStatus('', 'Idle');
        setMetric('metricStatus', 'IDLE', '');
        setButtons(false);
        showToast(data.error, true);
    });

    // ── Connection management ──
    eventSource.onerror = () => {
        console.warn('[SSE] Connection lost, reconnecting in 3s...');
        eventSource.close();
        setTimeout(connectSSE, 3000);
    };
}

// ─── API Helpers ─────────────────────────────────────────────────────────────
async function api(url, options = {}) {
    try {
        const res = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        return await res.json();
    } catch (e) {
        showToast(`API Error: ${e.message}`, true);
        return null;
    }
}

// ─── Fallback Status Check ──────────────────────────────────────────────────
async function fetchStatus() {
    const data = await api('/api/status');
    if (!data) return;

    updateDeviceBadge(data.device);

    if (data.is_training) {
        setStatus('training', 'Training...');
        setMetric('metricStatus', 'TRAINING', 'yellow');
        setButtons(true);
    } else if (data.is_inferring) {
        setStatus('training', 'Racing...');
        setMetric('metricStatus', 'RACING', 'yellow');
        setButtons(true);
    } else if (data.status === 'complete') {
        setStatus('active', 'Ready');
        setMetric('metricStatus', 'COMPLETE', 'green');
        setButtons(false);
    } else {
        setStatus('', 'Idle');
        setMetric('metricStatus', 'IDLE', '');
        setButtons(false);
    }

    if (data.current_timestep) setMetric('metricTimesteps', formatNumber(data.current_timestep));
    if (data.episodes) setMetric('metricEpisodes', data.episodes, 'blue');
    if (data.mean_reward != null) setMetric('metricReward', data.mean_reward.toFixed(1), 'green');

    if (data.reward_history?.length) {
        rewardHistory = data.reward_history;
        updateRewardChart();
    }
}

// ─── Training ────────────────────────────────────────────────────────────────
async function startTraining(isFinal) {
    const timesteps = parseInt(document.getElementById('timestepsInput').value) || 50000;

    const data = await api('/api/train', {
        method: 'POST',
        body: JSON.stringify({ timesteps, final: isFinal }),
    });

    if (data?.error) {
        showToast(data.error, true);
    } else if (data?.message) {
        showToast(data.message);
    }
}

// ─── Inference ───────────────────────────────────────────────────────────────
async function runInference() {
    showToast('Starting race...');
    setButtons(true);

    const data = await api('/api/infer', { method: 'POST' });

    if (data?.error) {
        showToast(data.error, true);
        setButtons(false);
    } else if (data?.message) {
        showToast(data.message);
        // UI updates will arrive via SSE events
    }
}

// ─── Chart Initialization ────────────────────────────────────────────────────
function initCharts() {
    Chart.defaults.color = '#666';
    Chart.defaults.borderColor = '#222';
    Chart.defaults.font.family = "'IBM Plex Mono', monospace";
    Chart.defaults.font.size = 10;

    const ctx1 = document.getElementById('rewardChart').getContext('2d');
    rewardChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Mean Reward (50ep)',
                data: [],
                borderColor: '#00e676',
                backgroundColor: 'rgba(0,230,118,0.08)',
                fill: true,
                tension: 0.4,
                pointRadius: 2,
                pointBackgroundColor: '#00e676',
                borderWidth: 1.5,
            }, {
                label: 'Episode Reward',
                data: [],
                borderColor: 'rgba(41,121,255,0.4)',
                backgroundColor: 'transparent',
                tension: 0.2,
                pointRadius: 1,
                borderWidth: 1,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { boxWidth: 10, padding: 12 } },
            },
            scales: {
                x: { title: { display: true, text: 'Episode' }, grid: { color: '#1a1a1a' } },
                y: { title: { display: true, text: 'Reward' }, grid: { color: '#1a1a1a' } },
            },
        },
    });

    const ctx2 = document.getElementById('tyreChart').getContext('2d');
    tyreChart = new Chart(ctx2, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Tyre Age (laps)' }, grid: { color: '#1a1a1a' } },
                y: { title: { display: true, text: 'Lap Time (s)' }, grid: { color: '#1a1a1a' } },
            },
        },
    });
}

// ─── Reward Chart Update ─────────────────────────────────────────────────────
function updateRewardChart() {
    if (!rewardChart || !rewardHistory.length) return;
    rewardChart.data.labels = rewardHistory.map(h => h.episode);
    rewardChart.data.datasets[0].data = rewardHistory.map(h => h.mean_reward);
    rewardChart.data.datasets[1].data = rewardHistory.map(h => h.reward);
    rewardChart.update('none');
}

// ─── Tyre Model ──────────────────────────────────────────────────────────────
async function fetchTyreModel() {
    const data = await api('/api/tyre-model');
    if (!data || data.error) return;

    const compounds = ['SOFT', 'MEDIUM', 'HARD'];
    const colors = [COMPOUND_COLORS.SOFT, COMPOUND_COLORS.MEDIUM, COMPOUND_COLORS.HARD];

    tyreChart.data.labels = data[compounds[0]].predicted.map(p => p.age);
    tyreChart.data.datasets = compounds.map((comp, i) => ({
        label: comp,
        data: data[comp].predicted.map(p => p.laptime),
        borderColor: colors[i],
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
    }));
    tyreChart.update();
}

// ─── Strategy Timeline ───────────────────────────────────────────────────────
function renderStrategyTimeline(laps) {
    const container = document.getElementById('strategyContainer');
    if (!laps || !laps.length) return;

    const stints = extractStints(laps);
    const totalLaps = laps.length;

    container.innerHTML = `
        <div class="strategy-row">
            <div class="strategy-label">RL Agent</div>
            <div class="strategy-bar">
                ${stints.map(s => {
                    const width = ((s.end - s.start + 1) / totalLaps) * 100;
                    const cls = 'stint-' + s.compound.toLowerCase();
                    return '<div class="stint-block ' + cls + '" style="width:' + width + '%" title="' + s.compound + ': Lap ' + s.start + '-' + s.end + ' (' + (s.end - s.start + 1) + 'L)">' + s.compound[0] + ' ' + (s.end - s.start + 1) + 'L</div>';
                }).join('')}
            </div>
        </div>
    `;
}

function clearStrategy() {
    document.getElementById('strategyContainer').innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">🏎️</div>
            <div class="empty-state-text">Race in progress...</div>
        </div>`;
}

function extractStints(laps) {
    const stints = [];
    let current = laps[0].compound;
    let start = laps[0].lap;
    for (const lap of laps) {
        if (lap.compound !== current) {
            stints.push({ start, end: lap.lap - 1, compound: current });
            current = lap.compound;
            start = lap.lap;
        }
    }
    stints.push({ start, end: laps[laps.length - 1].lap, compound: current });
    return stints;
}

// ─── Telemetry Table (Streaming) ─────────────────────────────────────────────
function clearTelemetry() {
    const container = document.getElementById('telemetryContainer');
    container.innerHTML = `
        <div class="telemetry-container">
            <table class="telem-table">
                <thead>
                    <tr>
                        <th>Lap</th><th>Compound</th><th>T.Age</th><th>Action</th>
                        <th>Pos</th><th>Lap Time</th><th>Fuel</th><th>Gap</th>
                        <th>SC</th><th>Reward</th>
                    </tr>
                </thead>
                <tbody id="telemetryBody"></tbody>
            </table>
        </div>`;
}

function appendTelemetryRow(l) {
    let tbody = document.getElementById('telemetryBody');
    if (!tbody) {
        clearTelemetry();
        tbody = document.getElementById('telemetryBody');
    }

    const isPit = l.action > 0;
    const isSC = l.safety_car;
    const cls = isPit ? 'pit-row' : (isSC ? 'sc-row' : '');
    const compColor = COMPOUND_COLORS[l.compound] || '#888';

    const tr = document.createElement('tr');
    tr.className = cls;
    tr.innerHTML = `
        <td>${l.lap}</td>
        <td><span class="compound-dot" style="background:${compColor}"></span>${l.compound}</td>
        <td>${l.tyre_age}</td>
        <td>${isPit ? ACTION_NAMES[l.action] : '—'}</td>
        <td>P${l.position}</td>
        <td>${l.laptime.toFixed(2)}s</td>
        <td>${l.fuel_kg.toFixed(1)}</td>
        <td>${l.gap_to_leader.toFixed(1)}s</td>
        <td>${isSC ? '🟡' : ''}</td>
        <td>${l.reward >= 0 ? '+' : ''}${l.reward.toFixed(1)}</td>
    `;

    tbody.appendChild(tr);

    // Auto-scroll
    const tc = tbody.closest('.telemetry-container');
    if (tc) tc.scrollTop = tc.scrollHeight;

    // Flash animation
    tr.style.backgroundColor = 'rgba(232,0,45,0.15)';
    setTimeout(() => { tr.style.backgroundColor = ''; tr.style.transition = 'background 0.5s'; }, 100);
}

// ─── UI Helpers ──────────────────────────────────────────────────────────────
function setStatus(dotClass, text) {
    const dot = document.getElementById('statusDot');
    const el = document.getElementById('statusText');
    dot.className = 'status-dot' + (dotClass ? ' ' + dotClass : '');
    el.textContent = text;
}

function setMetric(id, value, colorClass) {
    const el = document.getElementById(id);
    el.textContent = value;
    if (colorClass !== undefined) {
        el.className = 'metric-value' + (colorClass ? ' ' + colorClass : '');
    }
}

function setButtons(disabled) {
    document.getElementById('btnTrain').disabled = disabled;
    document.getElementById('btnFinal').disabled = disabled;
    document.getElementById('btnInfer').disabled = disabled;
}

function updateDeviceBadge(device) {
    const badge = document.getElementById('deviceBadge');
    badge.textContent = (device || 'cpu').toUpperCase();
    badge.className = 'device-badge device-' + (device || 'cpu');
}

function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = isError ? 'toast error show' : 'toast show';
    setTimeout(() => toast.classList.remove('show'), 4000);
}

function formatNumber(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toString();
}
