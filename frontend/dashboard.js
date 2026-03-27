/**
 * StintEngine v2 — Dashboard Client-Side Logic
 * 10-panel system with SSE streaming, Chart.js, SVG gauges, and race history.
 */

// ─── State ───────────────────────────────────────────────────────────────────
let rewardChart = null;
let tyreChart = null;
let positionChart = null;
let lapDeltaChart = null;
let eventSource = null;
let raceData = {};
let rewardHistory = [];
let telemetryLaps = [];

const DRIVER_COLORS = ['#2979FF', '#FF3333', '#00e676', '#FFD700', '#E040FB'];
let driverColorMap = {};

function getDriverColor(driver) {
    if (!driverColorMap[driver]) {
        driverColorMap[driver] = DRIVER_COLORS[Object.keys(driverColorMap).length % DRIVER_COLORS.length];
    }
    return driverColorMap[driver];
}

const COMPOUND_COLORS = {
    SOFT: '#FF3333', MEDIUM: '#FFD700', HARD: '#cccccc',
    INTER: '#00e676', WET: '#2979FF',
};
const ACTION_NAMES = ['STAY OUT', 'PIT SOFT', 'PIT MED', 'PIT HARD', 'PIT INTER', 'PIT WET'];
const STARTING_POSITION = 10;
const FUEL_LOAD_KG = 110;

// ─── Initialize ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    initCharts();
    initTrackViz();
    connectSSE();
    fetchStatus();
    fetchTyreModel();
    fetchRaceHistory();
    // Fallback poll every 10s in case SSE events are missed
    setInterval(fetchStatus, 10000);
});

// ─── SSE Connection ──────────────────────────────────────────────────────────
function connectSSE() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource('/api/stream');

    eventSource.addEventListener('connected', (e) => {
        const data = JSON.parse(e.data);
        updateDeviceBadge(data.device);
        // Catch up on status in case we missed events during disconnect
        fetchStatus();
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

        if (data.total > 0) {
            const pct = Math.min((data.timestep / data.total) * 100, 100);
            document.getElementById('progressFill').style.width = pct + '%';
        }

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
        showToast(`Training complete! ${data.episodes} ep, reward: ${data.mean_reward?.toFixed(1)}, ${data.elapsed?.toFixed(0)}s`);
        fetchStatus();
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
        raceData = {};
        driverColorMap = {};
        clearTelemetry();
        clearStrategy();
        clearSummary();
        resetGauges();
        resetPositionChart();
        resetLapDeltaChart();
        setButtons(true);
        showRaceOverlay(true);
        // Reset and initialize track viz for this race
        resetTrackViz();
        const trackInput = document.getElementById('trackInput');
        if (trackInput && trackInput.value) setCurrentTrack(trackInput.value);
    });

    eventSource.addEventListener('infer_lap', (e) => {
        const lap = JSON.parse(e.data);
        const d = lap.driver || 'Agent';
        if (!raceData[d]) raceData[d] = [];
        raceData[d].push(lap);
        
        telemetryLaps.push(lap);
        appendTelemetryRow(lap);
        updateGauges(lap);
        updatePositionChart(lap);
        updateLapDeltaChart(lap);
        updateWeatherBadge(lap);
        updateRaceOverlayLap(lap.lap);
        updateTrackFromLap(lap);
    });

    eventSource.addEventListener('infer_complete', (e) => {
        const data = JSON.parse(e.data);
        showRaceOverlay(false);
        setStatus('active', 'Ready');
        setMetric('metricStatus', 'COMPLETE', 'green');
        setButtons(false);
        renderStrategyTimeline(data.drivers);
        
        // Show summary for the first driver
        const firstDriver = Object.keys(data.drivers)[0];
        renderRaceSummary(data.drivers[firstDriver], firstDriver);
        
        fetchRaceHistory();
        showToast(`Race complete! Multiple drivers finished.`);
        enableReplayButton();
    });

    eventSource.addEventListener('infer_error', (e) => {
        const data = JSON.parse(e.data);
        showRaceOverlay(false);
        setStatus('', 'Idle');
        setMetric('metricStatus', 'IDLE', '');
        setButtons(false);
        showToast(data.error, true);
    });

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

// ─── Fallback Status ─────────────────────────────────────────────────────────
async function fetchStatus() {
    const data = await api('/api/status');
    if (!data) return;

    updateDeviceBadge(data.device);

    if (data.is_training) {
        setStatus('training', 'Training...');
        setMetric('metricStatus', 'TRAINING', 'yellow');
        setButtons(true);
        // Clear old metrics from previous run while we wait for SSE updates
        setMetric('metricTimesteps', '—');
        setMetric('metricEpisodes', '—', 'blue');
        setMetric('metricReward', '—', 'green');
    } else if (data.is_inferring) {
        setStatus('training', 'Racing...');
        setMetric('metricStatus', 'RACING', 'yellow');
        setButtons(true);
        // Racing uses telemetry, not training metrics
        setMetric('metricTimesteps', '—');
        setMetric('metricEpisodes', '—', 'blue');
        setMetric('metricReward', '—', 'green');
    } else {
        // Idle / Complete — show the last run's metrics
        if (data.status === 'complete') {
            setStatus('active', 'Ready');
            setMetric('metricStatus', 'COMPLETE', 'green');
        } else {
            setStatus('', 'Idle');
            setMetric('metricStatus', 'IDLE', '');
        }
        setButtons(false);
        
        if (data.current_timestep) setMetric('metricTimesteps', formatNumber(data.current_timestep));
        else setMetric('metricTimesteps', '0');
        
        if (data.episodes) setMetric('metricEpisodes', data.episodes, 'blue');
        if (data.mean_reward != null) setMetric('metricReward', data.mean_reward.toFixed(1), 'green');

        if (data.reward_history?.length) {
            rewardHistory = data.reward_history;
            updateRewardChart();
        }
    }
}

// ─── Training ────────────────────────────────────────────────────────────────
async function startTraining(isFinal) {
    const timesteps = parseInt(document.getElementById('timestepsInput').value) || 50000;
    
    // Get simulation settings
    const year = document.getElementById('yearInput').value;
    const track = document.getElementById('trackInput').value;
    const drivers = document.getElementById('driversInput').value;
    const weather_mode = document.getElementById('weatherModeInput').value;
    const sc_probability = document.getElementById('scProbInput').value;
    const starting_position = document.getElementById('startPosInput').value;
    
    const payload = { 
        timesteps, 
        final: isFinal,
        weather_mode,
        sc_probability,
        starting_position
    };
    if (year) payload.year = year;
    if (track) payload.track = track;
    if (drivers) payload.drivers = drivers;

    const data = await api('/api/train', {
        method: 'POST',
        body: JSON.stringify(payload),
    });
    if (data?.error) showToast(data.error, true);
    else if (data?.message) showToast(data.message);
}

// ─── Inference ───────────────────────────────────────────────────────────────
async function runInference() {
    showToast('Starting race...');
    setButtons(true);
    
    // Get simulation settings
    const year = document.getElementById('yearInput').value;
    const track = document.getElementById('trackInput').value;
    const drivers = document.getElementById('driversInput').value;
    const weather_mode = document.getElementById('weatherModeInput').value;
    const sc_probability = document.getElementById('scProbInput').value;
    const starting_position = document.getElementById('startPosInput').value;
    
    const payload = {
        weather_mode,
        sc_probability,
        starting_position
    };
    if (year) payload.year = year;
    if (track) payload.track = track;
    if (drivers) payload.drivers = drivers;

    const data = await api('/api/infer', { 
        method: 'POST',
        body: JSON.stringify(payload)
    });
    if (data?.error) { showToast(data.error, true); setButtons(false); }
    else if (data?.message) showToast(data.message);
}

// ─── Settings Panel (Slide-out) ──────────────────────────────────────────────
function initSettings() {
    // Populate track dropdown
    const trackSelect = document.getElementById('trackInput');
    if (trackSelect && typeof F1_TRACKS !== 'undefined') {
        trackSelect.innerHTML = Object.keys(F1_TRACKS).sort().map(k => {
            const t = F1_TRACKS[k];
            return `<option value="${k}">${t.fullName} (${t.country})</option>`;
        }).join('');
    }
}

function loadSettings() {
    initSettings();
    const settings = JSON.parse(localStorage.getItem('stintSettings') || '{}');
    if (settings.apiBase) document.getElementById('apiBaseInput').value = settings.apiBase;
    if (settings.year) document.getElementById('yearInput').value = settings.year;
    
    // Set track if valid, otherwise default to bahrain
    const trackVal = settings.track ? settings.track.toLowerCase() : 'bahrain';
    const trackSelect = document.getElementById('trackInput');
    if (trackSelect) {
        if (Array.from(trackSelect.options).some(o => o.value === trackVal)) {
            trackSelect.value = trackVal;
        } else {
            trackSelect.value = 'bahrain';
        }
    }
    
    if (settings.drivers !== undefined) document.getElementById('driversInput').value = settings.drivers;
    if (settings.weatherMode) document.getElementById('weatherModeInput').value = settings.weatherMode;
    if (settings.scProb !== undefined) {
        const scProbInput = document.getElementById('scProbInput');
        scProbInput.value = settings.scProb;
        document.getElementById('scProbLabel').textContent = Math.round(settings.scProb * 100) + '%';
    }
    if (settings.startPos) document.getElementById('startPosInput').value = settings.startPos;
}

function openSettings() {
    document.getElementById('settingsPanel').classList.add('show');
    document.getElementById('settingsOverlay').classList.add('show');
}

function closeSettings() {
    document.getElementById('settingsPanel').classList.remove('show');
    document.getElementById('settingsOverlay').classList.remove('show');
}

function saveSettings() {
    const trackVal = document.getElementById('trackInput').value;
    const settings = {
        apiBase: document.getElementById('apiBaseInput').value.trim(),
        year: document.getElementById('yearInput').value.trim(),
        track: trackVal,
        drivers: document.getElementById('driversInput').value.trim(),
        weatherMode: document.getElementById('weatherModeInput').value,
        scProb: parseFloat(document.getElementById('scProbInput').value) || 0,
        startPos: parseInt(document.getElementById('startPosInput').value) || 10
    };
    localStorage.setItem('stintSettings', JSON.stringify(settings));
    closeSettings();
    showToast('Settings saved successfully');
    
    // Update global state if needed
    window.API_BASE = settings.apiBase || 'http://127.0.0.1:5000';
    
    // Force a reload of the track visualization
    if (typeof loadTrack === 'function') {
        loadTrack(settings.track);
    }
}

// Close panel on Esc key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeSettings();
});
function initCharts() {
    Chart.defaults.color = '#666';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
    Chart.defaults.font.family = "'IBM Plex Mono', monospace";
    Chart.defaults.font.size = 10;

    // 02 — Reward Chart
    rewardChart = new Chart(document.getElementById('rewardChart').getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Mean (50ep)',
                data: [],
                borderColor: '#00e676',
                backgroundColor: 'rgba(0,230,118,0.08)',
                fill: true, tension: 0.4, pointRadius: 1.5, borderWidth: 1.5,
            }, {
                label: 'Episode',
                data: [],
                borderColor: 'rgba(41,121,255,0.3)',
                tension: 0.2, pointRadius: 0.5, borderWidth: 1,
            }],
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 8, padding: 8, font: { size: 9 } } } },
            scales: {
                x: { title: { display: true, text: 'Episode', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
                y: { title: { display: true, text: 'Reward', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
            },
        },
    });

    // 04 — Position Tracker
    positionChart = new Chart(document.getElementById('positionChart').getContext('2d'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: true, labels: { color: 'rgba(255,255,255,0.7)', font: { size: 10 } } } },
            scales: {
                x: { title: { display: true, text: 'Lap', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
                y: { reverse: true, min: 1, max: 20, title: { display: true, text: 'Position', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
            },
        },
    });

    // 05 — Lap Delta
    lapDeltaChart = new Chart(document.getElementById('lapDeltaChart').getContext('2d'), {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: true, labels: { color: 'rgba(255,255,255,0.7)', font: { size: 10 } } } },
            scales: {
                x: { title: { display: true, text: 'Lap', font: { size: 9 } }, grid: { display: false } },
                y: { title: { display: true, text: 'Delta (s)', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
            },
        },
    });

    // 08 — Tyre Degradation
    tyreChart = new Chart(document.getElementById('tyreChart').getContext('2d'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Tyre Age (laps)', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
                y: { title: { display: true, text: 'Lap Time (s)', font: { size: 9 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
            },
        },
    });
}

// ─── Chart Updates ───────────────────────────────────────────────────────────
function updateRewardChart() {
    if (!rewardChart || !rewardHistory.length) return;
    rewardChart.data.labels = rewardHistory.map(h => h.episode);
    rewardChart.data.datasets[0].data = rewardHistory.map(h => h.mean_reward);
    rewardChart.data.datasets[1].data = rewardHistory.map(h => h.reward);
    rewardChart.update('none');
}

function updatePositionChart(lap) {
    if (!positionChart) return;
    const d = lap.driver || 'Agent';
    let dataset = positionChart.data.datasets.find(ds => ds.label === d);
    if (!dataset) {
        dataset = {
            label: d,
            data: [],
            borderColor: getDriverColor(d),
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.2
        };
        positionChart.data.datasets.push(dataset);
    }
    
    if (!positionChart.data.labels.includes(lap.lap)) {
        positionChart.data.labels.push(lap.lap);
    }
    
    dataset.data.push(lap.position);
    positionChart.update('none');
}

function resetPositionChart() {
    if (!positionChart) return;
    positionChart.data.labels = [];
    positionChart.data.datasets = [];
    positionChart.update('none');
}

function updateLapDeltaChart(lap) {
    if (!lapDeltaChart) return;
    const d = lap.driver || 'Agent';
    let dataset = lapDeltaChart.data.datasets.find(ds => ds.label === d);
    if (!dataset) {
        dataset = {
            label: d,
            data: [],
            backgroundColor: [],
            borderRadius: 2,
            barPercentage: 0.8,
        };
        lapDeltaChart.data.datasets.push(dataset);
    }
    
    if (!lapDeltaChart.data.labels.includes(lap.lap)) {
        lapDeltaChart.data.labels.push(lap.lap);
    }
    
    const baseLaptime = 91.0; 
    const delta = lap.laptime - baseLaptime;
    dataset.data.push(delta > 20 ? null : delta); 
    
    const color = COMPOUND_COLORS[lap.compound] || '#888';
    dataset.backgroundColor.push(lap.safety_car ? 'rgba(255,214,0,0.3)' : color + '80');
    
    lapDeltaChart.update('none');
}

function resetLapDeltaChart() {
    if (!lapDeltaChart) return;
    lapDeltaChart.data.labels = [];
    lapDeltaChart.data.datasets = [];
    lapDeltaChart.update('none');
}

// ─── Tyre Model ──────────────────────────────────────────────────────────────
async function fetchTyreModel() {
    const data = await api('/api/tyre-model');
    if (!data || data.error) return;

    const compounds = Object.keys(data);
    const colors = compounds.map(c => COMPOUND_COLORS[c] || '#888');

    tyreChart.data.labels = data[compounds[0]].predicted.map(p => p.age);
    tyreChart.data.datasets = compounds.map((comp, i) => ({
        label: comp,
        data: data[comp].predicted.map(p => p.laptime),
        borderColor: colors[i],
        backgroundColor: 'transparent',
        tension: 0.3, pointRadius: 0, borderWidth: 2,
    }));
    tyreChart.update();
}

// ─── SVG Gauges ──────────────────────────────────────────────────────────────
const GAUGE_CIRCUMFERENCE = 314; // 2 * PI * 50

function updateGauges(lap) {
    // Fuel gauge (110kg -> 0kg)
    const fuelPct = Math.max(0, lap.fuel_kg / FUEL_LOAD_KG);
    const fuelOffset = GAUGE_CIRCUMFERENCE * (1 - fuelPct);
    document.getElementById('fuelGauge').style.strokeDashoffset = fuelOffset;
    document.getElementById('fuelText').textContent = Math.round(fuelPct * 100) + '%';

    // Tyre gauge (inverse of age — 100% new, 0% worn out)
    const tyreLife = Math.max(0, 1 - (lap.tyre_age / 40));
    const tyreOffset = GAUGE_CIRCUMFERENCE * (1 - tyreLife);
    document.getElementById('tyreGauge').style.strokeDashoffset = tyreOffset;
    document.getElementById('tyreText').textContent = Math.round(tyreLife * 100) + '%';

    // Tyre gauge color based on compound
    const tyreGauge = document.getElementById('tyreGauge');
    tyreGauge.style.stroke = COMPOUND_COLORS[lap.compound] || '#E8002D';

    // Info row
    document.getElementById('monitorCompound').textContent = lap.compound;
    document.getElementById('monitorCompound').style.color = COMPOUND_COLORS[lap.compound] || '#fff';
    document.getElementById('monitorTyreAge').textContent = lap.tyre_age + 'L';

    const weatherStr = lap.is_raining
        ? (lap.rain_intensity === 2 ? '🌧️ Heavy' : '🌦️ Light')
        : '☀️ Dry';
    document.getElementById('monitorWeather').textContent = weatherStr;
}

function resetGauges() {
    document.getElementById('fuelGauge').style.strokeDashoffset = 0;
    document.getElementById('tyreGauge').style.strokeDashoffset = 0;
    document.getElementById('fuelText').textContent = '100%';
    document.getElementById('tyreText').textContent = '100%';
    document.getElementById('monitorCompound').textContent = '—';
    document.getElementById('monitorTyreAge').textContent = '—';
    document.getElementById('monitorWeather').textContent = '☀️ Dry';
}

function updateWeatherBadge(lap) {
    const badge = document.getElementById('weatherBadge');
    if (lap.is_raining) {
        if (lap.rain_intensity === 2) {
            badge.textContent = '🌧️ HEAVY';
            badge.className = 'weather-badge weather-heavy';
        } else {
            badge.textContent = '🌦️ LIGHT';
            badge.className = 'weather-badge weather-light';
        }
    } else {
        badge.textContent = '☀️ DRY';
        badge.className = 'weather-badge weather-dry';
    }
}

// ─── Race Summary ────────────────────────────────────────────────────────────
function renderRaceSummary(data, driverName) {
    const container = document.getElementById('summaryContainer');
    const posGain = data.positions_gained;
    const posClass = posGain > 0 ? 'positive' : (posGain < 0 ? 'negative' : '');
    const posText = posGain > 0 ? `+${posGain}` : `${posGain}`;

    container.innerHTML = `
        <div style="font-size: 10px; color: #888; position: absolute; top: 10px; right: 15px;">${driverName}</div>
        <div class="summary-grid">
            <div class="summary-stat ${posClass}">
                <div class="summary-stat-value">P${data.final_position}</div>
                <div class="summary-stat-label">Final Position</div>
            </div>
            <div class="summary-stat ${posClass}">
                <div class="summary-stat-value">${posText}</div>
                <div class="summary-stat-label">Positions ${posGain >= 0 ? 'Gained' : 'Lost'}</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">${data.pit_stops}</div>
                <div class="summary-stat-label">Pit Stops</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">${data.total_reward?.toFixed(1)}</div>
                <div class="summary-stat-label">Total Reward</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">${data.best_lap?.time?.toFixed(2)}s</div>
                <div class="summary-stat-label">Best Lap (L${data.best_lap?.lap})</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">${data.rain_laps || 0}</div>
                <div class="summary-stat-label">Rain Laps</div>
            </div>
        </div>`;
}

function clearSummary() {
    document.getElementById('summaryContainer').innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">🏁</div>
            <div class="empty-state-text">Race in progress...</div>
        </div>`;
}

// ─── Strategy Timeline ───────────────────────────────────────────────────────
function renderStrategyTimeline(driversData) {
    const container = document.getElementById('strategyContainer');
    if (!driversData) return;

    let html = '';
    
    // We only need to extract rain zones once (use the first driver's history)
    let rainZones = [];
    const drivers = Object.keys(driversData);
    if (drivers.length > 0) {
        rainZones = extractRainZones(driversData[drivers[0]].laps);
    }

    for (const d of drivers) {
        const laps = driversData[d].laps;
        if (!laps || laps.length === 0) continue;
        
        const stints = extractStints(laps);
        const totalLaps = laps.length;
        const color = getDriverColor(d);

        html += `
            <div class="strategy-row">
                <div class="strategy-label" style="color: ${color}">${d}</div>
                <div class="strategy-bar" style="position:relative">
                    ${stints.map(s => {
                        const width = ((s.end - s.start + 1) / totalLaps) * 100;
                        const cls = 'stint-' + s.compound.toLowerCase();
                        return `<div class="stint-block ${cls}" style="width:${width}%"
                            title="${s.compound}: Lap ${s.start}-${s.end} (${s.end - s.start + 1}L)">
                            ${s.compound[0]} ${s.end - s.start + 1}L</div>`;
                    }).join('')}
                    ${rainZones.map(r => {
                        const left = ((r.start - 1) / totalLaps) * 100;
                        const width = ((r.end - r.start + 1) / totalLaps) * 100;
                        return `<div class="rain-marker" style="left:${left}%;width:${width}%" title="Rain: Lap ${r.start}-${r.end}"></div>`;
                    }).join('')}
                </div>
            </div>`;
    }

    html += `
        <div style="margin-top:6px;display:flex;gap:14px;padding-left:92px">
            <span style="font:9px 'IBM Plex Mono';color:#555">
                <span style="display:inline-block;width:8px;height:8px;background:rgba(41,121,255,0.3);margin-right:3px;vertical-align:middle;border-left:2px solid rgba(41,121,255,0.6)"></span>Rain
            </span>
        </div>`;
        
    container.innerHTML = html;
}

function extractRainZones(laps) {
    const zones = [];
    let start = null;
    for (const lap of laps) {
        if (lap.is_raining && start === null) start = lap.lap;
        if (!lap.is_raining && start !== null) {
            zones.push({ start, end: lap.lap - 1 });
            start = null;
        }
    }
    if (start !== null) zones.push({ start, end: laps[laps.length - 1].lap });
    return zones;
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

// ─── Telemetry Table ─────────────────────────────────────────────────────────
function clearTelemetry() {
    const container = document.getElementById('telemetryContainer');
    container.innerHTML = `
        <div class="telemetry-container">
            <table class="telem-table">
                <thead>
                    <tr>
                        <th>Driver</th><th>Lap</th><th>Compound</th><th>T.Age</th><th>Action</th>
                        <th>Pos</th><th>Lap Time</th><th>Fuel</th><th>Gap</th>
                        <th>SC</th><th>🌧</th><th>Reward</th>
                    </tr>
                </thead>
                <tbody id="telemetryBody"></tbody>
            </table>
        </div>`;
}

function appendTelemetryRow(l) {
    let tbody = document.getElementById('telemetryBody');
    if (!tbody) { clearTelemetry(); tbody = document.getElementById('telemetryBody'); }

    const isPit = l.action > 0;
    const isSC = l.safety_car;
    const isRain = l.is_raining;
    let cls = isPit ? 'pit-row' : (isSC ? 'sc-row' : (isRain ? 'rain-row' : ''));
    const compColor = COMPOUND_COLORS[l.compound] || '#888';
    const rainIcon = isRain ? (l.rain_intensity === 2 ? '🌧️' : '🌦️') : '';
    const d = l.driver || 'Agent';
    const dColor = getDriverColor(d);

    const tr = document.createElement('tr');
    tr.className = cls;
    tr.innerHTML = `
        <td style="color:${dColor}; font-weight:700;">${d}</td>
        <td>${l.lap}</td>
        <td><span class="compound-dot" style="background:${compColor}"></span>${l.compound}</td>
        <td>${l.tyre_age}</td>
        <td>${isPit ? ACTION_NAMES[l.action] : '—'}</td>
        <td>P${l.position}</td>
        <td>${l.laptime.toFixed(2)}s</td>
        <td>${l.fuel_kg.toFixed(1)}</td>
        <td>${l.gap_to_leader.toFixed(1)}s</td>
        <td>${isSC ? '🟡' : ''}</td>
        <td>${rainIcon}</td>
        <td>${l.reward >= 0 ? '+' : ''}${l.reward.toFixed(1)}</td>
    `;

    tbody.appendChild(tr);

    const tc = tbody.closest('.telemetry-container');
    if (tc) tc.scrollTop = tc.scrollHeight;

    tr.style.backgroundColor = 'rgba(232,0,45,0.12)';
    setTimeout(() => { tr.style.backgroundColor = ''; tr.style.transition = 'background 0.5s'; }, 100);
}

// ─── Race History ────────────────────────────────────────────────────────────
async function fetchRaceHistory() {
    const data = await api('/api/race-history');
    if (!data || !data.races) return;
    renderRaceHistory(data.races);
}

function renderRaceHistory(races) {
    const container = document.getElementById('historyContainer');
    if (!races.length) {
        container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📋</div><div class="empty-state-text">No races recorded yet</div></div>`;
        return;
    }

    const rows = races.slice(-10).reverse().map(r => {
        const posClass = r.positions_gained > 0 ? 'color:var(--green)' : (r.positions_gained < 0 ? 'color:var(--red)' : '');
        const posText = r.positions_gained > 0 ? `+${r.positions_gained}` : `${r.positions_gained}`;
        const driversStr = (r.drivers && r.drivers.length > 0) ? r.drivers.join(', ') : 'Agent';
        return `<tr>
            <td>#${r.id}</td>
            <td style="color:#aaa; font-size:9px;">${driversStr}</td>
            <td>P${r.final_position}</td>
            <td style="${posClass}">${posText}</td>
            <td>${r.pit_stops}</td>
            <td>${r.total_reward?.toFixed(1)}</td>
            <td>${r.rain_laps || 0}</td>
            <td>${r.best_lap_time?.toFixed(2) || '—'}</td>
        </tr>`;
    }).join('');

    container.innerHTML = `
        <div class="telemetry-container" style="max-height:280px">
            <table class="history-table">
                <thead><tr>
                    <th>#</th><th>Drivers</th><th>Pos</th><th>+/-</th><th>Pits</th><th>Reward</th><th>🌧</th><th>Best Lap</th>
                </tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
}

// ─── UI Helpers ──────────────────────────────────────────────────────────────
function setStatus(dotClass, text) {
    document.getElementById('statusDot').className = 'status-dot' + (dotClass ? ' ' + dotClass : '');
    document.getElementById('statusText').textContent = text;
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

// ─── Race Loading Overlay ────────────────────────────────────────────────────
function showRaceOverlay(show) {
    const overlay = document.getElementById('raceLoadingOverlay');
    if (!overlay) return;
    if (show) {
        overlay.classList.add('active');
        updateRaceOverlayLap(0);
    } else {
        overlay.classList.remove('active');
    }
}

function updateRaceOverlayLap(lapNum) {
    const el = document.getElementById('raceLoadingLap');
    if (!el) return;
    el.innerHTML = `LAP <span class="lap-num">${lapNum}</span> / 57`;
}

// ─── Track Visualization ─────────────────────────────────────────────────────
let trackCanvas, trackCtx;
let currentTrack = null;
let trackDriverStates = {};  // { driver: { lap, position, compound, laptime, progress } }
let trackReplayData = [];   // Array of lap snapshots for replay
let trackReplayTimer = null;
let trackReplayIndex = 0;
let isReplaying = false;
let trackAnimFrame = null;

/**
 * Initialize the track canvas and draw idle state.
 */
function initTrackViz() {
    trackCanvas = document.getElementById('trackCanvas');
    if (!trackCanvas) return;
    trackCtx = trackCanvas.getContext('2d');
    resizeTrackCanvas();
    window.addEventListener('resize', resizeTrackCanvas);
    
    // Default track from input field or Bahrain
    const trackInput = document.getElementById('trackInput');
    const trackName = trackInput ? trackInput.value : 'bahrain';
    setCurrentTrack(trackName);
}

function resizeTrackCanvas() {
    if (!trackCanvas) return;
    const container = trackCanvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    trackCanvas.width = container.clientWidth * dpr;
    trackCanvas.height = container.clientHeight * dpr;
    trackCtx.scale(dpr, dpr);
    trackCanvas.style.width = container.clientWidth + 'px';
    trackCanvas.style.height = container.clientHeight + 'px';
    drawTrack();
}

function setCurrentTrack(name) {
    if (typeof getTrackData === 'function') {
        currentTrack = getTrackData(name);
    } else {
        currentTrack = null;
    }
    const nameEl = document.getElementById('trackName');
    if (nameEl && currentTrack) nameEl.textContent = currentTrack.name;
    drawTrack();
}

// ─── Catmull-Rom Spline ──────────────────────────────────────────────────────
function catmullRomPoint(p0, p1, p2, p3, t) {
    const t2 = t * t, t3 = t2 * t;
    return [
        0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3),
        0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3),
    ];
}

/**
 * Generate smooth points along the track using Catmull-Rom.
 * Returns an array of [x,y] in canvas coordinates.
 */
function getTrackPath(track, w, h) {
    if (!track || !track.points || track.points.length < 4) return [];
    const pts = track.points;
    const pad = 30; // padding
    const pw = w - pad * 2, ph = h - pad * 2;
    const result = [];
    const segPoints = 20; // smoothness per segment

    for (let i = 0; i < pts.length; i++) {
        const p0 = pts[(i - 1 + pts.length) % pts.length];
        const p1 = pts[i];
        const p2 = pts[(i + 1) % pts.length];
        const p3 = pts[(i + 2) % pts.length];
        for (let t = 0; t < segPoints; t++) {
            const [x, y] = catmullRomPoint(p0, p1, p2, p3, t / segPoints);
            result.push([pad + x * pw, pad + y * ph]);
        }
    }
    return result;
}

/**
 * Get a position on the path at fraction [0,1].
 */
function getPointOnPath(path, fraction) {
    if (!path.length) return [0, 0];
    const f = ((fraction % 1) + 1) % 1;
    const idx = f * (path.length - 1);
    const i = Math.floor(idx);
    const t = idx - i;
    const a = path[i];
    const b = path[Math.min(i + 1, path.length - 1)];
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

// ─── Drawing ─────────────────────────────────────────────────────────────────
function drawTrack() {
    if (!trackCtx || !trackCanvas) return;
    const w = trackCanvas.width / (window.devicePixelRatio || 1);
    const h = trackCanvas.height / (window.devicePixelRatio || 1);
    
    trackCtx.clearRect(0, 0, w, h);
    
    if (!currentTrack) {
        // Draw placeholder
        trackCtx.fillStyle = '#333';
        trackCtx.font = '14px "IBM Plex Mono", monospace';
        trackCtx.textAlign = 'center';
        trackCtx.fillText('Select a track to view circuit', w/2, h/2);
        return;
    }

    const path = getTrackPath(currentTrack, w, h);
    if (!path.length) return;

    // Track outline (dark outer stroke)
    trackCtx.beginPath();
    trackCtx.moveTo(path[0][0], path[0][1]);
    for (let i = 1; i < path.length; i++) {
        trackCtx.lineTo(path[i][0], path[i][1]);
    }
    trackCtx.closePath();
    trackCtx.strokeStyle = 'rgba(255,255,255,0.08)';
    trackCtx.lineWidth = 22;
    trackCtx.lineJoin = 'round';
    trackCtx.stroke();

    // Track surface
    trackCtx.beginPath();
    trackCtx.moveTo(path[0][0], path[0][1]);
    for (let i = 1; i < path.length; i++) {
        trackCtx.lineTo(path[i][0], path[i][1]);
    }
    trackCtx.closePath();
    trackCtx.strokeStyle = 'rgba(255,255,255,0.18)';
    trackCtx.lineWidth = 14;
    trackCtx.stroke();

    // Centre line (racing line hint)
    trackCtx.beginPath();
    trackCtx.moveTo(path[0][0], path[0][1]);
    for (let i = 1; i < path.length; i++) {
        trackCtx.lineTo(path[i][0], path[i][1]);
    }
    trackCtx.closePath();
    trackCtx.strokeStyle = 'rgba(232,0,45,0.12)';
    trackCtx.lineWidth = 2;
    trackCtx.setLineDash([6, 8]);
    trackCtx.stroke();
    trackCtx.setLineDash([]);

    // Sector markers
    const sectorColors = ['rgba(255,255,255,0.4)', 'rgba(255,214,0,0.4)', 'rgba(0,230,118,0.4)'];
    currentTrack.sectors.forEach((frac, i) => {
        const pt = getPointOnPath(path, frac);
        trackCtx.beginPath();
        trackCtx.arc(pt[0], pt[1], 4, 0, Math.PI * 2);
        trackCtx.fillStyle = sectorColors[i] || '#fff';
        trackCtx.fill();
    });

    // Start/Finish line
    const sf = getPointOnPath(path, currentTrack.startFinish || 0);
    const sfNext = getPointOnPath(path, 0.002);
    const angle = Math.atan2(sfNext[1] - sf[1], sfNext[0] - sf[0]);
    const perpAngle = angle + Math.PI / 2;
    trackCtx.beginPath();
    trackCtx.moveTo(sf[0] + Math.cos(perpAngle) * 14, sf[1] + Math.sin(perpAngle) * 14);
    trackCtx.lineTo(sf[0] - Math.cos(perpAngle) * 14, sf[1] - Math.sin(perpAngle) * 14);
    trackCtx.strokeStyle = '#E8002D';
    trackCtx.lineWidth = 3;
    trackCtx.stroke();

    // Draw DRS text at start
    trackCtx.fillStyle = 'rgba(232,0,45,0.6)';
    trackCtx.font = '8px "IBM Plex Mono", monospace';
    trackCtx.textAlign = 'center';
    trackCtx.fillText('S/F', sf[0], sf[1] - 18);

    // Circuit name watermark
    trackCtx.fillStyle = 'rgba(255,255,255,0.03)';
    trackCtx.font = '60px "Bebas Neue", sans-serif';
    trackCtx.textAlign = 'center';
    trackCtx.fillText(currentTrack.name, w/2, h/2 + 20);

    // Draw driver markers
    drawDriverMarkers(path, w, h);
}

function drawDriverMarkers(path, w, h) {
    const drivers = Object.entries(trackDriverStates);
    if (!drivers.length || !path.length) return;
    
    // Sort by position (P1 on top)
    drivers.sort((a, b) => (a[1].position || 20) - (b[1].position || 20));

    drivers.forEach(([name, state]) => {
        const color = getDriverColor(name);
        const progress = state.progress || 0;
        const pt = getPointOnPath(path, progress);
        
        // Glow
        const grd = trackCtx.createRadialGradient(pt[0], pt[1], 0, pt[0], pt[1], 16);
        grd.addColorStop(0, color + '44');
        grd.addColorStop(1, 'transparent');
        trackCtx.fillStyle = grd;
        trackCtx.beginPath();
        trackCtx.arc(pt[0], pt[1], 16, 0, Math.PI * 2);
        trackCtx.fill();

        // Dot
        trackCtx.beginPath();
        trackCtx.arc(pt[0], pt[1], 6, 0, Math.PI * 2);
        trackCtx.fillStyle = color;
        trackCtx.fill();
        trackCtx.strokeStyle = 'rgba(0,0,0,0.5)';
        trackCtx.lineWidth = 1.5;
        trackCtx.stroke();

        // Position number inside dot
        trackCtx.fillStyle = '#fff';
        trackCtx.font = 'bold 7px "IBM Plex Mono", monospace';
        trackCtx.textAlign = 'center';
        trackCtx.textBaseline = 'middle';
        trackCtx.fillText('P' + (state.position || '?'), pt[0], pt[1]);

        // Name label
        trackCtx.fillStyle = color;
        trackCtx.font = '9px "IBM Plex Mono", monospace';
        trackCtx.textAlign = 'left';
        trackCtx.textBaseline = 'bottom';
        const shortName = name.length > 6 ? name.substring(0, 6) : name;
        trackCtx.fillText(shortName.toUpperCase(), pt[0] + 10, pt[1] - 4);

        // Compound dot
        if (state.compound && COMPOUND_COLORS[state.compound.toUpperCase()]) {
            trackCtx.beginPath();
            trackCtx.arc(pt[0] + 8, pt[1] + 6, 3, 0, Math.PI * 2);
            trackCtx.fillStyle = COMPOUND_COLORS[state.compound.toUpperCase()];
            trackCtx.fill();
        }
    });
}

// ─── Update from SSE ─────────────────────────────────────────────────────────
function updateTrackFromLap(lapData) {
    if (!currentTrack) {
        // Try to detect track from settings
        const trackInput = document.getElementById('trackInput');
        if (trackInput && trackInput.value) setCurrentTrack(trackInput.value);
    }
    
    const driver = lapData.driver || 'Agent';
    const totalLaps = currentTrack ? currentTrack.totalLaps : 57;
    
    // Calculate progress around the track (spread drivers by position)
    const baseFraction = (lapData.lap || 1) / totalLaps;
    // Offset each driver slightly by position to avoid overlap
    const posOffset = ((lapData.position || 1) - 1) * 0.015;
    
    trackDriverStates[driver] = {
        lap: lapData.lap || 0,
        position: lapData.position || 20,
        compound: lapData.compound || 'MEDIUM',
        laptime: lapData.laptime || 0,
        progress: (baseFraction + posOffset) % 1,
        gap: lapData.gap_to_leader || 0,
    };

    // Store snapshot for replay
    trackReplayData.push(JSON.parse(JSON.stringify({ 
        lap: lapData.lap, 
        driver, 
        states: { ...trackDriverStates } 
    })));

    // Update badge
    const badge = document.getElementById('trackLapBadge');
    if (badge) badge.textContent = `LAP ${lapData.lap} / ${totalLaps}`;

    // Update sidebar
    updateTrackSidebar();

    // Redraw
    drawTrack();
}

function updateTrackSidebar() {
    const list = document.getElementById('trackDriverList');
    if (!list) return;
    
    const drivers = Object.entries(trackDriverStates)
        .sort((a, b) => (a[1].position || 20) - (b[1].position || 20));
    
    if (!drivers.length) return;
    
    list.innerHTML = drivers.map(([name, state]) => {
        const color = getDriverColor(name);
        const compColor = COMPOUND_COLORS[(state.compound || 'MEDIUM').toUpperCase()] || '#888';
        const shortName = name.length > 8 ? name.substring(0, 8) : name;
        const gap = state.position === 1 ? 'LEADER' : `+${(state.gap || 0).toFixed(1)}s`;
        return `
            <li class="track-driver-item">
                <span class="track-driver-dot" style="background:${color}"></span>
                <span class="track-driver-pos">${state.position}</span>
                <span class="track-driver-name">${shortName}</span>
                <span class="track-driver-compound" style="background:${compColor}" title="${state.compound}"></span>
                <span style="font-size:8px;color:var(--muted)">${gap}</span>
            </li>`;
    }).join('');
}

function resetTrackViz() {
    trackDriverStates = {};
    trackReplayData = [];
    trackReplayIndex = 0;
    isReplaying = false;
    if (trackReplayTimer) clearInterval(trackReplayTimer);
    const btn = document.getElementById('btnReplay');
    if (btn) { btn.disabled = true; btn.textContent = '▶ REPLAY'; }
    const badge = document.getElementById('trackLapBadge');
    if (badge && currentTrack) badge.textContent = `LAP 0 / ${currentTrack.totalLaps}`;
    const list = document.getElementById('trackDriverList');
    if (list) list.innerHTML = '<li class="track-driver-item" style="color:var(--muted);font-size:9px;justify-content:center;padding:20px 0;">Awaiting race data...</li>';
    drawTrack();
}

// ─── Replay (smooth rAF-based) ───────────────────────────────────────────────
let replayStartTime = 0;
const REPLAY_SNAP_DURATION = 350; // ms per snapshot transition

function toggleTrackReplay() {
    if (isReplaying) {
        stopReplay();
    } else {
        startReplay();
    }
}

function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function lerpStates(statesA, statesB, t) {
    const eased = easeInOutCubic(Math.max(0, Math.min(1, t)));
    const merged = {};
    const allDrivers = new Set([...Object.keys(statesA || {}), ...Object.keys(statesB || {})]);
    for (const driver of allDrivers) {
        const a = (statesA || {})[driver];
        const b = (statesB || {})[driver];
        if (a && b) {
            merged[driver] = {
                lap: b.lap,
                position: b.position,
                compound: b.compound,
                laptime: b.laptime,
                progress: a.progress + (b.progress - a.progress) * eased,
                gap: a.gap + (b.gap - a.gap) * eased,
            };
        } else if (b) {
            merged[driver] = { ...b, progress: b.progress * eased };
        } else if (a) {
            merged[driver] = { ...a };
        }
    }
    return merged;
}

function startReplay() {
    if (!trackReplayData.length) return;
    isReplaying = true;
    trackReplayIndex = 0;
    trackDriverStates = {};
    const btn = document.getElementById('btnReplay');
    if (btn) btn.textContent = '⏸ PAUSE';
    replayStartTime = performance.now();
    trackAnimFrame = requestAnimationFrame(replayTick);
}

function replayTick(timestamp) {
    if (!isReplaying) return;

    const elapsed = timestamp - replayStartTime;
    const totalIdx = elapsed / REPLAY_SNAP_DURATION;
    const snapIdx = Math.floor(totalIdx);
    const t = totalIdx - snapIdx; // 0..1 fraction within current transition

    if (snapIdx >= trackReplayData.length) {
        // Finished
        const last = trackReplayData[trackReplayData.length - 1];
        trackDriverStates = last.states;
        updateTrackSidebar();
        drawTrack();
        stopReplay();
        return;
    }

    const current = trackReplayData[snapIdx];
    const next = snapIdx + 1 < trackReplayData.length ? trackReplayData[snapIdx + 1] : current;

    // Interpolate driver positions
    trackDriverStates = lerpStates(current.states, next.states, t);

    // Update lap badge (show current snapshot lap)
    const badge = document.getElementById('trackLapBadge');
    if (badge && currentTrack) badge.textContent = `LAP ${current.lap} / ${currentTrack.totalLaps}`;

    // Only update sidebar when snap changes to avoid DOM thrashing
    if (snapIdx !== trackReplayIndex) {
        trackReplayIndex = snapIdx;
        updateTrackSidebar();
    }

    drawTrack();
    trackAnimFrame = requestAnimationFrame(replayTick);
}

function stopReplay() {
    isReplaying = false;
    if (trackAnimFrame) cancelAnimationFrame(trackAnimFrame);
    trackAnimFrame = null;
    const btn = document.getElementById('btnReplay');
    if (btn) btn.textContent = '▶ REPLAY';
}

function enableReplayButton() {
    const btn = document.getElementById('btnReplay');
    if (btn) btn.disabled = false;
}
