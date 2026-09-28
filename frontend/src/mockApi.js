// Standalone mock backend — implements the exact JSON shapes from
// docs/technical_flow.md §3 so the UI can be built and demoed via
// `npm run dev` before the real FastAPI backend exists (or whenever it's
// unreachable). api.js falls back here automatically on fetch failure.
//
// This is a lightweight synthetic storm simulator: a few moving/decaying
// Gaussian "blobs" driving radar/satellite/lightning grids, mirroring the
// spirit (not the internals — those are the backend's) of
// docs/design.md §4's synthetic generator, just enough to produce
// physically-plausible-looking frames for UI development.

import { GRID, GRID_BOUNDS, CHANNELS } from './config';

const { h: H, w: W } = GRID;

function rand(seedState) {
  // xorshift32 — small deterministic PRNG so a given seed reproduces a scenario.
  seedState.v ^= seedState.v << 13;
  seedState.v ^= seedState.v >>> 17;
  seedState.v ^= seedState.v << 5;
  seedState.v >>>= 0;
  return seedState.v / 4294967296;
}

let seedState = { v: 42 };

function makeBlobs(numCells) {
  const blobs = [];
  for (let i = 0; i < numCells; i++) {
    blobs.push({
      x: 10 + rand(seedState) * (W - 20),
      y: 10 + rand(seedState) * (H - 20),
      vx: (rand(seedState) - 0.5) * 1.6,
      vy: (rand(seedState) - 0.5) * 1.6,
      radius: 6 + rand(seedState) * 6,
      intensity: 0.5 + rand(seedState) * 0.5,
      age: 0,
      lifespan: 40 + rand(seedState) * 60,
    });
  }
  return blobs;
}

// Static-ish environmental fields (CAPE / shear) — smooth random fields that
// barely change frame to frame per §2 ("held constant/interpolated").
function makeSmoothField(scale) {
  const field = [];
  for (let r = 0; r < H; r++) {
    const row = [];
    for (let c = 0; c < W; c++) {
      const v =
        0.5 +
        0.3 * Math.sin(r / 9 + scale) * Math.cos(c / 11 - scale) +
        0.15 * Math.sin((r + c) / 14);
      row.push(Math.max(0, Math.min(1, v)));
    }
    field.push(row);
  }
  return field;
}

const sim = {
  step: 0,
  blobs: makeBlobs(3),
  cape: makeSmoothField(0.3),
  shear: makeSmoothField(1.7),
  startTime: Date.now(),
};

function emptyGrid() {
  return Array.from({ length: H }, () => new Array(W).fill(0));
}

function renderBlobsToGrid(blobs, { radiusScale = 1, intensityFn = (b) => b.intensity } = {}) {
  const grid = emptyGrid();
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      let v = 0;
      for (const b of blobs) {
        const dx = c - b.x;
        const dy = r - b.y;
        const rad = b.radius * radiusScale;
        const d2 = (dx * dx + dy * dy) / (rad * rad);
        v += intensityFn(b) * Math.exp(-d2);
      }
      grid[r][c] = Math.max(0, Math.min(1, v));
    }
  }
  return grid;
}

function radarGrid(blobs) {
  return renderBlobsToGrid(blobs);
}

function satelliteGrid(blobs) {
  // Inversely correlated with reflectivity core (design.md §4): stronger
  // storms -> colder cloud tops -> LOWER normalized IR value (see colorScales
  // convention: 0 = cold). We approximate cold-top extent as a slightly wider
  // Gaussian than the radar core.
  const radar = renderBlobsToGrid(blobs, { radiusScale: 1.6 });
  return radar.map((row) => row.map((v) => Math.max(0, Math.min(1, 1 - v * 0.9))));
}

function lightningGrid(blobs) {
  // Flashes trigger stochastically once reflectivity core is strong (design.md §4).
  return renderBlobsToGrid(blobs, {
    radiusScale: 0.6,
    intensityFn: (b) => (b.intensity > 0.6 ? b.intensity : b.intensity * 0.15),
  });
}

function advanceBlobs(blobs) {
  return blobs
    .map((b) => {
      const age = b.age + 1;
      const lifeFrac = age / b.lifespan;
      // grow then decay lifecycle
      const lifecycle = lifeFrac < 0.3 ? lifeFrac / 0.3 : Math.max(0, 1 - (lifeFrac - 0.3) / 0.7);
      return {
        ...b,
        x: Math.max(2, Math.min(W - 2, b.x + b.vx)),
        y: Math.max(2, Math.min(H - 2, b.y + b.vy)),
        vx: b.vx + (rand(seedState) - 0.5) * 0.15,
        vy: b.vy + (rand(seedState) - 0.5) * 0.15,
        age,
        intensity: Math.max(0.05, Math.min(1, (0.4 + 0.6 * lifecycle) * (0.85 + rand(seedState) * 0.3))),
      };
    })
    .filter((b) => b.age < b.lifespan);
}

function maybeSpawn(blobs) {
  if (blobs.length < 2 || rand(seedState) < 0.08) {
    blobs.push(...makeBlobs(1));
  }
  return blobs;
}

function currentTimestamp() {
  return new Date(sim.startTime + sim.step * 10 * 60 * 1000).toISOString();
}

function buildStateResponse() {
  const frames = {
    radar_dbz: radarGrid(sim.blobs),
    satellite_ir_k: satelliteGrid(sim.blobs),
    lightning_density: lightningGrid(sim.blobs),
    cape: sim.cape,
    shear: sim.shear,
  };
  return {
    timestamp: currentTimestamp(),
    bounds: {
      lat_min: GRID_BOUNDS.lat_min,
      lat_max: GRID_BOUNDS.lat_max,
      lon_min: GRID_BOUNDS.lon_min,
      lon_max: GRID_BOUNDS.lon_max,
    },
    grid: { h: H, w: W },
    channels: [...CHANNELS],
    frames,
    units: {
      radar_dbz: 'dBZ (0-70 normalized)',
      satellite_ir_k: 'K (180-300 normalized)',
      cape: 'J/kg (0-5000 normalized)',
      shear: 'm/s (0-40 normalized)',
    },
  };
}

function projectBlobsForward(nSteps, jitter) {
  // Simple linear-velocity extrapolation of the current blob field — used to
  // synthesize plausible forecast frames without a real model.
  const frames = [];
  let cursor = sim.blobs.map((b) => ({ ...b }));
  for (let i = 0; i < nSteps; i++) {
    cursor = cursor.map((b) => ({
      ...b,
      x: b.x + b.vx * (1 + i * 0.05),
      y: b.y + b.vy * (1 + i * 0.05),
      intensity: Math.max(0, b.intensity * (1 - i * 0.04) + (rand(seedState) - 0.5) * jitter),
    }));
    frames.push(cursor);
  }
  return frames;
}

function scoresForModel(model, nSteps) {
  // ConvLSTM (ML) is tuned to score consistently higher than the optical-flow
  // baseline and decay more gracefully with lead time — this is the
  // "does ML beat the classical baseline" evidence the verification chart
  // exists to show (docs/technical_flow.md §5 item 4).
  const isMl = model === 'convlstm';
  const csi = [];
  const pod = [];
  const far = [];
  const hss = [];
  for (let i = 0; i < nSteps; i++) {
    const decay = i / Math.max(1, nSteps - 1);
    const noise = () => (rand(seedState) - 0.5) * 0.02;
    if (isMl) {
      csi.push(clamp01(0.72 - 0.28 * decay + noise()));
      pod.push(clamp01(0.82 - 0.25 * decay + noise()));
      far.push(clamp01(0.12 + 0.18 * decay + noise()));
      hss.push(clamp01(0.68 - 0.3 * decay + noise()));
    } else {
      csi.push(clamp01(0.55 - 0.3 * decay + noise()));
      pod.push(clamp01(0.62 - 0.32 * decay + noise()));
      far.push(clamp01(0.28 + 0.3 * decay + noise()));
      hss.push(clamp01(0.46 - 0.32 * decay + noise()));
    }
  }
  return { csi, pod, far, hss };
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function buildNowcastResponse({ horizon = 12, model = 'convlstm' } = {}) {
  const nSteps = Math.max(1, Math.min(12, Number(horizon) || 12));
  const jitter = model === 'baseline' ? 0.35 : 0.15;
  const forwardFrames = projectBlobsForward(nSteps, jitter);

  const radar_forecast = forwardFrames.map((blobs) => radarGrid(blobs));
  const lightning_probability = forwardFrames.map((blobs, i) => {
    const grid = lightningGrid(blobs);
    // widen/soften into a "probability" look and fade slightly with lead time
    const fade = model === 'baseline' ? 1 - i * 0.05 : 1 - i * 0.02;
    return grid.map((row) => row.map((v) => clamp01(Math.pow(v, 0.7) * fade)));
  });

  const lead_times_min = Array.from({ length: nSteps }, (_, i) => (i + 1) * 10);

  return {
    generated_at: currentTimestamp(),
    model,
    lead_times_min,
    radar_forecast,
    lightning_probability,
    scores: scoresForModel(model, nSteps),
  };
}

function buildAlertsResponse(threshold = 0.5) {
  const nowcast = buildNowcastResponse({ horizon: 6, model: 'convlstm' });
  const alerts = [];
  const latStep = (GRID_BOUNDS.lat_max - GRID_BOUNDS.lat_min) / H;
  const lonStep = (GRID_BOUNDS.lon_max - GRID_BOUNDS.lon_min) / W;

  nowcast.lightning_probability.forEach((grid, leadIdx) => {
    // coarse cell aggregation: scan on a stride so the "watch list" stays short
    for (let r = 0; r < H; r += 4) {
      for (let c = 0; c < W; c += 4) {
        let maxV = 0;
        for (let dr = 0; dr < 4 && r + dr < H; dr++) {
          for (let dc = 0; dc < 4 && c + dc < W; dc++) {
            maxV = Math.max(maxV, grid[r + dr][c + dc]);
          }
        }
        if (maxV >= threshold) {
          alerts.push({
            cell: [r, c],
            lat: Number((GRID_BOUNDS.lat_max - (r + 2) * latStep).toFixed(4)),
            lon: Number((GRID_BOUNDS.lon_min + (c + 2) * lonStep).toFixed(4)),
            probability: Number(maxV.toFixed(3)),
            lead_time_min: nowcast.lead_times_min[leadIdx],
          });
        }
      }
    }
  });

  // De-dupe by cell, keep the highest-probability occurrence, sort desc.
  const byCell = new Map();
  for (const a of alerts) {
    const key = `${a.cell[0]},${a.cell[1]}`;
    const existing = byCell.get(key);
    if (!existing || a.probability > existing.probability) byCell.set(key, a);
  }
  const deduped = [...byCell.values()].sort((a, b) => b.probability - a.probability);

  return { threshold: Number(threshold), alerts: deduped.slice(0, 25) };
}

export const mockApi = {
  getState() {
    return Promise.resolve(buildStateResponse());
  },
  getNowcast({ horizon = 12, model = 'convlstm' } = {}) {
    return Promise.resolve(buildNowcastResponse({ horizon, model }));
  },
  getAlerts({ threshold = 0.5 } = {}) {
    return Promise.resolve(buildAlertsResponse(threshold));
  },
  simulateStep() {
    sim.step += 1;
    sim.blobs = maybeSpawn(advanceBlobs(sim.blobs));
    return Promise.resolve(buildStateResponse());
  },
  simulateReset({ seed = 42, num_cells = 3 } = {}) {
    seedState = { v: seed || 42 };
    sim.step = 0;
    sim.blobs = makeBlobs(num_cells || 3);
    sim.cape = makeSmoothField(0.3);
    sim.shear = makeSmoothField(1.7);
    sim.startTime = Date.now();
    return Promise.resolve(buildStateResponse());
  },
};
