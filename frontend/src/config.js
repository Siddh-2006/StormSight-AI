// Constants mirrored from docs/technical_flow.md §2-3 (the frozen API contract).
// Nothing here should be guessed — it's transcribed straight from the doc so the
// frontend never has to hardcode assumptions the backend could change.

export const API_BASE = '/api';

// §2 Shared grid convention — default demo bounding box (Vidarbha/Nagpur region).
export const GRID_BOUNDS = {
  lat_min: 19.5,
  lat_max: 22.5,
  lon_min: 77.5,
  lon_max: 80.5,
};

export const GRID = { h: 64, w: 64 };

// §2 Channels, in fixed order — index matters.
export const CHANNELS = [
  'radar_dbz',
  'satellite_ir_k',
  'lightning_density',
  'cape',
  'shear',
];

export const CHANNEL_LABELS = {
  radar_dbz: 'Radar reflectivity',
  satellite_ir_k: 'Satellite IR',
  lightning_density: 'Observed lightning density',
  cape: 'CAPE',
  shear: 'Wind shear',
};

// Demo pacing for "Play live demo" (§5 item 5 / §6 step 4): ~2s per step.
export const LIVE_DEMO_INTERVAL_MS = 2000;

export const DEFAULT_HORIZON = 12;
export const DEFAULT_ALERT_THRESHOLD = 0.5;
