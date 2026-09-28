// Sequential colour ramps for the map heatmap overlays, plus the status
// severity scale for the alert panel.
//
// These are domain-convention ramps (radar reflectivity blue->red, satellite
// IR inverted grayscale, lightning yellow->purple) rather than the dataviz
// skill's categorical identity palette — a heatmap overlay encodes a single
// continuous magnitude per layer, viewed one/few at a time with a labeled
// legend, not several series compared side-by-side. Each ramp is still a
// clean, monotonic, high-contrast sequential scale so it reads correctly on
// both light and dark basemaps. See frontend/NOTES.md.

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function mixStops(stops, t) {
  // stops: [[t0, [r,g,b]], [t1, [r,g,b]], ...] sorted ascending by t0
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const localT = t1 === t0 ? 0 : (clamped - t0) / (t1 - t0);
      return [
        Math.round(lerp(c0[0], c1[0], localT)),
        Math.round(lerp(c0[1], c1[1], localT)),
        Math.round(lerp(c0[2], c1[2], localT)),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

// Radar reflectivity: blue -> cyan -> yellow -> red (classic met convention,
// generalized to a clean blue->red sweep as requested).
const RADAR_STOPS = [
  [0.0, [12, 44, 132]],
  [0.25, [37, 106, 191]],
  [0.5, [56, 176, 130]],
  [0.72, [237, 200, 40]],
  [1.0, [211, 33, 44]],
];

// Satellite IR: inverted grayscale — cold cloud tops (low normalized value,
// per §2 units "K (180-300 normalized)") render bright/white, warm clear-sky
// areas render dark, matching how IR imagery is conventionally displayed.
const SATELLITE_STOPS = [
  [0.0, [255, 255, 255]],
  [0.5, [150, 150, 150]],
  [1.0, [20, 20, 20]],
];

// Lightning (observed density + predicted probability): yellow -> purple.
const LIGHTNING_STOPS = [
  [0.0, [255, 236, 140]],
  [0.35, [250, 178, 51]],
  [0.65, [217, 61, 91]],
  [1.0, [92, 36, 145]],
];

function makeRamp(stops, { alphaFloor = 0.06, alphaMax = 0.85, constantAlpha = null } = {}) {
  return function scale(value) {
    const v = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
    const [r, g, b] = mixStops(stops, v);
    const a =
      constantAlpha !== null
        ? constantAlpha
        : v < 0.04
        ? 0
        : alphaFloor + v * (alphaMax - alphaFloor);
    return [r, g, b, Math.round(a * 255)];
  };
}

export const COLOR_SCALES = {
  radar_dbz: makeRamp(RADAR_STOPS, { alphaFloor: 0.15, alphaMax: 0.85 }),
  satellite_ir_k: makeRamp(SATELLITE_STOPS, { alphaFloor: 0.12, alphaMax: 0.6 }),
  lightning_density: makeRamp(LIGHTNING_STOPS, { alphaFloor: 0.2, alphaMax: 0.9 }),
  lightning_probability: makeRamp(LIGHTNING_STOPS, { alphaFloor: 0.2, alphaMax: 0.9 }),
  radar_forecast: makeRamp(RADAR_STOPS, { alphaFloor: 0.15, alphaMax: 0.85 }),
};

// Legend gradient CSS (for the swatch bar in the layer control).
export function legendGradientCss(key) {
  const stopsByKey = {
    radar_dbz: RADAR_STOPS,
    radar_forecast: RADAR_STOPS,
    satellite_ir_k: SATELLITE_STOPS,
    lightning_density: LIGHTNING_STOPS,
    lightning_probability: LIGHTNING_STOPS,
  };
  const stops = stopsByKey[key] || RADAR_STOPS;
  const css = stops
    .map(([t, [r, g, b]]) => `rgb(${r},${g},${b}) ${Math.round(t * 100)}%`)
    .join(', ');
  return `linear-gradient(to right, ${css})`;
}

// Status severity scale for alerts — fixed roles, never reused for series
// identity (per dataviz skill: status palette is reserved).
export const SEVERITY_BANDS = [
  { max: 0.6, key: 'advisory', label: 'Advisory', color: '#2a78d6', textColor: '#ffffff' },
  { max: 0.75, key: 'watch', label: 'Watch', color: '#eda100', textColor: '#0b0b0b' },
  { max: 0.9, key: 'warning', label: 'Warning', color: '#eb6834', textColor: '#ffffff' },
  { max: Infinity, key: 'severe', label: 'Severe', color: '#d03b3b', textColor: '#ffffff' },
];

export function severityFor(probability) {
  return SEVERITY_BANDS.find((b) => probability <= b.max) || SEVERITY_BANDS[SEVERITY_BANDS.length - 1];
}

// Fixed categorical roles for the verification chart — colour follows model
// identity, consistent across every metric facet (dataviz skill slot 1/2).
export const MODEL_COLORS = {
  convlstm: { light: '#2a78d6', dark: '#3987e5', label: 'ConvLSTM (ML)' },
  baseline: { light: '#eb6834', dark: '#d95926', label: 'Optical-flow baseline' },
};
