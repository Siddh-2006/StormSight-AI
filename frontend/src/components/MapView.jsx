import { useEffect, useMemo, useState } from 'react';
import { MapContainer, TileLayer, ImageOverlay, Rectangle, CircleMarker, Tooltip, useMap } from 'react-leaflet';
import { GRID_BOUNDS } from '../config';
import { COLOR_SCALES, legendGradientCss, severityFor } from '../lib/colorScales';
import { gridToDataUrl } from '../lib/heatmap';

const IMAGE_BOUNDS = [
  [GRID_BOUNDS.lat_min, GRID_BOUNDS.lon_min],
  [GRID_BOUNDS.lat_max, GRID_BOUNDS.lon_max],
];

const RECT_BOUNDS = IMAGE_BOUNDS;

const CENTER = [
  (GRID_BOUNDS.lat_min + GRID_BOUNDS.lat_max) / 2,
  (GRID_BOUNDS.lon_min + GRID_BOUNDS.lon_max) / 2,
];

// CARTO's raster basemaps now require an API key (free tier: 5M tiles/month
// non-commercial) or they render with an "API key required" watermark. Key
// goes in frontend/.env.local (gitignored — see .env.example), read via
// Vite's import.meta.env. Falls back to the old unauthenticated/watermarked
// URL if no key is configured, so the app still works without one.
const CARTO_API_KEY = import.meta.env.VITE_CARTO_API_KEY;
const TILE_URLS = CARTO_API_KEY
  ? {
      light: `https://basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}{r}.png?key=${CARTO_API_KEY}`,
      dark: `https://basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}{r}.png?key=${CARTO_API_KEY}`,
    }
  : {
      light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
      dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    };

const LAYER_DEFS = [
  {
    key: 'radar_dbz',
    label: 'Radar reflectivity',
    source: 'observed',
    scale: 'radar_dbz',
    defaultOn: true,
  },
  {
    key: 'satellite_ir_k',
    label: 'Satellite IR',
    source: 'observed',
    scale: 'satellite_ir_k',
    defaultOn: false,
  },
  {
    key: 'lightning_density',
    label: 'Observed lightning density',
    source: 'observed',
    scale: 'lightning_density',
    defaultOn: false,
  },
  {
    key: 'lightning_probability',
    label: 'Predicted lightning probability',
    source: 'forecast',
    scale: 'lightning_probability',
    defaultOn: true,
  },
  {
    key: 'radar_forecast',
    label: 'Predicted radar (forecast)',
    source: 'forecast',
    scale: 'radar_forecast',
    defaultOn: false,
    bonus: true,
  },
];

// Leaflet sizes its internal tile grid/canvas once at mount. When the
// surrounding flex/grid layout changes size afterward (data finishes
// loading, window resizes, the verification panel below reflows), Leaflet
// doesn't know unless told — the map then renders at its stale size and
// looks clipped/misaligned inside the new container. Watch the actual DOM
// size and call invalidateSize() whenever it changes.
function MapResizeHandler() {
  const map = useMap();
  useEffect(() => {
    const container = map.getContainer();
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(container);
    // Also catch the very first post-mount layout settle (e.g. data load
    // shifting the grid row heights) with a couple of delayed checks.
    const timers = [50, 250, 800].map((ms) => setTimeout(() => map.invalidateSize(), ms));
    return () => {
      observer.disconnect();
      timers.forEach(clearTimeout);
    };
  }, [map]);
  return null;
}

function useOverlayUrl(grid, scaleKey) {
  return useMemo(() => {
    if (!grid) return null;
    return gridToDataUrl(grid, COLOR_SCALES[scaleKey]);
  }, [grid, scaleKey]);
}

function Overlay({ def, observed, forecast, leadIndex, enabled, opacity }) {
  const grid =
    def.source === 'observed'
      ? observed?.frames?.[def.key]
      : forecast?.[def.key === 'lightning_probability' ? 'lightning_probability' : 'radar_forecast']?.[leadIndex];

  const url = useOverlayUrl(enabled ? grid : null, def.scale);
  if (!enabled || !url) return null;
  return <ImageOverlay url={url} bounds={IMAGE_BOUNDS} opacity={opacity} className="pointer-events-none" />;
}

export default function MapView({ observed, forecast, leadIndex, theme, alerts }) {
  const [enabledLayers, setEnabledLayers] = useState(() =>
    Object.fromEntries(LAYER_DEFS.map((d) => [d.key, d.defaultOn]))
  );
  const [opacity, setOpacity] = useState(0.75);

  const toggle = (key) => setEnabledLayers((prev) => ({ ...prev, [key]: !prev[key] }));

  const activeLegends = LAYER_DEFS.filter((d) => enabledLayers[d.key]);

  return (
    <div className="flex h-full min-h-0 gap-3">
      <div className="relative flex-1 min-w-0 overflow-hidden rounded-lg border border-black/10 dark:border-white/10">
        <MapContainer
          bounds={IMAGE_BOUNDS}
          center={CENTER}
          zoom={8}
          scrollWheelZoom
          className="h-full w-full"
          style={{ background: 'var(--surface-1)' }}
        >
          <MapResizeHandler />
          <TileLayer
            url={TILE_URLS[theme] || TILE_URLS.light}
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a> &copy; OpenStreetMap contributors'
          />
          {LAYER_DEFS.map((def) => (
            <Overlay
              key={def.key}
              def={def}
              observed={observed}
              forecast={forecast}
              leadIndex={leadIndex}
              enabled={!!enabledLayers[def.key]}
              opacity={opacity}
            />
          ))}
          <Rectangle
            bounds={RECT_BOUNDS}
            pathOptions={{ color: '#898781', weight: 1, dashArray: '4 4', fill: false }}
          />
          {alerts?.map((a) => {
            const sev = severityFor(a.probability);
            return (
              <CircleMarker
                key={`${a.cell[0]}-${a.cell[1]}`}
                center={[a.lat, a.lon]}
                radius={5}
                pathOptions={{ color: sev.color, fillColor: sev.color, fillOpacity: 0.85, weight: 1.5 }}
              >
                <Tooltip direction="top" offset={[0, -4]}>
                  {sev.label} · p={a.probability.toFixed(2)} · +{a.lead_time_min}min
                </Tooltip>
              </CircleMarker>
            );
          })}
        </MapContainer>

        <div className="pointer-events-none absolute left-2 top-2 rounded bg-[var(--surface-1)]/90 px-2 py-1 text-[11px] text-[var(--text-secondary)] shadow">
          Vidarbha region · {GRID_BOUNDS.lat_min}–{GRID_BOUNDS.lat_max}°N, {GRID_BOUNDS.lon_min}–
          {GRID_BOUNDS.lon_max}°E · 64×64 grid
        </div>
      </div>

      <div className="w-56 shrink-0 overflow-y-auto rounded-lg border border-black/10 dark:border-white/10 bg-[var(--surface-1)] p-3">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
          Layers
        </h3>
        <div className="space-y-2">
          {LAYER_DEFS.map((def) => (
            <label key={def.key} className="flex cursor-pointer items-center gap-2 text-sm text-[var(--text-primary)]">
              <input
                type="checkbox"
                checked={!!enabledLayers[def.key]}
                onChange={() => toggle(def.key)}
                className="h-3.5 w-3.5 accent-[#2a78d6]"
              />
              <span>{def.label}</span>
              {def.bonus && (
                <span className="ml-auto rounded bg-black/5 dark:bg-white/10 px-1 text-[9px] text-[var(--text-muted)]">
                  extra
                </span>
              )}
            </label>
          ))}
        </div>

        <div className="mt-4">
          <label className="mb-1 flex justify-between text-xs text-[var(--text-secondary)]">
            <span>Overlay opacity</span>
            <span>{Math.round(opacity * 100)}%</span>
          </label>
          <input
            type="range"
            min={0.2}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="w-full accent-[#2a78d6]"
          />
        </div>

        {activeLegends.length > 0 && (
          <div className="mt-4 space-y-3 border-t border-black/10 dark:border-white/10 pt-3">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
              Legend
            </h3>
            {activeLegends.map((def) => (
              <div key={def.key}>
                <div className="mb-1 text-[11px] text-[var(--text-secondary)]">{def.label}</div>
                <div
                  className="h-2 w-full rounded-full"
                  style={{ background: legendGradientCss(def.scale) }}
                />
                <div className="mt-0.5 flex justify-between text-[10px] text-[var(--text-muted)]">
                  <span>0.0</span>
                  <span>1.0 (normalized)</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
