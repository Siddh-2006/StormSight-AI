import { severityFor } from '../lib/colorScales';

const THRESHOLD_OPTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

export default function AlertPanel({ alerts, threshold, onThresholdChange, loading }) {
  const sorted = [...(alerts || [])].sort((a, b) => b.probability - a.probability);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 border-b border-black/10 dark:border-white/10 pb-2">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">Lightning risk alerts</h2>
        <label className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
          threshold
          <select
            value={threshold}
            onChange={(e) => onThresholdChange(Number(e.target.value))}
            className="rounded border border-black/10 dark:border-white/10 bg-[var(--surface-1)] px-1.5 py-0.5 text-xs text-[var(--text-primary)]"
          >
            {THRESHOLD_OPTIONS.map((t) => (
              <option key={t} value={t}>
                {t.toFixed(1)}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="mt-2 flex-1 overflow-y-auto">
        {loading && <p className="p-2 text-xs text-[var(--text-muted)]">Loading alerts…</p>}
        {!loading && sorted.length === 0 && (
          <p className="p-2 text-xs text-[var(--text-muted)]">
            No cells above p ≥ {threshold.toFixed(1)} right now.
          </p>
        )}
        <ul className="divide-y divide-black/5 dark:divide-white/5">
          {sorted.map((a, i) => {
            const sev = severityFor(a.probability);
            return (
              <li key={`${a.cell[0]}-${a.cell[1]}-${i}`} className="flex items-center gap-2 py-1.5">
                <span
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] font-bold"
                  style={{ background: sev.color, color: sev.textColor }}
                  title={sev.label}
                >
                  {sev.label[0]}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-xs font-medium text-[var(--text-primary)]">{sev.label}</span>
                    <span className="tabular-nums text-xs font-semibold text-[var(--text-primary)]">
                      {(a.probability * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="truncate text-[10px] text-[var(--text-muted)]">
                    {a.lat.toFixed(2)}°N, {a.lon.toFixed(2)}°E · cell [{a.cell[0]},{a.cell[1]}] · +
                    {a.lead_time_min} min
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 border-t border-black/10 dark:border-white/10 pt-2 text-[10px] text-[var(--text-muted)]">
        {[0.3, 0.65, 0.8, 0.95].map((p) => {
          const sev = severityFor(p);
          return (
            <span key={sev.key} className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: sev.color }} />
              {sev.label}
            </span>
          );
        })}
      </div>
    </div>
  );
}
