import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { MODEL_COLORS } from '../lib/colorScales';

// Chart chrome tokens, per dataviz skill palette.md, picked explicitly per
// theme rather than via CSS vars (Recharts renders raw SVG attributes, so we
// resolve the values in JS instead of relying on var() cascade in <svg>).
const CHROME = {
  light: { grid: '#e1e0d9', axis: '#898781', ink: '#52514e', surface: '#fcfcfb' },
  dark: { grid: '#2c2c2a', axis: '#898781', ink: '#c3c2b7', surface: '#1a1a19' },
};

const METRICS = [
  { key: 'csi', label: 'CSI', hint: 'Critical Success Index — higher is better' },
  { key: 'pod', label: 'POD', hint: 'Probability of Detection — higher is better' },
  { key: 'far', label: 'FAR', hint: 'False Alarm Ratio — lower is better' },
  { key: 'hss', label: 'HSS', hint: 'Heidke Skill Score — higher is better' },
];

function buildSeries(leadTimes, convlstmScores, baselineScores, metricKey) {
  return leadTimes.map((lt, i) => ({
    leadTime: lt,
    convlstm: convlstmScores?.[metricKey]?.[i] ?? null,
    baseline: baselineScores?.[metricKey]?.[i] ?? null,
  }));
}

function MetricMiniChart({ metric, data, theme }) {
  const c = CHROME[theme] || CHROME.light;
  return (
    <div className="rounded-lg border border-black/10 dark:border-white/10 bg-[var(--surface-1)] p-2">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-xs font-semibold text-[var(--text-primary)]">{metric.label}</span>
        <span className="text-[10px] text-[var(--text-muted)]">{metric.hint}</span>
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: -16 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis
            dataKey="leadTime"
            tick={{ fontSize: 10, fill: c.axis }}
            stroke={c.axis}
            tickLine={false}
            axisLine={{ stroke: c.grid }}
            label={{ value: 'lead time (min)', position: 'insideBottom', offset: -2, fontSize: 9, fill: c.axis }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 10, fill: c.axis }}
            stroke={c.axis}
            tickLine={false}
            axisLine={false}
            width={30}
          />
          <Tooltip
            contentStyle={{
              background: c.surface,
              border: `1px solid ${c.grid}`,
              fontSize: 11,
              color: c.ink,
            }}
            formatter={(value, name) => [Number(value).toFixed(3), MODEL_COLORS[name]?.label || name]}
            labelFormatter={(v) => `+${v} min`}
          />
          <Line
            type="monotone"
            dataKey="convlstm"
            name="convlstm"
            stroke={MODEL_COLORS.convlstm[theme] || MODEL_COLORS.convlstm.light}
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="baseline"
            name="baseline"
            stroke={MODEL_COLORS.baseline[theme] || MODEL_COLORS.baseline.light}
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function VerificationChart({ convlstm, baseline, theme, loading }) {
  if (loading || !convlstm?.lead_times_min) {
    return <p className="text-xs text-[var(--text-muted)]">Loading verification scores…</p>;
  }

  const leadTimes = convlstm.lead_times_min;

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">
          Verification — ML vs. classical baseline
        </h2>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1.5 text-[var(--text-secondary)]">
            <span
              className="inline-block h-0.5 w-4"
              style={{ background: MODEL_COLORS.convlstm[theme] || MODEL_COLORS.convlstm.light }}
            />
            {MODEL_COLORS.convlstm.label}
          </span>
          <span className="flex items-center gap-1.5 text-[var(--text-secondary)]">
            <svg width="16" height="2">
              <line
                x1="0"
                y1="1"
                x2="16"
                y2="1"
                stroke={MODEL_COLORS.baseline[theme] || MODEL_COLORS.baseline.light}
                strokeWidth="2"
                strokeDasharray="5 3"
              />
            </svg>
            {MODEL_COLORS.baseline.label}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {METRICS.map((m) => (
          <MetricMiniChart
            key={m.key}
            metric={m}
            theme={theme}
            data={buildSeries(leadTimes, convlstm.scores, baseline?.scores, m.key)}
          />
        ))}
      </div>
    </div>
  );
}
