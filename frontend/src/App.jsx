import { useCallback, useEffect, useRef, useState } from 'react';
import MapView from './components/MapView';
import TimelineSlider from './components/TimelineSlider';
import AlertPanel from './components/AlertPanel';
import VerificationChart from './components/VerificationChart';
import ThemeToggle from './components/ThemeToggle';
import { useTheme } from './hooks/useTheme';
import { getState, getNowcast, getAlerts, simulateStep } from './api';
import { DEFAULT_HORIZON, DEFAULT_ALERT_THRESHOLD, LIVE_DEMO_INTERVAL_MS } from './config';

export default function App() {
  const { theme, toggle } = useTheme();

  const [observed, setObserved] = useState(null);
  const [nowcastConvlstm, setNowcastConvlstm] = useState(null);
  const [nowcastBaseline, setNowcastBaseline] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [threshold, setThreshold] = useState(DEFAULT_ALERT_THRESHOLD);

  const [leadIndex, setLeadIndex] = useState(0);
  const [timelinePlaying, setTimelinePlaying] = useState(false);
  const [liveDemoOn, setLiveDemoOn] = useState(false);

  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [error, setError] = useState(null);

  const refreshAll = useCallback(async () => {
    try {
      const [stateRes, convRes, baseRes, alertsRes] = await Promise.all([
        getState(),
        getNowcast({ horizon: DEFAULT_HORIZON, model: 'convlstm' }),
        getNowcast({ horizon: DEFAULT_HORIZON, model: 'baseline' }),
        getAlerts({ threshold }),
      ]);
      setObserved(stateRes);
      setNowcastConvlstm(convRes);
      setNowcastBaseline(baseRes);
      setAlerts(alertsRes.alerts || []);
      setLeadIndex((prev) => Math.min(prev, (convRes.lead_times_min?.length || 1) - 1));
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(String(err?.message || err));
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threshold]);

  // Initial load.
  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refetch alerts (only) when the threshold selector changes, without
  // disturbing the map/timeline state — cheap and keeps the panel responsive.
  const isFirstThreshold = useRef(true);
  useEffect(() => {
    if (isFirstThreshold.current) {
      isFirstThreshold.current = false;
      return;
    }
    getAlerts({ threshold }).then((res) => setAlerts(res.alerts || []));
  }, [threshold]);

  // "Play live demo" (§5 item 5 / §6 step 4): advance the synthetic backend
  // by one step every ~2s, then re-pull state + nowcast + alerts.
  useEffect(() => {
    if (!liveDemoOn) return undefined;
    const id = setInterval(async () => {
      try {
        await simulateStep();
        await refreshAll();
      } catch (err) {
        setError(String(err?.message || err));
      }
    }, LIVE_DEMO_INTERVAL_MS);
    return () => clearInterval(id);
  }, [liveDemoOn, refreshAll]);

  const leadTimesMin = nowcastConvlstm?.lead_times_min || [];

  return (
    <div className="flex h-screen flex-col bg-[var(--surface-2)] text-[var(--text-primary)]">
      <header className="flex items-center justify-between border-b border-black/10 dark:border-white/10 bg-[var(--surface-1)] px-4 py-2.5">
        <div>
          <h1 className="text-sm font-bold tracking-tight">
            Thunderstorm &amp; Lightning Nowcasting
          </h1>
          <p className="text-[11px] text-[var(--text-muted)]">
            SIH 26072 · IMD / Ministry of Earth Sciences · prototype dashboard
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-[11px] text-[var(--text-muted)]">
              updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={() => setLiveDemoOn((v) => !v)}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              liveDemoOn
                ? 'bg-[#d03b3b] text-white hover:bg-[#b93333]'
                : 'bg-[#2a78d6] text-white hover:bg-[#256abf]'
            }`}
          >
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${liveDemoOn ? 'bg-white animate-pulse' : 'bg-white/70'}`}
            />
            {liveDemoOn ? 'Stop live demo' : 'Play live demo'}
          </button>
          <ThemeToggle theme={theme} onToggle={toggle} />
        </div>
      </header>

      {error && (
        <div className="bg-[#eda100]/15 px-4 py-1 text-xs text-[#8a6400] dark:text-[#eda100]">
          {error} — see console; falling back to mock data where applicable.
        </div>
      )}

      <main className="grid min-h-0 flex-1 grid-cols-[1fr_320px] grid-rows-[minmax(420px,1fr)_minmax(0,300px)] gap-3 overflow-hidden p-3">
        <section className="min-h-0 row-span-1">
          <MapView
            observed={observed}
            forecast={{
              lightning_probability: nowcastConvlstm?.lightning_probability,
              radar_forecast: nowcastConvlstm?.radar_forecast,
            }}
            leadIndex={leadIndex}
            theme={theme}
            alerts={alerts}
          />
        </section>

        <aside className="row-span-2 min-h-0 overflow-hidden rounded-lg border border-black/10 dark:border-white/10 bg-[var(--surface-1)] p-3">
          <AlertPanel
            alerts={alerts}
            threshold={threshold}
            onThresholdChange={setThreshold}
            loading={loading && alerts.length === 0}
          />
        </aside>

        <section className="col-start-1 row-start-2 space-y-3 overflow-y-auto rounded-lg border border-black/10 dark:border-white/10 bg-[var(--surface-1)] p-3">
          <div>
            <h2 className="mb-2 text-sm font-semibold">Forecast timeline</h2>
            <TimelineSlider
              leadTimesMin={leadTimesMin}
              leadIndex={leadIndex}
              onChange={setLeadIndex}
              playing={timelinePlaying}
              onTogglePlay={() => setTimelinePlaying((v) => !v)}
            />
          </div>
          <VerificationChart
            convlstm={nowcastConvlstm}
            baseline={nowcastBaseline}
            theme={theme}
            loading={loading}
          />
        </section>
      </main>
    </div>
  );
}
