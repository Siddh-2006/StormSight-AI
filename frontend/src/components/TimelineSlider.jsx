import { useEffect, useRef } from 'react';

export default function TimelineSlider({ leadTimesMin, leadIndex, onChange, playing, onTogglePlay }) {
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!playing || !leadTimesMin?.length) return undefined;
    intervalRef.current = setInterval(() => {
      onChange((leadIndex + 1) % leadTimesMin.length);
    }, 900);
    return () => clearInterval(intervalRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, leadIndex, leadTimesMin]);

  if (!leadTimesMin?.length) {
    return <div className="text-sm text-[var(--text-muted)]">Loading forecast timeline…</div>;
  }

  const current = leadTimesMin[leadIndex] ?? leadTimesMin[0];

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={onTogglePlay}
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#2a78d6] text-white hover:bg-[#256abf] transition-colors"
        title={playing ? 'Pause auto-advance' : 'Play forecast through lead times'}
        aria-label={playing ? 'Pause' : 'Play'}
      >
        {playing ? (
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
            <rect x="1" y="1" width="3.5" height="10" />
            <rect x="7" y="1" width="3.5" height="10" />
          </svg>
        ) : (
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
            <path d="M2 1 L11 6 L2 11 Z" />
          </svg>
        )}
      </button>

      <div className="flex-1">
        <input
          type="range"
          min={0}
          max={leadTimesMin.length - 1}
          step={1}
          value={leadIndex}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full accent-[#2a78d6]"
        />
        <div className="mt-1 flex justify-between text-[10px] text-[var(--text-muted)]">
          <span>+{leadTimesMin[0]} min</span>
          <span>+{leadTimesMin[leadTimesMin.length - 1]} min</span>
        </div>
      </div>

      <div className="w-24 shrink-0 text-right">
        <div className="text-lg font-semibold tabular-nums text-[var(--text-primary)]">+{current}</div>
        <div className="text-[10px] text-[var(--text-muted)]">minutes ahead</div>
      </div>
    </div>
  );
}
