import { useEffect, useState } from 'react';

function getInitialTheme() {
  try {
    const saved = localStorage.getItem('nowcast-theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    /* localStorage unavailable (private window etc.) — fall through */
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function useTheme() {
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
    try {
      localStorage.setItem('nowcast-theme', theme);
    } catch {
      /* per-viewer convenience only */
    }
  }, [theme]);

  const toggle = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  return { theme, toggle };
}
