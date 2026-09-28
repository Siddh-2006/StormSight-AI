export default function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      onClick={onToggle}
      className="rounded-md border border-black/10 dark:border-white/10 px-3 py-1.5 text-xs font-medium
                 text-[var(--text-secondary)] hover:text-[var(--text-primary)]
                 bg-[var(--surface-1)] hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
      title="Toggle light / dark theme"
    >
      {theme === 'dark' ? 'Dark' : 'Light'} mode
    </button>
  );
}
