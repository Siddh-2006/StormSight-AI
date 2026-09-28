# Frontend notes — assumptions & deviations

Built strictly against `docs/technical_flow.md` §3 (REST contract) and §5
(frontend contract). Nothing here changes the API shapes; these are the
judgment calls made where the contract left room, plus one openly-labeled
addition beyond the required scope.

## Assumptions

- **Which layers scrub with the timeline.** §5 item 1 lists exactly four map
  overlays: radar, satellite IR, *observed* lightning density, and
  *predicted* lightning probability. Read literally, only the last one is a
  forecast product. So radar/satellite/lightning-density overlays always show
  the current `/api/state` frame, and only the "predicted lightning
  probability" overlay (from `/api/nowcast`) changes as the timeline slider
  scrubs `lead_times_min`. This matches §6 step 3 ("no refetch needed per
  scrub") — all forecast frames from the one `/api/nowcast` call are already
  in memory.
- **Alert severity bands.** The contract says only "a simple severity color
  scale" with no cut points. Chose four bands against `probability`:
  Advisory (<0.6, blue), Watch (<0.75, yellow), Warning (<0.9, orange),
  Severe (>=0.9, red). Colors come from the house status palette
  (good/warning/serious/critical roles), each alert row also carries a text
  label so severity is never color-only.
- **Alert threshold control.** `/api/alerts?threshold=` is a query param with
  no UI requirement attached. Added a small select (0.3–0.8, default 0.5 —
  `DEFAULT_ALERT_THRESHOLD` in `src/config.js`) so the threshold is
  demoable live; refetches only `/api/alerts`, not the whole dashboard.
- **Verification chart layout.** §5 item 4 asks for CSI/POD/FAR/HSS vs.
  lead time with both models overlaid. Plotting all 4 metrics x 2 models as
  8 lines on one axis would be unreadable and violates the "few series per
  chart" rule we hold ourselves to for chart legibility. Instead: four small
  multiples (one per metric), each with exactly two lines — ConvLSTM (solid)
  vs. baseline (dashed) — sharing one fixed color-per-model legend across all
  four, so identity reads consistently. This is still literally "one chart
  capable of overlaying two series," just faceted by metric for readability.
- **Both models return the same `lead_times_min`.** Needed to plot both
  series against a shared x-axis; assumed since `horizon` is the only input
  that could change it, and both calls use the same `DEFAULT_HORIZON`.
- **Basemap tiles.** Used public CARTO Positron/Dark Matter tiles (no API
  key) switched by theme. Requires internet for the tile images themselves;
  all four data overlays (the actual contract requirement) render from the
  API/mock data and work regardless of tile availability.
- **Heatmap color ramps are domain-convention, not the categorical identity
  palette.** Radar = blue→red, satellite IR = inverted grayscale, lightning
  (both observed density and predicted probability) = yellow→purple, per the
  task's explicit request and standard met-ops convention. These are viewed
  one-or-few-at-a-time with a labeled legend (magnitude encoding), not
  compared side-by-side as categorical series, so the house categorical
  8-hue rule doesn't apply here; the UI chrome around them (text, panels,
  status colors, chart lines) does use the validated house palette.

## Deviation (additive only, clearly labeled)

- Added a 5th, **off-by-default** map layer toggle, "Predicted radar
  (forecast)", using the already-fetched `radar_forecast` array. It's tagged
  "extra" in the UI. Purely additive — doesn't change or replace any of the
  four required layers, uses only data the contract already returns.

## Mock mode

`src/mockApi.js` implements the exact response shapes from §3 with a small
synthetic storm generator (moving/decaying Gaussian blobs → radar/satellite/
lightning grids), so `npm run dev` is fully demoable with zero backend.
`src/api.js` tries the real `/api/*` endpoint first (through the Vite proxy
to `http://localhost:8000`) and falls back to `mockApi` on any fetch failure
or timeout, logging a loud styled `console.warn` the first time this happens
so it's obvious whether a demo is running live or in mock mode.
