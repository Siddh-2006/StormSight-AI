// Draws a 64x64 grid of normalized [0,1] floats to an offscreen canvas and
// returns a PNG data URL, for use as a react-leaflet <ImageOverlay>.

/**
 * @param {number[][]} grid - [row][col] values in [0,1]
 * @param {(v: number) => [number, number, number, number]} colorScale
 * @returns {string} data URL
 */
export function gridToDataUrl(grid, colorScale) {
  const h = grid.length;
  const w = grid[0]?.length ?? 0;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(w, h);

  for (let row = 0; row < h; row++) {
    for (let col = 0; col < w; col++) {
      const value = grid[row][col];
      const [r, g, b, a] = colorScale(value);
      const idx = (row * w + col) * 4;
      imageData.data[idx] = r;
      imageData.data[idx + 1] = g;
      imageData.data[idx + 2] = b;
      imageData.data[idx + 3] = a;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

// Small memoization so repeated renders (e.g. re-render on unrelated state
// change) don't re-walk the grid when the same frame + scale are reused.
const cache = new Map();
const MAX_CACHE = 64;

export function gridToDataUrlCached(grid, colorScale, cacheKey) {
  if (cacheKey && cache.has(cacheKey)) return cache.get(cacheKey);
  const url = gridToDataUrl(grid, colorScale);
  if (cacheKey) {
    if (cache.size >= MAX_CACHE) {
      const firstKey = cache.keys().next().value;
      cache.delete(firstKey);
    }
    cache.set(cacheKey, url);
  }
  return url;
}
