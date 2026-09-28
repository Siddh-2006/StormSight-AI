// Thin fetch wrappers against docs/technical_flow.md §3. Every call falls
// back to src/mockApi.js on network failure (backend not running / not built
// yet), so `npm run dev` is always demoable standalone. Once the real FastAPI
// backend is up at :8000, Vite's dev proxy (vite.config.js) forwards /api/*
// to it and these same calls hit the real thing — no code change needed.

import { API_BASE } from './config';
import { mockApi } from './mockApi';

let mockWarned = false;
function warnMock(reason) {
  if (mockWarned) return;
  mockWarned = true;
  // Loud, deliberate console signal per task spec — this must be obvious
  // when eyeballing the demo that it's running without a live backend.
  console.warn(
    '%c[mock mode] backend unreachable — serving synthetic mock data from src/mockApi.js',
    'color: #eda100; font-weight: bold;',
    reason
  );
}

async function getJson(path, { timeoutMs = 4000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function postJson(path, body, { timeoutMs = 4000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

export async function getState() {
  try {
    return await getJson('/state');
  } catch (err) {
    warnMock(err);
    return mockApi.getState();
  }
}

export async function getNowcast({ horizon = 12, model = 'convlstm' } = {}) {
  try {
    return await getJson(`/nowcast?horizon=${horizon}&model=${model}`);
  } catch (err) {
    warnMock(err);
    return mockApi.getNowcast({ horizon, model });
  }
}

export async function getAlerts({ threshold = 0.5 } = {}) {
  try {
    return await getJson(`/alerts?threshold=${threshold}`);
  } catch (err) {
    warnMock(err);
    return mockApi.getAlerts({ threshold });
  }
}

export async function simulateStep() {
  try {
    return await postJson('/simulate/step');
  } catch (err) {
    warnMock(err);
    return mockApi.simulateStep();
  }
}

export async function simulateReset(body) {
  try {
    return await postJson('/simulate/reset', body);
  } catch (err) {
    warnMock(err);
    return mockApi.simulateReset(body);
  }
}
