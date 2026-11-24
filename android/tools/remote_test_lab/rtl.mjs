#!/usr/bin/env node
// Samsung Remote Test Lab from the command line: rent a phone, open its web viewer in a headless Chrome
// (so it keeps running while the Mac is locked), and connect it to adb through the rdb bridge.
//
//   node rtl.mjs login                         # one time: sign in to Samsung in the tool's own Chrome profile
//   node rtl.mjs devices [--model "Tab S9"] [--free]     # any of ~1,300 Galaxy phones, tablets, watches, TVs
//   node rtl.mjs rent [--model "Galaxy A56"] [--device SM-S948U-US01 | 13819] [--minutes 30] [--location Korea] [--os 17]
//                                              # --model matches part of the name; no filter = any free device
//   node rtl.mjs status                        # reservations, minutes left, adb state
//   node rtl.mjs viewer                        # (re)open the viewer for the current reservation
//   node rtl.mjs release                       # return the phone and stop the headless Chrome
//
// Needs Google Chrome, Node 22+ (built-in WebSocket), adb, and Samsung's rdb bridge running
// (~/Downloads/rdb/rdb by default; set RDB=/path/to/rdb). Nothing is downloaded.
import { spawn, execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const HOME = join(homedir(), '.llmhub-rtl');
const PROFILE = join(HOME, 'chrome-profile');
const STATE = join(HOME, 'state.json');
const PORT = Number(process.env.RTL_CDP_PORT || 9333);
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const RDB = process.env.RDB || join(homedir(), 'Downloads/rdb/rdb');
const SITE = 'https://developer.samsung.com/remotetestlab/';
const API = SITE + 'rtl/api/v1/';
mkdirSync(PROFILE, { recursive: true });

const args = process.argv.slice(2);
const cmd = args[0];
const opt = (name, def) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : def; };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const log = (...a) => console.error('[rtl]', ...a);
const state = () => (existsSync(STATE) ? JSON.parse(readFileSync(STATE, 'utf8')) : {});
const saveState = s => writeFileSync(STATE, JSON.stringify(s, null, 2));

// ---------- Chrome + CDP ----------
async function cdpUp() { try { return await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); } catch { return null; } }

async function startChrome(headless) {
  if (await cdpUp()) return;
  const flags = [`--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`, '--no-first-run', '--no-default-browser-check',
    '--disable-popup-blocking', '--autoplay-policy=no-user-gesture-required', '--window-size=1500,950'];
  if (headless) flags.push('--headless=new');
  const p = spawn(CHROME, [...flags, SITE + 'devices'], { detached: true, stdio: 'ignore' });
  p.unref();
  for (let i = 0; i < 60 && !(await cdpUp()); i++) await sleep(250);
  if (!(await cdpUp())) throw new Error('Chrome did not start with remote debugging');
}

async function stopChrome() {
  const v = await cdpUp(); if (!v) return;
  const ws = await connect(v.webSocketDebuggerUrl);
  try { await ws.send('Browser.close'); } catch {}
}

function connect(url) {
  return new Promise((resolve, reject) => {
    const sock = new WebSocket(url); let id = 0; const pending = new Map(); const listeners = [];
    sock.onopen = () => resolve({
      send: (method, params = {}, sessionId) => new Promise((res, rej) => {
        const m = { id: ++id, method, params }; if (sessionId) m.sessionId = sessionId;
        pending.set(m.id, { res, rej }); sock.send(JSON.stringify(m));
      }),
      on: f => listeners.push(f),
      close: () => sock.close(),
    });
    sock.onerror = e => reject(e);
    sock.onmessage = ev => {
      const m = JSON.parse(ev.data);
      if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.rej(new Error(m.error.message)) : p.res(m.result); }
      else listeners.forEach(f => f(m));
    };
  });
}

async function browser() { const v = await cdpUp(); if (!v) throw new Error('Chrome is not running'); return connect(v.webSocketDebuggerUrl); }

async function pages(b) { return (await b.send('Target.getTargets')).targetInfos.filter(t => t.type === 'page'); }

async function attach(b, targetId) {
  const { sessionId } = await b.send('Target.attachToTarget', { targetId, flatten: true });
  await b.send('Runtime.enable', {}, sessionId);
  await b.send('Page.enable', {}, sessionId);
  return sessionId;
}

async function evaluate(b, sessionId, expression) {
  const r = await b.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture: true }, sessionId);
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
}

// The site page (devices list) we run API calls from: same origin, so the Samsung session cookies apply.
async function sitePage(b) {
  let p = (await pages(b)).find(t => t.url.startsWith(SITE) && !t.url.includes('/webclient'));
  if (!p) { const { targetId } = await b.send('Target.createTarget', { url: SITE + 'devices' }); p = { targetId }; }
  const s = await attach(b, p.targetId);
  for (let i = 0; i < 80; i++) {
    const ok = await evaluate(b, s, `location.href.startsWith(${JSON.stringify(SITE)}) && document.readyState === 'complete'`).catch(() => false);
    if (ok) break;
    await sleep(250);
  }
  return s;
}

const api = (b, s, path, method = 'GET', params) => evaluate(b, s, `(async () => {
  const r = await fetch(${JSON.stringify(API + path)} + ${params ? `'?' + new URLSearchParams(${JSON.stringify(params)})` : "''"}, { method: ${JSON.stringify(method)}, headers: { Accept: 'application/json, text/plain, */*' } });
  const t = await r.text(); let body; try { body = JSON.parse(t); } catch { body = t; }
  return { status: r.status, body };
})()`);

async function me(b, s) { const r = await api(b, s, 'users/me'); return r.status === 200 ? r.body : null; }

async function allDevices(b, s) {
  let out = [], page = 1, total = 1;
  do { const r = await api(b, s, `devices?pageNum=${page}&pageSize=100&clientType=web`); out = out.concat(r.body.content || []); total = r.body.totalPage || 1; page++; } while (page <= total && page < 40);
  // workStatus 1 = online; `waiting` set = someone is using it; workStatus 2 = offline.
  return out.map(d => ({ id: d.deviceId, product: d.productName, name: d.deviceName, location: d.location, os: d.osVersionName,
    oneUi: d.oneUiVer, free: d.workStatus === 1 && (d.waiting == null), busySeconds: d.waiting, offline: d.workStatus !== 1 }));
}

// ---------- viewer ----------
// The viewer page reads its session from cookies set by devices/webclient/{start,restart} and needs window.opener
// (the site opens it with window.open). A click on an injected button gives window.open a real user gesture.
async function openViewer(b, s, reservationId) {
  for (const p of await pages(b)) if (p.url.includes('/webclient')) await b.send('Target.closeTarget', { targetId: p.targetId });
  if (reservationId) {
    const r = await api(b, s, 'devices/webclient/restart', 'POST', { reservationId, _ts: Date.now() });
    if (r.status !== 200) throw new Error('viewer restart failed: ' + JSON.stringify(r.body));
  }
  await evaluate(b, s, `(() => { document.getElementById('rtl-open')?.remove(); const x = document.createElement('button'); x.id = 'rtl-open';
    x.style.cssText = 'position:fixed;top:0;left:0;width:60px;height:60px;z-index:2147483647'; x.onclick = () => { window.open(${JSON.stringify(SITE + 'webclient')}, 'webclient'); x.remove(); };
    document.body.appendChild(x); return true; })()`);
  for (const type of ['mousePressed', 'mouseReleased']) await b.send('Input.dispatchMouseEvent', { type, x: 30, y: 30, button: 'left', clickCount: 1 }, s);
  let viewer;
  for (let i = 0; i < 40 && !viewer; i++) { await sleep(250); viewer = (await pages(b)).find(t => t.url.includes('/webclient')); }
  if (!viewer) throw new Error('viewer window did not open');
  const vs = await attach(b, viewer.targetId);
  // Headless pages are visible; also pin focus so the viewer never pauses.
  await b.send('Emulation.setFocusEmulationEnabled', { enabled: true }, vs).catch(() => {});
  return { targetId: viewer.targetId, session: vs };
}

async function viewerText(b, vs) { return evaluate(b, vs, `document.body ? document.body.innerText.slice(0, 3000) : ''`); }

// ---------- adb ----------
function rdbPort() {
  try {
    const out = execFileSync('lsof', ['-nP', '-iTCP', '-sTCP:LISTEN'], { encoding: 'utf8' });
    const ports = out.split('\n').filter(l => /^rdb\b/.test(l)).map(l => Number(l.match(/:(\d+) \(LISTEN\)/)?.[1])).filter(p => p && p !== 8888);
    return ports[0] || null;
  } catch { return null; }
}
function ensureRdb() {
  if (rdbPort()) return;
  if (!existsSync(RDB)) throw new Error(`rdb not found at ${RDB} (set RDB=/path/to/rdb)`);
  spawn(RDB, [], { detached: true, stdio: 'ignore' }).unref();
}
const adb = (...a) => { try { return execFileSync('adb', a, { encoding: 'utf8', timeout: 20000 }).trim(); } catch (e) { return (e.stdout || '') + (e.stderr || ''); } };
async function waitAdb(seconds = 120) {
  const port = rdbPort(); if (!port) return null;
  const serial = `localhost:${port}`;
  for (let i = 0; i < seconds / 3; i++) {
    adb('connect', serial);
    if (adb('-s', serial, 'get-state') === 'device' && adb('-s', serial, 'shell', 'getprop', 'ro.product.model')) return serial;
    if (i % 5 === 4) { adb('disconnect', serial); }
    await sleep(3000);
  }
  return null;
}

// ---------- commands ----------
async function main() {
  if (cmd === 'login') {
    await stopChrome(); await sleep(1000);
    await startChrome(false);
    const b = await browser(); const s = await sitePage(b);
    log('Sign in to Samsung in the Chrome window that just opened. Waiting up to 10 minutes...');
    for (let i = 0; i < 240; i++) { const u = await me(b, s).catch(() => null); if (u) { log('signed in, credits:', u.point); break; } await sleep(2500); }
    b.close(); await stopChrome(); return;
  }
  await startChrome(true);
  const b = await browser(); const s = await sitePage(b);
  const user = await me(b, s);
  if (!user) throw new Error('Not signed in. Run: node rtl.mjs login');

  if (cmd === 'devices') {
    const model = opt('model', ''); const onlyFree = args.includes('--free');
    const list = (await allDevices(b, s)).filter(d => (!model || d.product.toLowerCase().includes(model.toLowerCase())) && (!onlyFree || d.free));
    console.log(JSON.stringify({ credits: user.point, devices: list }, null, 1));
  } else if (cmd === 'rent') {
    const model = opt('model', ''); const device = opt('device', '');
    const minutes = Number(opt('minutes', '30')); const units = Math.max(1, Math.ceil(minutes / 15)); // 1 credit per 15 minutes
    const loc = opt('location', ''); const os = opt('os', '');
    const all = await allDevices(b, s);
    // Exact model name first (e.g. "Galaxy S26" should not pick an S26 Ultra), then any name containing it.
    const byModel = d => !model || d.product.toLowerCase() === model.toLowerCase();
    const byPart = d => !model || d.product.toLowerCase().includes(model.toLowerCase());
    const byDevice = d => !device || String(d.id) === device || d.name.toLowerCase() === device.toLowerCase();
    const where = d => d.free && byDevice(d) && (!loc || d.location.toLowerCase().includes(loc.toLowerCase())) && (!os || d.os.includes(os));
    let cands = all.filter(d => where(d) && byModel(d));
    if (!cands.length) cands = all.filter(d => where(d) && byPart(d));
    if (!cands.length) throw new Error(`no free device matches${model ? ' ' + model : ''}${device ? ' ' + device : ''}${loc ? ' in ' + loc : ''}${os ? ' on Android ' + os : ''} right now`);
    let rented;
    for (const d of cands) {
      const r = await api(b, s, 'devices/webclient/start', 'POST', { did: d.id, tsg: units, _ts: Date.now() });
      if (r.status === 200) { rented = d; break; }
      log(`${d.name}: ${JSON.stringify(r.body)}`);
    }
    if (!rented) throw new Error('could not rent any of the free devices');
    const res = (await api(b, s, 'devices/reservation')).body.find(x => x.deviceId === rented.id);
    saveState({ reservationId: res?.reservationId, device: rented, rentedAt: Date.now(), minutes: units * 15 });
    log(`rented ${rented.product} ${rented.name} (${rented.location}, ${rented.os}) for ${units * 15} minutes`);
    const v = await openViewer(b, s, null);
    await sleep(8000); log('viewer:', (await viewerText(b, v.session)).replace(/\s+/g, ' ').slice(0, 300));
    ensureRdb();
    const serial = await waitAdb(Number(opt('wait', '120')));
    console.log(JSON.stringify({ rented: rented.name, product: rented.product, reservationId: res?.reservationId, minutes: units * 15, adb: serial }, null, 1));
  } else if (cmd === 'viewer') {
    const id = state().reservationId; if (!id) throw new Error('no reservation in ' + STATE);
    const v = await openViewer(b, s, id); await sleep(8000);
    console.log((await viewerText(b, v.session)).slice(0, 1500));
  } else if (cmd === 'status') {
    const res = (await api(b, s, 'devices/reservation')).body;
    const port = rdbPort(); const serial = port ? `localhost:${port}` : null;
    const viewers = (await pages(b)).filter(t => t.url.includes('/webclient')).length;
    console.log(JSON.stringify({ credits: user.point, reservations: (res || []).map(r => ({ id: r.reservationId, device: r.deviceName, product: r.productName, minutesLeft: r.durationHour * 60 + r.durationMinute })),
      viewerOpen: viewers > 0, adb: serial ? `${serial} ${adb('-s', serial, 'get-state') || 'not connected'}` : 'rdb not running' }, null, 1));
  } else if (cmd === 'eval') { // debugging: run JS in the viewer page
    const vp = (await pages(b)).find(t => t.url.includes('/webclient')); if (!vp) throw new Error('no viewer');
    const vs = await attach(b, vp.targetId); console.log(JSON.stringify(await evaluate(b, vs, args[1]), null, 1));
  } else if (cmd === 'screenshot') { // debugging: PNG of the viewer page
    const vp = (await pages(b)).find(t => t.url.includes('/webclient')); if (!vp) throw new Error('no viewer');
    const vs = await attach(b, vp.targetId); const { data } = await b.send('Page.captureScreenshot', { format: 'png' }, vs);
    const out = opt('out', join(HOME, 'viewer.png')); writeFileSync(out, Buffer.from(data, 'base64')); console.log(out);
  } else if (cmd === 'release') {
    for (const r of (await api(b, s, 'devices/reservation')).body || []) {
      const x = await api(b, s, 'devices/cancel-reservation', 'POST', { seq: r.reservationId }); log(`returned ${r.deviceName}: ${x.status}`);
    }
    saveState({}); b.close(); await stopChrome(); return;
  } else {
    console.log(readFileSync(new URL(import.meta.url), 'utf8').split('\n').slice(1, 12).join('\n'));
  }
  b.close();
}
main().catch(e => { console.error('[rtl] error:', e.message); process.exit(1); });
