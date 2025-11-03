// 前端显示与后端交互（主题/参数编辑/轨迹显示；控制与仿真在后端）

// 车辆参数（后端字段，仅用于显示与表单绑定）；控制与仿真均在后端
let params = {
  m: 1500, Iz: 2500, a: 1.2, b: 1.6,
  width: 1.8, track: 1.5,
  kf: 1.6e5, kr: 1.7e5,
  U: 20, mu: 0.85, g: 9.81, U_min: 0.001, U_blend: 0.3,
  tire_model: 'pacejka',
};
let state = { x: 0, y: 0, psi: 0, beta: 0, r: 0 };
let ctrl = { U: 20, df: 0, dr: 0, running: false };

// 轨迹缓冲与设置
let track = [];
let plan = [];
const trackCfg = { enabled: true, retentionSec: 30, maxPoints: 20000 };
// 到达终点检测与动画状态
let arrived = false;
let arriveTick = 0;
const arrivalAnim = { active: false, t0: 0, duration: 2000 };

// Canvas 与视图
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
let view = { scale: 20, offsetX: 0, offsetY: 0, follow: true, showGrid: true };

// CSS 变量读取
function getCSSVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// 后端连通指示器
let backendConnected = false;
function setConnStatus(ok) {
  backendConnected = ok;
  const dot = document.getElementById('conn_dot');
  if (!dot) return;
  dot.classList.toggle('ok', ok);
  dot.classList.toggle('bad', !ok);
}
async function pingBackend() {
  try { const res = await fetch('/api/params', { cache: 'no-store' }); setConnStatus(res.ok); }
  catch { setConnStatus(false); }
}

// 绑定 UI 控件
function bindUI() {
  // 主题与外观（右上角）
  const themeSel = document.getElementById('theme_select');
  const customThemeRow = document.getElementById('custom_theme');
  const accentPicker = document.getElementById('accent_color');
  const okPicker = document.getElementById('ok_color');
  // 左侧参数栏折叠/展开
  const gridEl = document.querySelector('.app-grid');
  const toggleLeftBtn = document.getElementById('toggle_left_panel');
  try {
    const saved = localStorage.getItem('leftCollapsed');
    if (saved === '1') { gridEl && gridEl.classList.add('collapsed-left'); if (toggleLeftBtn) toggleLeftBtn.textContent = '展开参数栏'; }
  } catch {}
  if (toggleLeftBtn) {
    toggleLeftBtn.addEventListener('click', () => {
      if (!gridEl) return;
      const collapsed = gridEl.classList.toggle('collapsed-left');
      try { localStorage.setItem('leftCollapsed', collapsed ? '1' : '0'); } catch {}
      toggleLeftBtn.textContent = collapsed ? '展开参数栏' : '折叠参数栏';
      draw();
    });
  }

  const prefersDarkMQ = window.matchMedia('(prefers-color-scheme: dark)');
  let systemListener = null;
  function applyTheme(v) {
    localStorage.setItem('theme', v);
    customThemeRow.style.display = v === 'custom' ? 'flex' : 'none';
    // tear down system listener if leaving system mode
    if (systemListener && v !== 'system') { try { prefersDarkMQ.removeEventListener('change', systemListener); } catch {} systemListener = null; }
    if (v === 'light') {
      document.documentElement.classList.add('light');
      document.documentElement.style.removeProperty('--accent');
      document.documentElement.style.removeProperty('--ok');
    } else if (v === 'dark') {
      document.documentElement.classList.remove('light');
      document.documentElement.style.removeProperty('--accent');
      document.documentElement.style.removeProperty('--ok');
    } else if (v === 'system') {
      const isDark = !!prefersDarkMQ.matches;
      document.documentElement.classList.toggle('light', !isDark);
      document.documentElement.style.removeProperty('--accent');
      document.documentElement.style.removeProperty('--ok');
      if (!systemListener) {
        systemListener = (e) => { document.documentElement.classList.toggle('light', !e.matches); };
        try { prefersDarkMQ.addEventListener('change', systemListener); } catch {}
      }
    } else if (v === 'custom') {
      // keep current light/dark class as-is; apply custom colors
      document.documentElement.style.setProperty('--accent', accentPicker.value);
      document.documentElement.style.setProperty('--ok', okPicker.value);
    }
  }
  const savedTheme = localStorage.getItem('theme') || 'light';
  const savedAccent = localStorage.getItem('accent') || getCSSVar('--accent');
  const savedOk = localStorage.getItem('ok') || getCSSVar('--ok');
  themeSel.value = savedTheme;
  accentPicker.value = savedAccent || '#6aa0ff';
  okPicker.value = savedOk || '#4caf50';
  applyTheme(savedTheme);
  themeSel.addEventListener('change', () => { applyTheme(themeSel.value); });

  // 模式与场景切换
  const modeSel = document.getElementById('mode_select');
  async function loadMode() {
    try {
      const mRes = await fetch('/api/mode', { cache: 'no-store' });
      if (mRes.ok) { const m = await mRes.json(); if (m && m.mode) modeSel.value = m.mode; }
    } catch {}
  }
  loadMode();
  modeSel.addEventListener('change', async () => {
    try { await fetch('/api/mode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: modeSel.value }) }); }
    catch {}
  });
  accentPicker.addEventListener('input', () => {
    document.documentElement.style.setProperty('--accent', accentPicker.value);
    localStorage.setItem('accent', accentPicker.value);
  });
  okPicker.addEventListener('input', () => {
    document.documentElement.style.setProperty('--ok', okPicker.value);
    localStorage.setItem('ok', okPicker.value);
  });

  // 控制按钮
  const startPauseBtn = document.getElementById('start_pause_btn');
  const resetBtn = document.getElementById('reset_btn');
  startPauseBtn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/sim/start_pause', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ running: !ctrl.running }) });
      const data = await res.json();
      ctrl.running = !!data.running; startPauseBtn.textContent = ctrl.running ? '暂停' : '开始';
    } catch (e) { console.warn('POST /api/sim/start_pause 失败', e); }
  });
  resetBtn.addEventListener('click', async () => {
    try { await fetch('/api/sim/reset', { method: 'POST' }); ctrl.running = false; startPauseBtn.textContent = '开始'; } catch (e) {}
  });

  // 起始位姿
  const applyInitPoseBtn = document.getElementById('apply_init_pose');
  const initX = document.getElementById('init_x');
  const initY = document.getElementById('init_y');
  const initPsi = document.getElementById('init_psi');
  applyInitPoseBtn.addEventListener('click', async () => {
    const x0 = parseFloat(initX.value) || 0;
    const y0 = parseFloat(initY.value) || 0;
    const psi0deg = parseFloat(initPsi.value) || 0;
    try { await fetch('/api/init_pose', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ x: x0, y: y0, psi: psi0deg }) }); } catch (e) {}
    view.offsetX = 0; view.offsetY = 0; if (view.follow) centerOnVehicle();
  });

  // 终点与规划
  const goalX = document.getElementById('goal_x');
  const goalY = document.getElementById('goal_y');
  const goalPsi = document.getElementById('goal_psi');
  const planOnlyBtn = document.getElementById('plan_only');
  const autopStartBtn = document.getElementById('autop_start');
  const planMethodSel = document.getElementById('plan_method_select');
  // 规划时长 T 已移除，由后端自动估算
  const autopModeSel = document.getElementById('autop_mode_select');
  const autopStopBtn = document.getElementById('autop_stop');
  // 初始化自动模式选择器与后端同步
  try {
    fetch('/api/autop', { cache: 'no-store' }).then(res => res.ok ? res.json() : null).then(data => {
      if (data && data.mode && autopModeSel) autopModeSel.value = data.mode;
    }).catch(() => {});
  } catch {}
  if (autopModeSel) {
    autopModeSel.addEventListener('change', async () => {
      try { await fetch('/api/autop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: autopModeSel.value }) }); } catch {}
    });
  }
  if (planOnlyBtn) {
    planOnlyBtn.addEventListener('click', async () => {
      const x0 = parseFloat(initX.value) || 0;
      const y0 = parseFloat(initY.value) || 0;
      const psi0deg = parseFloat(initPsi.value) || 0;
      const x1 = parseFloat(goalX.value) || 0;
      const y1 = parseFloat(goalY.value) || 0;
      const psi1deg = parseFloat(goalPsi.value) || 0;
      const method = (planMethodSel && planMethodSel.value) || 'quintic';
      const url = method === 'circle' ? '/api/plan/circle' : '/api/plan/quintic';
      try {
        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ start: { x: x0, y: y0, psi: psi0deg }, end: { x: x1, y: y1, psi: psi1deg } }) });
        if (res.ok) {
          // 仅规划不启动自动跟踪；重置到达状态与动画
          arrived = false; arriveTick = 0; arrivalAnim.active = false; arrivalAnim.t0 = 0;
          // 聚焦车辆视图
          view.follow = true; centerOnVehicle();
        }
      } catch (e) { console.warn('规划失败', e); }
    });
  }
  if (autopStartBtn) {
    autopStartBtn.addEventListener('click', async () => {
      try {
        const mode = (autopModeSel && autopModeSel.value) || 'mpc';
        await fetch('/api/autop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: true, start: true, mode }) });
        ctrl.running = true; startPauseBtn.textContent = '暂停';
        view.follow = true; centerOnVehicle();
      } catch (e) { console.warn('启动自动跟踪失败', e); }
    });
  }
  autopStopBtn.addEventListener('click', async () => {
    try {
      // 1) 关闭自动跟踪
      await fetch('/api/autop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: false }) });
      // 2) 暂停仿真
      await fetch('/api/sim/start_pause', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ running: false }) });
      // 3) 归零前后轮角，避免残留角度继续影响行驶
      const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ df_deg: 0, dr_deg: 0 }) });
      ctrl = await res.json();
      // 4) 同步 UI 状态
      startPauseBtn.textContent = '开始';
      arrived = false; arriveTick = 0; arrivalAnim.active = false; arrivalAnim.t0 = 0;
      draw();
    } catch (e) { console.warn('停止自动跟踪失败', e); }
  });

  // 跟随与网格
  const followChk = document.getElementById('follow_mode');
  const gridChk = document.getElementById('show_grid');
  followChk.addEventListener('change', () => { view.follow = followChk.checked; if (view.follow) centerOnVehicle(); });
  gridChk.addEventListener('change', () => { view.showGrid = gridChk.checked; draw(); });

  // 轨迹设置
  const showTrackChk = document.getElementById('show_track');
  const trackKeepInput = document.getElementById('track_keep_sec');
  if (showTrackChk) {
    trackCfg.enabled = showTrackChk.checked;
    showTrackChk.addEventListener('change', async () => {
      trackCfg.enabled = showTrackChk.checked;
      try { await fetch('/api/track/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: trackCfg.enabled }) }); } catch (e) {}
    });
  }
  if (trackKeepInput) {
    const v = parseFloat(trackKeepInput.value);
    if (!Number.isNaN(v)) trackCfg.retentionSec = Math.max(0, v);
    trackKeepInput.addEventListener('change', async () => {
      const n = parseFloat(trackKeepInput.value);
      trackCfg.retentionSec = Number.isNaN(n) ? trackCfg.retentionSec : Math.max(0, n);
      try { await fetch('/api/track/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ retentionSec: trackCfg.retentionSec }) }); } catch (e) {}
    });
  }

  // 轨迹导出为 CSV
  const exportBtn = document.getElementById('export_track');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => { try { exportTrackCSV(); } catch (e) {} });
  }

  // 已移除：自动跟踪调度（Stanley & PID），仅保留 MPC 选项

  // 手动设置：车速与前后轮角
  const speedInput = document.getElementById('speed_input');
  const dfInput = document.getElementById('df_input');
  const drInput = document.getElementById('dr_input');
  speedInput.addEventListener('change', async () => {
    const U = parseFloat(speedInput.value);
    try { const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ U }) }); ctrl = await res.json(); } catch (e) {}
  });
  dfInput.addEventListener('change', async () => {
    const df = parseFloat(dfInput.value);
    try { const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ df_deg: df }) }); ctrl = await res.json(); } catch (e) {}
  });
  drInput.addEventListener('change', async () => {
    const dr = parseFloat(drInput.value);
    try { const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dr_deg: dr }) }); ctrl = await res.json(); } catch (e) {}
  });

  // 键盘控制（保持与手动输入同步）
  const KEY_INC = { speed: 0.5, steer: 0.5 };
  window.addEventListener('keydown', async (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    let patch = null;
    switch(e.key.toLowerCase()) {
      case 'w': patch = { U: (ctrl.U + KEY_INC.speed) }; break;
      case 's': patch = { U: (ctrl.U - KEY_INC.speed) }; break;
      case 'a': patch = { df_deg: (deg(ctrl.df) + KEY_INC.steer) }; break;
      case 'd': patch = { df_deg: (deg(ctrl.df) - KEY_INC.steer) }; break;
      case 'q': patch = { dr_deg: (deg(ctrl.dr) - KEY_INC.steer) }; break;
      case 'e': patch = { dr_deg: (deg(ctrl.dr) + KEY_INC.steer) }; break;
      default: break;
    }
    if (patch) {
      try { const res = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) }); ctrl = await res.json(); } catch (err) {}
    }
  });

  // 车辆参数输入（回写后端）
  const pInputs = {
    m: document.getElementById('p_m'), Iz: document.getElementById('p_Iz'),
    a: document.getElementById('p_a'), b: document.getElementById('p_b'),
    width: document.getElementById('p_width'), track: document.getElementById('p_track'),
    kf: document.getElementById('p_kf'), kr: document.getElementById('p_kr'),
    mu: document.getElementById('p_mu'), g: document.getElementById('p_g'), U_min: document.getElementById('p_Umin'), U_blend: document.getElementById('p_Ublend'),
  };
  for (const key of Object.keys(pInputs)) {
    pInputs[key].addEventListener('change', () => {
      const val = parseFloat(pInputs[key].value);
      if (!Number.isNaN(val)) { params[key] = val; schedulePostParams(); updateDerivedStats(); }
    });
  }
  // 轮胎模型选择（字符串）
  const tireSel = document.getElementById('p_tire_model');
  if (tireSel) {
    tireSel.value = params.tire_model || 'pacejka';
    tireSel.addEventListener('change', () => {
      params.tire_model = tireSel.value;
      schedulePostParams();
    });
  }

  // 画布交互 + 自适应
  setupCanvasInteractions();
  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // 工具栏
  document.getElementById('zoom_in').addEventListener('click', () => { view.scale *= 1.2; draw(); });
  document.getElementById('zoom_out').addEventListener('click', () => { view.scale /= 1.2; draw(); });
  document.getElementById('reset_view').addEventListener('click', () => { view.scale = 20; view.offsetX = 0; view.offsetY = 0; centerOnVehicle(); draw(); });

  // 后端心跳
  pingBackend(); setInterval(pingBackend, 5000);

  // 显示循环：轮询后端状态与轨迹，仅负责显示
  async function pollOnce() {
    try {
      const [stRes, trRes, ctRes, plRes] = await Promise.all([
        fetch('/api/state', { cache: 'no-store' }),
        fetch('/api/track', { cache: 'no-store' }),
        fetch('/api/control', { cache: 'no-store' }),
        fetch('/api/plan', { cache: 'no-store' }),
      ]);
      if (stRes.ok) { state = await stRes.json(); }
      if (trRes.ok) { const td = await trRes.json(); track = Array.isArray(td.points) ? td.points : []; }
      if (ctRes.ok) { ctrl = await ctRes.json(); }
      if (plRes.ok) { const pd = await plRes.json(); plan = Array.isArray(pd.points) ? pd.points : []; }
      // 同步按钮文字与输入框显示（避免覆盖正在编辑的元素）
      startPauseBtn.textContent = ctrl.running ? '暂停' : '开始';
      const safeSet = (el, v) => { if (!el || document.activeElement === el) return; el.value = String(v); };
      safeSet(speedInput, (ctrl.U ?? params.U).toFixed(2));
      safeSet(dfInput, deg(ctrl.df ?? 0).toFixed(1));
      safeSet(drInput, deg(ctrl.dr ?? 0).toFixed(1));
      // 同步模式与场景显示
      if (ctrl.mode && modeSel && document.activeElement !== modeSel) modeSel.value = ctrl.mode;
      updateDashboard();
      if (view.follow) centerOnVehicle();
      // 到达终点判定与处理（触发一次）
      const tgt = (Array.isArray(plan) && plan.length > 0) ? plan[plan.length - 1] : null;
      if (tgt) {
        const dx = state.x - (tgt.x || 0);
        const dy = state.y - (tgt.y || 0);
        const dist = Math.hypot(dx, dy);
        const dpsi = angleDiff(state.psi || 0, tgt.psi || 0);
        const distTh = 0.8; // m
        const angTh = 10 * Math.PI / 180; // rad
        if (dist <= distTh && Math.abs(dpsi) <= angTh) {
          arriveTick++;
        } else {
          arriveTick = 0;
        }
        if (!arrived && arriveTick >= 5) { // ~0.5s 稳定到达
          arrived = true;
          arrivalAnim.active = true; arrivalAnim.t0 = performance.now();
          // 停止自动跟踪，并将前后轮角设为 0 保持直行
          try { await fetch('/api/autop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: false }) }); } catch (e) {}
          try { await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ df_deg: 0, dr_deg: 0 }) }); } catch (e) {}
          ctrl.df = 0; ctrl.dr = 0;
        }
      }
      draw();
    } catch (e) {}
  }
  setInterval(pollOnce, 100); // 10Hz 轮询
}

// 前端不再进行积分与复位，复位由后端处理
function resetSim() { /* no-op; 后端负责 */ }

// 交互：滚轮缩放、拖拽平移
function setupCanvasInteractions() {
  let dragging = false; let last = { x: 0, y: 0 };
  canvas.addEventListener('mousedown', (e) => { dragging = true; last.x = e.clientX; last.y = e.clientY; });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', (e) => {
    if (!dragging || view.follow) return;
    const dx = e.clientX - last.x; const dy = e.clientY - last.y;
    last.x = e.clientX; last.y = e.clientY;
    view.offsetX += dx; view.offsetY += dy; draw();
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault(); const factor = e.deltaY < 0 ? 1.1 : 0.9; view.scale *= factor; draw();
  }, { passive: false });
}

// 已迁移到后端：积分与控制

function centerOnVehicle() {
  const rect = canvas.getBoundingClientRect();
  const cx = rect.width / 2; const cy = rect.height / 2;
  const sx = worldToScreenX(state.x); const sy = worldToScreenY(state.y);
  view.offsetX += (cx - sx); view.offsetY += (cy - sy);
}

// 仪表盘与派生量更新
function updateDashboard() {
  document.getElementById('stat_x').textContent = state.x.toFixed(2);
  document.getElementById('stat_y').textContent = state.y.toFixed(2);
  document.getElementById('stat_psi').textContent = (state.psi * 180 / Math.PI).toFixed(2);
  document.getElementById('stat_u').textContent = (ctrl.U ?? params.U).toFixed(2);
  document.getElementById('stat_df').textContent = deg(ctrl.df ?? 0).toFixed(1);
  document.getElementById('stat_dr').textContent = deg(ctrl.dr ?? 0).toFixed(1);
  const v = (typeof state.speed === 'number') ? state.speed : (ctrl.U ?? params.U);
  document.getElementById('stat_v').textContent = v.toFixed(2);
  const R = (typeof state.radius === 'number') ? state.radius : null;
  document.getElementById('stat_R').textContent = (R !== null) ? R.toFixed(2) : '—';
  const dfDot = (typeof state.df_dot === 'number') ? state.df_dot : 0;
  const drDot = (typeof state.dr_dot === 'number') ? state.dr_dot : 0;
  document.getElementById('stat_df_dot').textContent = deg(dfDot).toFixed(1);
  document.getElementById('stat_dr_dot').textContent = deg(drDot).toFixed(1);
}
function updateDerivedStats() {
  // L 与 K 在后端计算亦可，这里简易前端计算与展示
  const L = params.a + params.b;
  const K = (params.m / L) * (params.b / params.kr - params.a / params.kf);
  const pL = document.getElementById('p_L'); const pK = document.getElementById('p_K');
  if (pL) pL.textContent = L.toFixed(3);
  if (pK) pK.textContent = K.toFixed(6);
}

// 绘制
function worldToScreenX(x) { return x * view.scale + view.offsetX; }
function worldToScreenY(y) { return -y * view.scale + view.offsetY + canvas.height / (window.devicePixelRatio || 1); }
function drawGrid() {
  if (!view.showGrid) return;
  const rect = canvas.getBoundingClientRect(); const step = view.scale;
  ctx.strokeStyle = getCSSVar('--grid') || 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1; ctx.beginPath();
  for (let x = view.offsetX % step; x < rect.width; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); }
  for (let y = view.offsetY % step; y < rect.height; y += step) { ctx.moveTo(0, y); ctx.lineTo(rect.width, y); }
  ctx.stroke();
}
function drawTrack() {
  if (!trackCfg.enabled || track.length < 2) return;
  const rect = canvas.getBoundingClientRect();
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = getCSSVar('--accent') || '#6aa0ff';
  ctx.globalAlpha = 0.6;
  ctx.beginPath();
  // 只绘制当前视窗范围内的点以提升性能（简化，仍遍历一次）
  for (let i = 0; i < track.length; i++) {
    const sx = worldToScreenX(track[i].x);
    const sy = worldToScreenY(track[i].y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.restore();
}
function drawPlan() {
  if (!Array.isArray(plan) || plan.length < 2) return;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = getCSSVar('--accent') || '#6aa0ff';
  ctx.globalAlpha = 0.35;
  ctx.beginPath();
  for (let i = 0; i < plan.length; i++) {
    const sx = worldToScreenX(plan[i].x);
    const sy = worldToScreenY(plan[i].y);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.restore();
}
function drawVehicle() {
  const x = worldToScreenX(state.x); const y = worldToScreenY(state.y);
  const L = (params.a + params.b);
  const W = (params.width || 1.8);
  const TR = (params.track || Math.max(0.5, W - 0.3));
  const w = L * view.scale; const h = W * view.scale;
  ctx.save(); ctx.translate(x, y); ctx.rotate(-state.psi);
  // 车身：矩形（后部） + 梯形（前部）组合
  const aPix = params.a * view.scale;
  const bPix = params.b * view.scale;
  const kFront = 0.65; // 前端收窄比例（0~1）
  const tPixRaw = L * 0.10 * view.scale; // 梯形深度为总长的10%
  const tPix = Math.min(aPix, Math.max(0, tPixRaw));
  ctx.fillStyle = 'rgba(106,160,255,0.9)'; ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1.5;
  // 后部矩形：从 -b 到 0，宽度为 W
  // 调整为从 -b 到 (a - tPix)，预留前10%长度画梯形
  ctx.beginPath();
  ctx.rect(-bPix, -h/2, (aPix - tPix) + bPix, h);
  ctx.fill(); ctx.stroke();
  // 前部梯形：从 0 到 +a，前端收窄为 kFront*W
  ctx.beginPath();
  ctx.moveTo(aPix - tPix, -h/2);                    // 接矩形上边
  ctx.lineTo(+aPix, -h/2 * kFront);                 // 前左（收窄）
  ctx.lineTo(+aPix, +h/2 * kFront);                 // 前右（收窄）
  ctx.lineTo(aPix - tPix, +h/2);                    // 接矩形下边
  ctx.closePath();
  ctx.fill(); ctx.stroke();

  // 车轴线
  ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1.2;
  const trPix = TR * view.scale;
  // 轴线位置
  ctx.beginPath();
  ctx.moveTo(+aPix, -trPix/2); ctx.lineTo(+aPix, +trPix/2);
  ctx.moveTo(-bPix, -trPix/2); ctx.lineTo(-bPix, +trPix/2);
  ctx.stroke();

  // 轮胎矩形（按轮角旋转）
  const wheelLen = 0.6 * view.scale;
  const wheelWid = 0.25 * view.scale;
  const df = (typeof state.df === 'number') ? state.df : (ctrl.df || 0);
  const dr = (typeof state.dr === 'number') ? state.dr : (ctrl.dr || 0);
  const drawWheel = (cx, cy, ang) => {
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(-ang);
    ctx.fillStyle = 'rgba(30,30,30,0.9)'; ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 1.0;
    ctx.beginPath(); ctx.rect(-wheelLen/2, -wheelWid/2, wheelLen, wheelWid); ctx.fill(); ctx.stroke();
    ctx.restore();
  };
  drawWheel(+aPix, +trPix/2, df);
  drawWheel(+aPix, -trPix/2, df);
  drawWheel(-bPix, +trPix/2, dr);
  drawWheel(-bPix, -trPix/2, dr);

  ctx.restore();
  // 质心标记
  ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI*2); ctx.fill();
}
function drawGoalVehicle() {
  const tgt = (Array.isArray(plan) && plan.length > 0) ? plan[plan.length - 1] : null;
  if (!tgt) return;
  const x = worldToScreenX(tgt.x); const y = worldToScreenY(tgt.y);
  const L = (params.a + params.b);
  const W = (params.width || 1.8);
  const TR = (params.track || Math.max(0.5, W - 0.3));
  const w = L * view.scale; const h = W * view.scale;
  const psi = tgt.psi || 0;
  ctx.save(); ctx.translate(x, y); ctx.rotate(-psi);
  // 浅色车身：后矩形 + 前梯形
  const aPix = params.a * view.scale;
  const bPix = params.b * view.scale;
  const kFront = 0.65;
  const tPixRaw = L * 0.10 * view.scale;
  const tPix = Math.min(aPix, Math.max(0, tPixRaw));
  ctx.fillStyle = 'rgba(106,160,255,0.35)'; ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1.2;
  // 后部矩形
  ctx.beginPath();
  ctx.rect(-bPix, -h/2, (aPix - tPix) + bPix, h);
  ctx.fill(); ctx.stroke();
  // 前部梯形
  ctx.beginPath();
  ctx.moveTo(aPix - tPix, -h/2);
  ctx.lineTo(+aPix, -h/2 * kFront);
  ctx.lineTo(+aPix, +h/2 * kFront);
  ctx.lineTo(aPix - tPix, +h/2);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
  // 车轴线
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1.0;
  const trPix = TR * view.scale;
  ctx.beginPath();
  ctx.moveTo(+aPix, -trPix/2); ctx.lineTo(+aPix, +trPix/2);
  ctx.moveTo(-bPix, -trPix/2); ctx.lineTo(-bPix, +trPix/2);
  ctx.stroke();
  // 轮胎（不显示转角，作为终点标识）
  const wheelLen = 0.6 * view.scale;
  const wheelWid = 0.25 * view.scale;
  const drawWheel = (cx, cy) => {
    ctx.save(); ctx.translate(cx, cy);
    ctx.fillStyle = 'rgba(30,30,30,0.6)'; ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.rect(-wheelLen/2, -wheelWid/2, wheelLen, wheelWid); ctx.fill(); ctx.stroke();
    ctx.restore();
  };
  drawWheel(+aPix, +trPix/2);
  drawWheel(+aPix, -trPix/2);
  drawWheel(-bPix, +trPix/2);
  drawWheel(-bPix, -trPix/2);
  ctx.restore();
}
function drawArrivalEffect() {
  if (!arrivalAnim.active) return;
  const t = performance.now() - (arrivalAnim.t0 || 0);
  if (t > arrivalAnim.duration) { arrivalAnim.active = false; return; }
  const tgt = (Array.isArray(plan) && plan.length > 0) ? plan[plan.length - 1] : null;
  if (!tgt) return;
  const x = worldToScreenX(tgt.x || 0);
  const y = worldToScreenY(tgt.y || 0);
  const ok = getCSSVar('--ok') || '#4caf50';
  ctx.save();
  // 脉冲圆环
  const cycles = 3;
  for (let i = 0; i < cycles; i++) {
    const phase = Math.max(0, Math.min(1, (t / arrivalAnim.duration) + i * 0.25));
    const r = (params.width || 1.8) * view.scale * (0.6 + 0.8 * phase);
    const alpha = 1.0 - phase;
    ctx.beginPath();
    ctx.strokeStyle = ok;
    ctx.globalAlpha = 0.7 * alpha;
    ctx.lineWidth = 3;
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  // 成功提示
  ctx.globalAlpha = 0.95;
  ctx.fillStyle = ok;
  ctx.font = `${Math.round(14 * (window.devicePixelRatio || 1))}px sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  ctx.fillText('到达终点', x, y - (params.width || 1.8) * view.scale * 0.8);
  ctx.textBaseline = 'top';
  ctx.fillText('✓', x, y + (params.width || 1.8) * view.scale * 0.1);
  ctx.restore();
}
function draw() { const rect = canvas.getBoundingClientRect(); ctx.clearRect(0, 0, rect.width, rect.height); drawGrid(); drawPlan(); drawTrack(); drawVehicle(); drawGoalVehicle(); drawArrivalEffect(); }

// 导出当前保留的轨迹为 CSV 文件
function exportTrackCSV() {
  try {
    if (!Array.isArray(track) || track.length === 0) { alert('当前无可导出的轨迹'); return; }
    const hasTime = typeof track[0].t === 'number';
    const t0 = hasTime ? track[0].t : null;
    const rows = [];
    rows.push('t,x,y');
    for (let i = 0; i < track.length; i++) {
      const p = track[i] || {};
      const t = (hasTime && typeof p.t === 'number') ? (t0 !== null ? (p.t - t0) : p.t) : '';
      const x = (typeof p.x === 'number') ? p.x : '';
      const y = (typeof p.y === 'number') ? p.y : '';
      rows.push(`${t},${x},${y}`);
    }
    const csv = rows.join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    a.download = `track_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
  } catch (e) {
    console.warn('导出 CSV 失败', e);
    alert('导出失败，请稍后重试');
  }
}

// 后端交互（节流）
let postTimer = null;
function schedulePostParams() {
  if (postTimer) clearTimeout(postTimer);
  postTimer = setTimeout(async () => {
    try {
      const patch = { m: params.m, Iz: params.Iz, a: params.a, b: params.b, width: params.width, track: params.track, kf: params.kf, kr: params.kr, U: params.U, mu: params.mu, g: params.g, U_min: params.U_min, U_blend: params.U_blend, tire_model: params.tire_model };
      const res = await fetch('/api/params', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch) });
      setConnStatus(res.ok);
    } catch (e) { console.warn('POST /api/params 失败', e); setConnStatus(false); }
  }, 200);
}
async function loadParams() {
  try {
    const res = await fetch('/api/params'); setConnStatus(res.ok);
    const data = await res.json();
    // 覆写参数
    for (const k of ['m','Iz','a','b','width','track','kf','kr','U','mu','g','U_min','U_blend','tire_model']) { if (typeof data[k] !== 'undefined') params[k] = data[k]; }
    // 更新输入框
    const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = String(v); };
    setVal('speed_input', (ctrl.U ?? params.U).toFixed(2));
    setVal('df_input', deg(ctrl.df ?? 0).toFixed(1));
    setVal('dr_input', deg(ctrl.dr ?? 0).toFixed(1));
    setVal('p_m', params.m); setVal('p_Iz', params.Iz); setVal('p_a', params.a); setVal('p_b', params.b);
    setVal('p_kf', params.kf); setVal('p_kr', params.kr); setVal('p_width', params.width); setVal('p_track', params.track); setVal('p_mu', params.mu); setVal('p_g', params.g); setVal('p_Umin', params.U_min); setVal('p_Ublend', params.U_blend);
    setVal('p_tire_model', params.tire_model);
    // 更新派生量显示（也可从后端 data.L / data.K）
    const pL = document.getElementById('p_L'); const pK = document.getElementById('p_K');
    if (pL && typeof data.L !== 'undefined') pL.textContent = Number(data.L).toFixed(3);
    if (pK && typeof data.K !== 'undefined') pK.textContent = Number(data.K).toFixed(6);
  updateDashboard();
  } catch (e) { console.warn('GET /api/params 失败', e); setConnStatus(false); }
}

// 初始化
window.addEventListener('load', async () => { bindUI(); await loadParams(); updateDerivedStats(); });

// 工具：度/弧度转换
function normAngle(a) { let x = a; while (x > Math.PI) x -= 2 * Math.PI; while (x < -Math.PI) x += 2 * Math.PI; return x; }
function angleDiff(a, b) { return normAngle(a - b); }
function deg(rad) { return rad * 180 / Math.PI; }
function rad(deg) { return deg * Math.PI / 180; }