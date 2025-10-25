// 前端交互与地图渲染（新界面 + 连通指示器 + 右上角主题 + 参数编辑 + 轨迹显示）

// 车辆参数（含后端字段）与本地控制量
let params = {
  m: 1500, Iz: 2500, a: 1.2, b: 1.6,
  kf: 1.6e5, kr: 1.7e5,
  U: 20, mu: 0.85, g: 9.81, U_min: 0.5,
  df: 0, dr: 0,
};
let state = { x: 0, y: 0, psi: 0, beta: 0, r: 0 };

// 轨迹缓冲与设置
let track = [];
const trackCfg = { enabled: true, retentionSec: 30, maxPoints: 20000 };
function nowSec() { return performance.now() / 1000; }
function pushTrackPoint() {
  track.push({ x: state.x, y: state.y, t: nowSec() });
  // 保留时长裁剪与最大点数限制
  const keep = trackCfg.retentionSec;
  if (keep > 0) {
    const tcut = nowSec() - keep;
    // 快速裁剪前端旧点
    let i = 0; while (i < track.length && track[i].t < tcut) i++;
    if (i > 0) track.splice(0, i);
  }
  if (track.length > trackCfg.maxPoints) track.splice(0, track.length - trackCfg.maxPoints);
}

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

  const savedTheme = localStorage.getItem('theme') || 'dark';
  const savedAccent = localStorage.getItem('accent') || getCSSVar('--accent');
  const savedOk = localStorage.getItem('ok') || getCSSVar('--ok');
  themeSel.value = savedTheme;
  document.documentElement.classList.toggle('light', savedTheme === 'light');
  customThemeRow.style.display = savedTheme === 'custom' ? 'flex' : 'none';
  accentPicker.value = savedAccent || '#6aa0ff';
  okPicker.value = savedOk || '#4caf50';
  if (savedTheme === 'custom') {
    document.documentElement.style.setProperty('--accent', accentPicker.value);
    document.documentElement.style.setProperty('--ok', okPicker.value);
  }
  themeSel.addEventListener('change', () => {
    const v = themeSel.value;
    document.documentElement.classList.toggle('light', v === 'light');
    customThemeRow.style.display = v === 'custom' ? 'flex' : 'none';
    localStorage.setItem('theme', v);
    if (v !== 'custom') {
      document.documentElement.style.removeProperty('--accent');
      document.documentElement.style.removeProperty('--ok');
    } else {
      document.documentElement.style.setProperty('--accent', accentPicker.value);
      document.documentElement.style.setProperty('--ok', okPicker.value);
    }
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
  let running = false;
  startPauseBtn.addEventListener('click', () => {
    running = !running;
    startPauseBtn.textContent = running ? '暂停' : '开始';
  });
  resetBtn.addEventListener('click', () => {
    running = false; startPauseBtn.textContent = '开始'; resetSim(); draw();
  });

  // 起始位姿
  const applyInitPoseBtn = document.getElementById('apply_init_pose');
  const initX = document.getElementById('init_x');
  const initY = document.getElementById('init_y');
  const initPsi = document.getElementById('init_psi');
  applyInitPoseBtn.addEventListener('click', () => {
    const x0 = parseFloat(initX.value) || 0;
    const y0 = parseFloat(initY.value) || 0;
    const psi0deg = parseFloat(initPsi.value) || 0;
    state.x = x0; state.y = y0; state.psi = psi0deg * Math.PI / 180;
    view.offsetX = 0; view.offsetY = 0; if (view.follow) centerOnVehicle(); draw();
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
    showTrackChk.addEventListener('change', () => { trackCfg.enabled = showTrackChk.checked; draw(); });
  }
  if (trackKeepInput) {
    const v = parseFloat(trackKeepInput.value);
    if (!Number.isNaN(v)) trackCfg.retentionSec = Math.max(0, v);
    trackKeepInput.addEventListener('change', () => {
      const n = parseFloat(trackKeepInput.value);
      trackCfg.retentionSec = Number.isNaN(n) ? trackCfg.retentionSec : Math.max(0, n);
      // 立即裁剪并刷新
      const keep = trackCfg.retentionSec;
      if (keep > 0) {
        const tcut = nowSec() - keep;
        let i = 0; while (i < track.length && track[i].t < tcut) i++;
        if (i > 0) track.splice(0, i);
      }
      draw();
    });
  }

  // 手动设置：车速与前后轮角
  const speedInput = document.getElementById('speed_input');
  const dfInput = document.getElementById('df_input');
  const drInput = document.getElementById('dr_input');
  speedInput.addEventListener('change', () => { params.U = parseFloat(speedInput.value) || params.U; schedulePostParams(); updateDashboard(); });
  dfInput.addEventListener('change', () => { params.df = parseFloat(dfInput.value) || params.df; updateDashboard(); });
  drInput.addEventListener('change', () => { params.dr = parseFloat(drInput.value) || params.dr; updateDashboard(); });

  // 键盘控制（保持与手动输入同步）
  const KEY_INC = { speed: 0.5, steer: 0.5 };
  window.addEventListener('keydown', (e) => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    switch(e.key.toLowerCase()) {
      case 'w': params.U += KEY_INC.speed; speedInput.value = params.U.toFixed(2); schedulePostParams(); break;
      case 's': params.U = Math.max(0, params.U - KEY_INC.speed); speedInput.value = params.U.toFixed(2); schedulePostParams(); break;
      case 'a': params.df -= KEY_INC.steer; dfInput.value = params.df.toFixed(1); break;
      case 'd': params.df += KEY_INC.steer; dfInput.value = params.df.toFixed(1); break;
      case 'q': params.dr -= KEY_INC.steer; drInput.value = params.dr.toFixed(1); break;
      case 'e': params.dr += KEY_INC.steer; drInput.value = params.dr.toFixed(1); break;
      default: break;
    }
    updateDashboard();
  });

  // 车辆参数输入（回写后端）
  const pInputs = {
    m: document.getElementById('p_m'), Iz: document.getElementById('p_Iz'),
    a: document.getElementById('p_a'), b: document.getElementById('p_b'),
    kf: document.getElementById('p_kf'), kr: document.getElementById('p_kr'),
    mu: document.getElementById('p_mu'), g: document.getElementById('p_g'), U_min: document.getElementById('p_Umin'),
  };
  for (const key of Object.keys(pInputs)) {
    pInputs[key].addEventListener('change', () => {
      const val = parseFloat(pInputs[key].value);
      if (!Number.isNaN(val)) { params[key] = val; schedulePostParams(); updateDerivedStats(); }
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

  // 仿真循环
  function step() {
    if (running) { integrate(0.02); pushTrackPoint(); }
    if (view.follow) centerOnVehicle();
    draw();
    requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function resetSim() { state.x = 0; state.y = 0; state.psi = 0; state.beta = 0; state.r = 0; track = []; }

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

// 简化运动学（示意）
function integrate(dt) {
  const df = params.df * Math.PI / 180; const dr = params.dr * Math.PI / 180;
  const yawRate = (Math.tan(df) - Math.tan(dr)) * params.U * 0.02; // demo 公式
  state.psi += yawRate * dt;
  state.x += params.U * Math.cos(state.psi) * dt;
  state.y += params.U * Math.sin(state.psi) * dt;
  updateDashboard();
}

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
  document.getElementById('stat_u').textContent = params.U.toFixed(2);
  document.getElementById('stat_df').textContent = params.df.toFixed(1);
  document.getElementById('stat_dr').textContent = params.dr.toFixed(1);
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
function drawVehicle() {
  const x = worldToScreenX(state.x); const y = worldToScreenY(state.y);
  const w = 2.0 * view.scale; const h = 1.0 * view.scale;
  ctx.save(); ctx.translate(x, y); ctx.rotate(-state.psi);
  ctx.fillStyle = 'rgba(106,160,255,0.9)'; ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.rect(-w/2, -h/2, w, h); ctx.fill(); ctx.stroke(); ctx.restore();
  ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI*2); ctx.fill();
}
function draw() { const rect = canvas.getBoundingClientRect(); ctx.clearRect(0, 0, rect.width, rect.height); drawGrid(); drawTrack(); drawVehicle(); }

// 后端交互（节流）
let postTimer = null;
function schedulePostParams() {
  if (postTimer) clearTimeout(postTimer);
  postTimer = setTimeout(async () => {
    try {
      const patch = { m: params.m, Iz: params.Iz, a: params.a, b: params.b, kf: params.kf, kr: params.kr, U: params.U, mu: params.mu, g: params.g, U_min: params.U_min };
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
    for (const k of ['m','Iz','a','b','kf','kr','U','mu','g','U_min']) { if (typeof data[k] !== 'undefined') params[k] = data[k]; }
    // 更新输入框
    const setVal = (id, v) => { const el = document.getElementById(id); if (el) el.value = String(v); };
    setVal('speed_input', params.U.toFixed(2));
    setVal('df_input', params.df.toFixed(1));
    setVal('dr_input', params.dr.toFixed(1));
    setVal('p_m', params.m); setVal('p_Iz', params.Iz); setVal('p_a', params.a); setVal('p_b', params.b);
    setVal('p_kf', params.kf); setVal('p_kr', params.kr); setVal('p_mu', params.mu); setVal('p_g', params.g); setVal('p_Umin', params.U_min);
    // 更新派生量显示（也可从后端 data.L / data.K）
    const pL = document.getElementById('p_L'); const pK = document.getElementById('p_K');
    if (pL && typeof data.L !== 'undefined') pL.textContent = Number(data.L).toFixed(3);
    if (pK && typeof data.K !== 'undefined') pK.textContent = Number(data.K).toFixed(6);
    updateDashboard();
  } catch (e) { console.warn('GET /api/params 失败', e); setConnStatus(false); }
}

// 初始化
window.addEventListener('load', async () => { bindUI(); await loadParams(); updateDerivedStats(); });