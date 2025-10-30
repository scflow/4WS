from flask import Flask, jsonify, request
from flask import send_from_directory
from src.params import VehicleParams
from src.sim import SimEngine
from src.config import load_config, apply_config, current_config, save_config
from src.planner import plan_quintic_xy
import numpy as np
import os

app = Flask(__name__, static_folder='web', static_url_path='')

# 启动时加载配置文件并应用
def _resolve_cfg_path() -> str:
    env_path = os.environ.get("SIM_CONFIG_PATH")
    if env_path:
        return env_path
    # 优先使用 YAML（若存在），否则回退 JSON
    yaml_path = os.path.join(os.getcwd(), "config.yaml")
    json_path = os.path.join(os.getcwd(), "config.json")
    if os.path.exists(yaml_path):
        return yaml_path
    return json_path

CFG_PATH = _resolve_cfg_path()
_CFG = load_config(CFG_PATH)
VP = VehicleParams()
ENGINE = SimEngine(VP, dt=float((_CFG.get("control") or {}).get("dt", 0.02)))
apply_config(_CFG, VP, ENGINE)

# 静态首页与静态资源
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)

# 获取/更新车辆参数
@app.route('/api/params', methods=['GET', 'POST', 'PATCH'])
def api_params():
    if request.method == 'GET':
        return jsonify(VP.to_dict())
    data = request.get_json(force=True) or {}
    if request.method == 'POST':
        # 兼容旧参数平铺结构
        for k in ['m', 'Iz', 'a', 'b', 'width', 'track', 'kf', 'kr', 'U', 'mu', 'g', 'U_min', 'U_blend']:
            if k in data:
                try:
                    setattr(VP, k, float(data[k]))
                except (TypeError, ValueError):
                    pass
        # 非数值参数：轮胎模型选择
        if 'tire_model' in data:
            v = str(data['tire_model']).lower().strip()
            if v in ('pacejka', 'linear'):
                VP.tire_model = v
        # 若更新了 U，则同步到控制
        ENGINE.set_ctrl(U=VP.U)
        # 持久化到配置文件
        save_config(CFG_PATH, current_config(VP, ENGINE))
        return jsonify(VP.to_dict())
    else:
        # PATCH：支持嵌套结构（tire/control），并兼容平铺更新
        # 平铺部分（与 POST 相同）
        for k in ['m', 'Iz', 'a', 'b', 'width', 'track', 'kf', 'kr', 'U', 'mu', 'g', 'U_min', 'U_blend']:
            if k in data:
                try:
                    setattr(VP, k, float(data[k]))
                except (TypeError, ValueError):
                    pass
        if 'tire_model' in data:
            v = str(data['tire_model']).lower().strip()
            if v in ('pacejka', 'linear'):
                VP.tire_model = v

        # 嵌套：tire
        tire = data.get('tire') or {}
        if isinstance(tire, dict):
            # model
            model = tire.get('model')
            if isinstance(model, str):
                v = model.lower().strip()
                if v in ('pacejka', 'linear'):
                    VP.tire_model = v
            # lateral.mu_y
            lat = tire.get('lateral') or {}
            if isinstance(lat, dict) and 'mu_y' in lat:
                try:
                    VP.mu = float(lat['mu_y'])
                except (TypeError, ValueError):
                    pass
            # longitudinal.mu_x（当前与 mu 统一映射，保留字段以兼容扩展）
            lon = tire.get('longitudinal') or {}
            if isinstance(lon, dict) and 'mu_x' in lon:
                try:
                    # 简化：先与 mu 同步（未来可拆分为独立字段）
                    VP.mu = float(lon['mu_x'])
                except (TypeError, ValueError):
                    pass

        # 嵌套：control（仿真控制/配置）
        control = data.get('control') or {}
        if isinstance(control, dict):
            for k in ['k_v', 'tau_ctrl', 'tau_low', 'tau_beta', 'yaw_damp', 'yaw_sat_gain', 'drive_bias_front', 'drive_bias_rear']:
                if k in control:
                    try:
                        val = float(control[k])
                        if k == 'k_v':
                            ENGINE.k_v = val
                        elif k == 'tau_ctrl':
                            ENGINE.tau_ctrl = val
                        elif k == 'tau_low':
                            ENGINE.tau_low = val
                        elif k == 'tau_beta':
                            ENGINE.tau_beta = val
                        elif k == 'yaw_damp':
                            ENGINE.yaw_damp = val
                        elif k == 'yaw_sat_gain':
                            ENGINE.yaw_sat_gain = val
                        elif k == 'drive_bias_front':
                            ENGINE.drive_bias_front = val
                        elif k == 'drive_bias_rear':
                            ENGINE.drive_bias_rear = val
                    except (TypeError, ValueError):
                        pass

        # 若更新了 U，则同步到控制
        ENGINE.set_ctrl(U=VP.U)
        # 持久化到配置文件
        save_config(CFG_PATH, current_config(VP, ENGINE))
        return jsonify(VP.to_dict())

# 派生量与横摆率指令（基于当前参数与输入前轮转角）
@app.route('/api/derived', methods=['GET'])
def api_derived():
    try:
        df = float(request.args.get('delta_f', 0.0))
    except (TypeError, ValueError):
        df = 0.0
    return jsonify({
        'K': VP.understeer_gradient(),
        'r_ref': VP.r_ref(df),
        'r_cmd': VP.yaw_rate_cmd(df),
    })

# 状态查询（x, y, psi, beta, r）
@app.route('/api/state', methods=['GET'])
def api_state():
    return jsonify(ENGINE.get_state())

# 控制量查询/更新（U, df, dr, running）
@app.route('/api/control', methods=['GET', 'POST'])
def api_control():
    if request.method == 'GET':
        return jsonify(ENGINE.get_ctrl())
    data = request.get_json(force=True) or {}
    # 角度限幅（更符合直觉的默认范围，单位：度）
    ANGLE_LIMIT_DEG = 30.0
    ANGLE_LIMIT_RAD = np.deg2rad(ANGLE_LIMIT_DEG)

    # 解析角度输入：优先 df_deg/dr_deg；其次 df_rad/dr_rad；最后 df/dr（按“度”理解）
    def parse_angle_rad(d: dict, base: str):
        val = d.get(f"{base}_deg")
        if val is not None:
            try:
                return float(val) * np.pi / 180.0
            except (TypeError, ValueError):
                return None
        val = d.get(f"{base}_rad")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        val = d.get(base)
        if val is not None:
            try:
                # 为直觉一致，df/dr 默认为度
                return float(val) * np.pi / 180.0
            except (TypeError, ValueError):
                return None
        return None

    # 更新 U、df、dr，并做角度限幅
    U = data.get('U')
    df_rad = parse_angle_rad(data, 'df')
    dr_rad = parse_angle_rad(data, 'dr')
    if df_rad is not None:
        df_rad = float(np.clip(df_rad, -ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD))
    if dr_rad is not None:
        dr_rad = float(np.clip(dr_rad, -ANGLE_LIMIT_RAD, ANGLE_LIMIT_RAD))

    ENGINE.set_ctrl(U=U if U is not None else None,
                    df=df_rad,
                    dr=dr_rad)
    # 同步到参数的 U（用于派生量计算）
    if U is not None:
        try:
            VP.U = float(U)
        except (TypeError, ValueError):
            pass
    # 持久化到配置文件
    save_config(CFG_PATH, current_config(VP, ENGINE))
    return jsonify(ENGINE.get_ctrl())

# 仿真开始/暂停
@app.route('/api/sim/start_pause', methods=['POST'])
def api_start_pause():
    data = request.get_json(force=True) or {}
    running = data.get('running')
    if isinstance(running, bool):
        if running:
            ENGINE.start()
        else:
            ENGINE.pause()
    else:
        running = ENGINE.toggle()
    return jsonify({'running': bool(running)})

# 仿真重置
@app.route('/api/sim/reset', methods=['POST'])
def api_reset():
    ENGINE.reset()
    return jsonify({'ok': True})

# 设置初始位姿
@app.route('/api/init_pose', methods=['POST'])
def api_init_pose():
    data = request.get_json(force=True) or {}
    def parse_float(name, default=0.0):
        try:
            return float(data.get(name, default))
        except (TypeError, ValueError):
            return default
    x0 = parse_float('x', 0.0)
    y0 = parse_float('y', 0.0)
    psi = data.get('psi')
    # 接收度或弧度
    psi_rad = None
    try:
        v = float(psi)
        psi_rad = v * np.pi / 180.0 if abs(v) > np.pi else v
    except (TypeError, ValueError):
        psi_rad = 0.0
    ENGINE.set_init_pose(x0, y0, psi_rad)
    return jsonify({'ok': True})

# 轨迹查询与设置
@app.route('/api/track', methods=['GET'])
def api_track():
    return jsonify({'points': ENGINE.get_track()})

@app.route('/api/track/settings', methods=['POST'])
def api_track_settings():
    data = request.get_json(force=True) or {}
    enabled = data.get('enabled')
    keep = data.get('retentionSec')
    maxp = data.get('maxPoints')
    ENGINE.set_track_settings(enabled=enabled, retention_sec=keep, max_points=maxp)
    # 持久化到配置文件
    save_config(CFG_PATH, current_config(VP, ENGINE))
    return jsonify({'ok': True})

# 参考规划轨迹查询（points: [{t,x,y,psi}]）
@app.route('/api/plan', methods=['GET'])
def api_plan_get():
    try:
        pts = getattr(ENGINE, 'plan', []) or []
        return jsonify({'points': pts})
    except Exception as e:
        return jsonify({'error': str(e), 'points': []}), 200

# 五次多项式规划：起点-终点（含航向角）生成参考轨迹
@app.route('/api/plan/quintic', methods=['POST'])
def api_plan_quintic():
    data = request.get_json(force=True) or {}

    def parse_pose(obj, fallback=None):
        if not isinstance(obj, dict):
            return fallback or {'x': 0.0, 'y': 0.0, 'psi': 0.0}
        def parse_val(k, d=0.0):
            try:
                return float(obj.get(k, d))
            except (TypeError, ValueError):
                return d
        x = parse_val('x', (fallback or {}).get('x', 0.0))
        y = parse_val('y', (fallback or {}).get('y', 0.0))
        psi_raw = obj.get('psi', (fallback or {}).get('psi', 0.0))
        psi_rad = 0.0
        try:
            v = float(psi_raw)
            psi_rad = v * np.pi / 180.0 if abs(v) > np.pi else v
        except (TypeError, ValueError):
            psi_rad = float((fallback or {}).get('psi', 0.0))
        return {'x': x, 'y': y, 'psi': psi_rad}

    # 起点：优先使用传入的 start，否则用当前仿真状态
    st = ENGINE.get_state()
    start_default = {'x': float(st['x']), 'y': float(st['y']), 'psi': float(st['psi'])}
    start = parse_pose(data.get('start'), start_default)

    # 终点：必须传入
    end = parse_pose(data.get('end'), None)
    if end is None:
        return jsonify({'error': 'missing end pose'}), 400

    # 规划时长与采样数
    # 规划时长 T 已移除：后端自动估算（不再从请求读取）
    dist = float(np.hypot(end['x'] - start['x'], end['y'] - start['y']))
    U = float(VP.U)
    U = max(0.3, abs(U))
    T = max(1.0, dist / U)
    T = min(T, 30.0)
    try:
        N = int(data.get('N', 200))
    except (TypeError, ValueError):
        N = 200
    N = max(20, N)

    # 规划
    plan = plan_quintic_xy(start, end, T, N, U_start=float(VP.U))
    ENGINE.load_plan(plan)
    return jsonify({'ok': True, 'N': N, 'count': len(plan)})

# 自动跟踪开关/查询（根据参考轨迹沿路输出前后轮角）
@app.route('/api/autop', methods=['GET', 'POST'])
def api_autop():
    if request.method == 'GET':
        return jsonify({'enabled': bool(ENGINE.autop_enabled), 'mode': getattr(ENGINE, 'autop_mode', 'simple'), 'plan_count': len(getattr(ENGINE, 'plan', []) or [])})
    data = request.get_json(force=True) or {}
    enabled = bool(data.get('enabled', True))
    ENGINE.set_autop(enabled)
    # 可选：切换模式（'simple' 或 'mpc'）
    mode = data.get('mode')
    if isinstance(mode, str):
        try:
            ENGINE.set_autop_mode(mode)
        except Exception:
            pass
    # 可选：启动仿真
    if bool(data.get('start', False)):
        ENGINE.start()
    return jsonify({'enabled': bool(ENGINE.autop_enabled), 'mode': getattr(ENGINE, 'autop_mode', 'simple')})

# 仿真模式：2DOF / 3DOF
@app.route('/api/mode', methods=['GET', 'POST'])
def api_mode():
    if request.method == 'GET':
        return jsonify({'mode': ENGINE.get_ctrl().get('mode', '2dof')})
    data = request.get_json(force=True) or {}
    mode = data.get('mode')
    ENGINE.set_mode(mode if isinstance(mode, str) else '2dof')
    # 持久化到配置文件
    save_config(CFG_PATH, current_config(VP, ENGINE))
    return jsonify({'mode': ENGINE.get_ctrl().get('mode', '2dof')})

# 配置查询（完整 vehicle/control 配置）
@app.route('/api/config', methods=['GET'])
def api_config_get():
    try:
        return jsonify(current_config(VP, ENGINE))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        # 允许通过环境变量 PORT 指定端口，默认改为 18000，避免 8000 被占用
        port = int(os.environ.get('PORT', '18000'))
        app.run(host='0.0.0.0', port=port, debug=True)
    finally:
        # 调试退出时关闭引擎线程
        try:
            ENGINE.shutdown()
        except Exception:
            pass