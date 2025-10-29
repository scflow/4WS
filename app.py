from flask import Flask, jsonify, request
from flask import send_from_directory
from src.params import VehicleParams
from src.sim import SimEngine
import numpy as np
import os

app = Flask(__name__, static_folder='web', static_url_path='')

# 全局车辆参数实例（可持久化于内存）
VP = VehicleParams()
ENGINE = SimEngine(VP, dt=0.02)

# 静态首页与静态资源
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_proxy(path: str):
    return send_from_directory(app.static_folder, path)

# 获取/更新车辆参数
@app.route('/api/params', methods=['GET', 'POST'])
def api_params():
    if request.method == 'GET':
        return jsonify(VP.to_dict())
    # 更新参数（允许部分字段）
    data = request.get_json(force=True) or {}
    for k in ['m', 'Iz', 'a', 'b', 'width', 'track', 'kf', 'kr', 'U', 'mu', 'g', 'U_min']:
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
    return jsonify({'ok': True})

# 仿真模式：2DOF / 3DOF
@app.route('/api/mode', methods=['GET', 'POST'])
def api_mode():
    if request.method == 'GET':
        return jsonify({'mode': ENGINE.get_ctrl().get('mode', '2dof')})
    data = request.get_json(force=True) or {}
    mode = data.get('mode')
    ENGINE.set_mode(mode if isinstance(mode, str) else '2dof')
    return jsonify({'mode': ENGINE.get_ctrl().get('mode', '2dof')})


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