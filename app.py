from flask import Flask, jsonify, request
from flask import send_from_directory
from src.params import VehicleParams

app = Flask(__name__, static_folder='web', static_url_path='')

# 全局车辆参数实例（可持久化于内存）
VP = VehicleParams()

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
    for k in ['m', 'Iz', 'a', 'b', 'kf', 'kr', 'U', 'mu', 'g', 'U_min']:
        if k in data:
            try:
                setattr(VP, k, float(data[k]))
            except (TypeError, ValueError):
                pass
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)