from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple
import numpy as np

from .tire import (
    PacejkaParams,
    PacejkaLongParams,
    pacejka_lateral,
    pacejka_longitudinal,
    combine_friction_ellipse,
)


@dataclass
class Vehicle3DOF:
    """
    3-DOF nonlinear lateral dynamics model parameters.

    Coordinates and parameters follow documents/3dof.md.
    """
    m: float = 1500.0            # mass [kg]
    Iz: float = 2500.0           # yaw inertia [kg*m^2]
    a: float = 1.2               # CG to front axle [m]
    b: float = 1.6               # CG to rear axle [m]
    g: float = 9.81              # gravity [m/s^2]
    U_min: float = 0.5           # min longitudinal speed to avoid singularities [m/s]
    # Linear cornering stiffness for optional linear tire model
    kf: float = 1.6e5
    kr: float = 1.7e5
    # Tire model selection: 'pacejka' or 'linear'
    tire_model: str = 'pacejka'

    # Tire & steering
    tire_params_f: PacejkaParams = field(default_factory=PacejkaParams)
    tire_params_r: PacejkaParams = field(default_factory=PacejkaParams)
    n_sw: float = 16.0           # steering ratio (steering wheel to front wheel)

    # 4WS control (example): delta_r = -c1 * delta_f + c2 * r
    c1: float = 0.3
    c2: float = 0.05

    # Simplified yaw damping torque coefficient [N·m·s/rad]
    # Approximates self-aligning torque aggregate effect to avoid unbounded yaw acceleration
    yaw_damp: float = 300.0
    # Yaw-rate saturation gain (extra damping when |r| exceeds friction-limited bound)
    yaw_sat_gain: float = 4.0

    # Resistances (simplified; set to 0 by default)
    Fwx: float = 0.0
    Fwy: float = 0.0
    Fsx: float = 0.0
    Fsy: float = 0.0

    # Longitudinal tire parameters (pure slip) for combined condition
    tire_long_params_f: PacejkaLongParams = field(default_factory=PacejkaLongParams)
    tire_long_params_r: PacejkaLongParams = field(default_factory=PacejkaLongParams)

    @property
    def L(self) -> float:
        return self.a + self.b

    def static_loads(self) -> Tuple[float, float]:
        """Front/rear static vertical loads (ignoring load transfer)."""
        # Static distribution by distances
        Fzf = self.m * self.g * (self.b / self.L)
        Fzr = self.m * self.g * (self.a / self.L)
        return Fzf, Fzr


@dataclass
class State3DOF:
    vx: float = 20.0   # longitudinal speed [m/s]
    vy: float = 0.0    # lateral speed [m/s]
    r: float = 0.0     # yaw rate [rad/s]
    x: float = 0.0     # X position [m]
    y: float = 0.0     # Y position [m]
    psi: float = 0.0   # yaw angle [rad]


def control_4ws(delta_sw: float, r: float, vp: Vehicle3DOF) -> Tuple[float, float]:
    """Compute front and rear wheel angles from steering wheel angle and yaw rate."""
    df = delta_sw / vp.n_sw
    dr = -vp.c1 * df + vp.c2 * r
    return df, dr


def slip_angles(vx: float, vy: float, r: float, df: float, dr: float, vp: Vehicle3DOF) -> Tuple[float, float]:
    """Compute front/rear slip angles alpha_f, alpha_r (bicycle model)."""
    vx_eff = max(vp.U_min, vx)
    alpha_f = np.arctan2(vy + vp.a * r, vx_eff) - df
    alpha_r = np.arctan2(vy - vp.b * r, vx_eff) - dr
    return alpha_f, alpha_r


def tire_forces(alpha_f: float, alpha_r: float, vp: Vehicle3DOF) -> Tuple[float, float]:
    """Compute lateral tire forces Fy front/rear based on selected model."""
    Fzf, Fzr = vp.static_loads()
    if (vp.tire_model or '').lower() == 'linear':
        # Linear bicycle model lateral forces
        Fy_f = -vp.kf * float(alpha_f)
        Fy_r = -vp.kr * float(alpha_r)
    else:
        # Default: Pacejka magic formula
        Fy_f = pacejka_lateral(alpha_f, Fzf, vp.tire_params_f)
        Fy_r = pacejka_lateral(alpha_r, Fzr, vp.tire_params_r)
    return Fy_f, Fy_r


def derivatives(s: State3DOF, delta_sw: float, vp: Vehicle3DOF) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute time derivatives for the 3-DOF nonlinear model.

    Returns
    - ds: np.array([vx_dot, vy_dot, r_dot, x_dot, y_dot, psi_dot])
    - aux: dict of auxiliary values for logging/analysis
    """
    df, dr = control_4ws(delta_sw, s.r, vp)
    alpha_f, alpha_r = slip_angles(s.vx, s.vy, s.r, df, dr, vp)
    Fy_f, Fy_r = tire_forces(alpha_f, alpha_r, vp)

    # Longitudinal slip ratios (defaults zero if not provided externally)
    # Placeholders; actual values should be provided by simulate() via closure
    lmbd_f = derivatives._lambda_f(s)
    lmbd_r = derivatives._lambda_r(s)

    # Pure forces
    Fx_f_pure = pacejka_longitudinal(lmbd_f, vp.static_loads()[0], vp.tire_long_params_f)
    Fx_r_pure = pacejka_longitudinal(lmbd_r, vp.static_loads()[1], vp.tire_long_params_r)

    Fy_f_pure = Fy_f
    Fy_r_pure = Fy_r

    # Combine via friction ellipse per axle
    Fx_f, Fy_f = combine_friction_ellipse(
        Fx_f_pure, Fy_f_pure, vp.static_loads()[0], vp.tire_long_params_f.mu_x, vp.tire_params_f.mu_y
    )
    Fx_r, Fy_r = combine_friction_ellipse(
        Fx_r_pure, Fy_r_pure, vp.static_loads()[1], vp.tire_long_params_r.mu_x, vp.tire_params_r.mu_y
    )

    # Dynamics (documents/3dof.md)
    vx_dot = (1.0 / vp.m) * (
        Fx_f * np.cos(df) - Fy_f * np.sin(df) + Fx_r * np.cos(dr) - Fy_r * np.sin(dr)
        - vp.Fwx - vp.Fsx
    ) + s.r * s.vy

    vy_dot = (1.0 / vp.m) * (
        Fx_f * np.sin(df) + Fy_f * np.cos(df) + Fx_r * np.sin(dr) + Fy_r * np.cos(dr)
        + vp.Fwy + vp.Fsy
    ) - s.r * s.vx

    Mz = (
        (Fx_f * np.sin(df) + Fy_f * np.cos(df)) * vp.a
        - (Fx_r * np.sin(dr) + Fy_r * np.cos(dr)) * vp.b
    )
    # Add yaw damping torque to approximate self-aligning torque effects
    r_dot = (Mz - vp.yaw_damp * s.r) / vp.Iz

    # Friction-limited yaw-rate bound: |r| ≤ mu_y * g / U
    vx_eff = max(vp.U_min, s.vx)
    mu_y = min(vp.tire_params_f.mu_y, vp.tire_params_r.mu_y)
    r_max = mu_y * vp.g / vx_eff
    if abs(s.r) > r_max:
        r_dot -= vp.yaw_sat_gain * (abs(s.r) - r_max) * np.sign(s.r)

    psi_dot = s.r
    x_dot = s.vx * np.cos(s.psi) - s.vy * np.sin(s.psi)
    y_dot = s.vx * np.sin(s.psi) + s.vy * np.cos(s.psi)

    ds = np.array([vx_dot, vy_dot, r_dot, x_dot, y_dot, psi_dot], dtype=float)
    ay = vy_dot + s.r * s.vx
    aux = {
        "df": df,
        "dr": dr,
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
        "Fy_f": Fy_f,
        "Fy_r": Fy_r,
        "Fx_f": Fx_f,
        "Fx_r": Fx_r,
        "ay": ay,
    }
    return ds, aux


def simulate(
    T: float,
    dt: float,
    vp: Vehicle3DOF,
    s0: State3DOF,
    delta_sw_fn: Callable[[float], float],
    lambda_f_fn: Callable[[float, State3DOF], float] | None = None,
    lambda_r_fn: Callable[[float, State3DOF], float] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Run time-domain simulation for given steering wheel input function.

    Outputs arrays for t, vx, vy, r, x, y, psi, df, dr, Fy_f, Fy_r, alpha_f, alpha_r.
    """
    N = int(np.floor(T / dt)) + 1
    t = np.linspace(0.0, T, N)

    vx = np.empty(N)
    vy = np.empty(N)
    r = np.empty(N)
    x = np.empty(N)
    y = np.empty(N)
    psi = np.empty(N)

    df = np.empty(N)
    dr = np.empty(N)
    Fy_f = np.empty(N)
    Fy_r = np.empty(N)
    Fx_f = np.empty(N)
    Fx_r = np.empty(N)
    alpha_f = np.empty(N)
    alpha_r = np.empty(N)
    ay = np.empty(N)

    s = State3DOF(**s0.__dict__)

    # Provide longitudinal slip closure to derivatives
    def _lambda_f(s: State3DOF, ti: float) -> float:
        if lambda_f_fn is None:
            return 0.0
        return float(lambda_f_fn(ti, s))

    def _lambda_r(s: State3DOF, ti: float) -> float:
        if lambda_r_fn is None:
            return 0.0
        return float(lambda_r_fn(ti, s))

    derivatives._lambda_f = lambda s_obj: _lambda_f(s_obj, ti_current[0])
    derivatives._lambda_r = lambda s_obj: _lambda_r(s_obj, ti_current[0])

    ti_current = [0.0]

    for i, ti in enumerate(t):
        ti_current[0] = ti
        delta_sw = delta_sw_fn(ti)
        ds, aux = derivatives(s, delta_sw, vp)

        # Store
        vx[i], vy[i], r[i], x[i], y[i], psi[i] = s.vx, s.vy, s.r, s.x, s.y, s.psi
        df[i], dr[i] = aux["df"], aux["dr"]
        Fy_f[i], Fy_r[i] = aux["Fy_f"], aux["Fy_r"]
        Fx_f[i], Fx_r[i] = aux["Fx_f"], aux["Fx_r"]
        alpha_f[i], alpha_r[i] = aux["alpha_f"], aux["alpha_r"]
        ay[i] = aux["ay"]

        # Integrate: explicit Euler (small dt recommended)
        s.vx += ds[0] * dt
        s.vy += ds[1] * dt
        s.r += ds[2] * dt
        s.x += ds[3] * dt
        s.y += ds[4] * dt
        s.psi += ds[5] * dt

    return {
        "t": t,
        "vx": vx,
        "vy": vy,
        "r": r,
        "x": x,
        "y": y,
        "psi": psi,
        "df": df,
        "dr": dr,
        "Fx_f": Fx_f,
        "Fx_r": Fx_r,
        "Fy_f": Fy_f,
        "Fy_r": Fy_r,
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
        "ay": ay,
    }