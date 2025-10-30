import numpy as np
from typing import List, Dict, Tuple


def solve_quintic(p0: float, v0: float, a0: float,
                  pT: float, vT: float, aT: float,
                  T: float) -> np.ndarray:
    """
    Solve a quintic polynomial coefficients for boundary conditions at t=0 and t=T.
    p(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5
    """
    T = float(T)
    M = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3],
    ], dtype=float)
    y = np.array([p0, v0, a0, pT, vT, aT], dtype=float)
    coeffs = np.linalg.solve(M, y)
    return coeffs


def sample_poly(coeffs: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample polynomial, first and second derivatives at t points."""
    a0, a1, a2, a3, a4, a5 = coeffs
    p = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    v = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    a = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    return p, v, a


def plan_quintic_xy(start: Dict[str, float], end: Dict[str, float],
                    T: float, N: int,
                    U_start: float) -> List[Dict[str, float]]:
    """
    Plan a 2D path using independent quintic polynomials in x and y.
    - start: {x, y, psi} psi in radians
    - end: {x, y, psi} psi in radians
    - U_start: initial speed magnitude (m/s)
    Returns list of {t, x, y, psi}
    """
    x0, y0, psi0 = float(start['x']), float(start['y']), float(start['psi'])
    xT, yT, psiT = float(end['x']), float(end['y']), float(end['psi'])

    # Initial/terminal velocities projected onto x/y axes
    vx0 = U_start * np.cos(psi0)
    vy0 = U_start * np.sin(psi0)
    # Terminal velocity magnitude uses same U_start for simplicity here
    vxT = U_start * np.cos(psiT)
    vyT = U_start * np.sin(psiT)

    ax0 = 0.0
    ay0 = 0.0
    axT = 0.0
    ayT = 0.0

    cx = solve_quintic(x0, vx0, ax0, xT, vxT, axT, T)
    cy = solve_quintic(y0, vy0, ay0, yT, vyT, ayT, T)

    t = np.linspace(0.0, T, int(N))
    x, vx, _ = sample_poly(cx, t)
    y, vy, _ = sample_poly(cy, t)
    psi = np.arctan2(vy, vx)

    plan = [
        {
            't': float(tt),
            'x': float(xx),
            'y': float(yy),
            'psi': float(ppsi),
        }
        for tt, xx, yy, ppsi in zip(t, x, y, psi)
    ]
    return plan