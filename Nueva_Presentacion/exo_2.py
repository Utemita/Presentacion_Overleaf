# ============================================================
# FRAMEWORK COMPLETO
# OPTIMIZACIÓN BIOMECÁNICA DE EXOESQUELETO
# MECANISMO COMPUESTO:
#
# 1) PRIMER 5 BARRAS
# 2) PRIMER 4 BARRAS
# 3) SISTEMA AUXILIAR
# 4) SEGUNDO 5 BARRAS
# 5) SEGUNDO 4 BARRAS
# 6) FALANGES
#
# MODELO BASADO EN:
# - DOCUMENTO TESIS
# - ECUACIONES COMPLETAS
# - CONFIGURACIÓN ABIERTA
# - CONTINUIDAD ANGULAR
#
# AUTOR: CHATGPT
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import differential_evolution

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def wrap_to_pi(angle):

    return (angle + np.pi) % (2*np.pi) - np.pi


def rms_error(x1, y1, x2, y2):

    return np.sqrt(
        np.mean(
            (x1 - x2)**2 +
            (y1 - y2)**2
        )
    )

# ============================================================
# SOLUCIÓN 5 BARRAS
# ============================================================

def solve_five_bar(
    r1,
    r2,
    r3,
    r4,
    r5,
    theta1,
    theta2
):

    den = (
        r4*np.cos(theta2)
        -
        r1*np.cos(theta1)
        +
        2*r3
    )

    if np.abs(den) < 1e-8:
        return None

    e = (
        r1*np.sin(theta1)
        -
        r4*np.sin(theta2)
    ) / den

    temp = (
        2*(
            r1*r3*np.cos(theta1)
            +
            r3*r4*np.cos(theta2)
        )
        -
        r1**2
        +
        r2**2
        +
        r4**2
        -
        r5**2
    )

    f = temp / (2*den)

    d = e**2 + 1

    g = 2*(
        e*f
        -
        e*r1*np.cos(theta1)
        +
        e*r3
        -
        r1*np.sin(theta1)
    )

    h = (
        f**2
        -
        2*f*(r1*np.cos(theta1)-r3)
        -
        2*r1*r3*np.cos(theta1)
        +
        r1**2
        +
        r3**2
        -
        r2**2
    )

    disc = g**2 - 4*d*h

    if disc < 0:
        return None

    py = (
        -g + np.sqrt(disc)
    ) / (2*d)

    px = e*py + f

    return px, py

# ============================================================
# SOLUCIÓN 4 BARRAS
# CONFIGURACIÓN ABIERTA
# ============================================================

def solve_four_bar(
    a,
    b,
    c,
    d,
    theta2,
    theta1
):

    k1 = a*np.cos(theta2) + d*np.cos(theta1)
    k2 = a*np.sin(theta2) + d*np.sin(theta1)

    k3 = (
        k1**2 +
        k2**2 +
        c**2 -
        b**2
    )

    A1 = -2*k1*c - k3
    B1 = 4*k2*c
    C1 = 2*k1*c - k3

    disc = B1**2 - 4*A1*C1

    if disc < 0:
        return None

    tan_theta4 = (
        -B1 -
        np.sqrt(disc)
    ) / (2*A1)

    theta4 = 2*np.arctan(tan_theta4)

    return theta4

# ============================================================
# MODELO CINEMÁTICO COMPLETO
# ============================================================

def full_model(
    p,
    theta_input
):

    (
        Bancada1,
        Bancada2,

        Link1,
        Link2,
        Link3,
        Link4,
        Link5,
        Link6,
        Link7,
        Link8,

        hsp,
        dsp,

        Fp,
        Fm,
        Fd,

        theta_aux_fm,
        theta_aux_fd,

        gear_ratio,

        theta1_offset

    ) = p

    # ========================================================
    # ASIGNACIÓN
    # ========================================================

    r4 = Link1
    r5 = Link2
    r2 = Link3
    r1 = Link4
    r3 = Bancada1 / 2

    a = Link4
    b = Link5

    c = np.sqrt(hsp**2 + dsp**2)

    d = Bancada2

    # ========================================================
    # SEGUNDO 5 BARRAS
    # ========================================================

    r1m2 = Fp - 2*dsp
    r2m2 = Link7
    r3m2 = Link5 / 2
    r4m2 = Link3
    r5m2 = Link6

    # ========================================================
    # SEGUNDO 4 BARRAS
    # ========================================================

    a2 = Link7
    b2 = Link8
    c2 = 0.046
    d2 = np.sqrt(hsp**2 + dsp**2)

    # ========================================================
    # ALMACENAMIENTO
    # ========================================================

    PXifp = []
    PYifp = []

    PXifd = []
    PYifd = []

    PXtip = []
    PYtip = []

    # ========================================================
    # ITERACIÓN
    # ========================================================

    for th2 in theta_input:

        th1 = (
            th2 / gear_ratio
            +
            theta1_offset
        )

        # ====================================================
        # PRIMER 5 BARRAS
        # ====================================================

        res5 = solve_five_bar(
            r1,
            r2,
            r3,
            r4,
            r5,
            th1,
            th2
        )

        if res5 is None:
            return None

        pxP, pyP = res5

        # ====================================================
        # PRIMER 4 BARRAS
        # ====================================================

        theta14B = np.pi/2

        theta4 = solve_four_bar(
            a,
            b,
            c,
            d,
            th1,
            theta14B
        )

        if theta4 is None:
            return None

        # ====================================================
        # FALANGE PROXIMAL
        # ====================================================

        theta_ps1 = np.arctan2(
            hsp,
            dsp
        )

        theta_fp = theta4 + theta_ps1

        px_ifp = (
            Fp*np.cos(theta_fp)
            -
            d*np.cos(th1)
            -
            r3
        )

        py_ifp = (
            Fp*np.sin(theta_fp)
            -
            d*np.sin(th1)
        )

        # ====================================================
        # SOPORTES S1
        # ====================================================

        pxs1 = (
            c*np.cos(theta4)
            -
            d*np.cos(th1)
            -
            r3
        )

        pys1 = (
            c*np.sin(theta4)
            -
            d*np.sin(th1)
        )

        # ====================================================
        # SOPORTES S2
        # ====================================================

        rs2 = np.sqrt(
            hsp**2 +
            (Fp - dsp)**2
        )

        theta_aux_s2 = np.arctan2(
            hsp,
            Fp - dsp
        )

        theta_ps2 = theta_fp - theta_aux_s2

        pxs2 = (
            rs2*np.cos(theta_ps2)
            -
            d*np.cos(th1)
            -
            r3
        )

        pys2 = (
            rs2*np.sin(theta_ps2)
            -
            d*np.sin(th1)
        )

        # ====================================================
        # PUNTO M4
        # ====================================================

        pxm4 = (
            a*np.cos(th2)
            -
            r3
        )

        pym4 = (
            a*np.sin(th2)
        )

        # ====================================================
        # SEGUNDO SISTEMA
        # ====================================================

        theta_roll = np.arctan2(
            pym4 - pys1,
            pxm4 - pxs1
        )

        theta1m2so = np.arctan2(
            pys2 - pys1,
            pxs2 - pxs1
        )

        theta1m2 = (
            theta1m2so -
            theta_roll
        )

        theta2m2so = np.arctan2(
            pyP - pym4,
            pxP - pxm4
        )

        theta2m2 = (
            theta2m2so -
            theta_roll
        )

        # ====================================================
        # SEGUNDO 5 BARRAS
        # ====================================================

        res5_2 = solve_five_bar(
            r1m2,
            r2m2,
            r3m2,
            r4m2,
            r5m2,
            theta1m2,
            theta2m2
        )

        if res5_2 is None:
            return None

        px_local, py_local = res5_2

        # ====================================================
        # TRASLACIÓN
        # ====================================================

        px_aux = (pxs1 + pxm4)/2
        py_aux = (pys1 + pym4)/2

        mag = np.sqrt(
            px_local**2 +
            py_local**2
        )

        theta_local = np.arctan2(
            py_local,
            px_local
        )

        pxP2 = (
            mag*np.cos(
                theta_local + theta_roll
            )
            +
            px_aux
        )

        pyP2 = (
            mag*np.sin(
                theta_local + theta_roll
            )
            +
            py_aux
        )

        # ====================================================
        # SEGUNDO 4 BARRAS
        # ====================================================

        theta1m42 = np.arctan2(
            pys2 - py_ifp,
            pxs2 - px_ifp
        )

        theta2m42 = np.arctan2(
            pyP2 - pys2,
            pxP2 - pxs2
        )

        theta4m2 = solve_four_bar(
            a2,
            b2,
            c2,
            d2,
            theta2m42,
            theta1m42
        )

        if theta4m2 is None:
            return None

        # ====================================================
        # FALANGE MEDIAL
        # ====================================================

        theta_fm = (
            theta4m2 +
            theta_aux_fm
        )

        px_ifd = (
            Fm*np.cos(theta_fm)
            +
            px_ifp
        )

        py_ifd = (
            Fm*np.sin(theta_fm)
            +
            py_ifp
        )

        # ====================================================
        # FALANGE DISTAL
        # ====================================================

        theta_fd = (
            theta_fm +
            theta_aux_fd
        )

        px_tip = (
            Fd*np.cos(theta_fd)
            +
            px_ifd
        )

        py_tip = (
            Fd*np.sin(theta_fd)
            +
            py_ifd
        )

        # ====================================================
        # GUARDAR
        # ====================================================

        PXifp.append(px_ifp)
        PYifp.append(py_ifp)

        PXifd.append(px_ifd)
        PYifd.append(py_ifd)

        PXtip.append(px_tip)
        PYtip.append(py_tip)

    return {

        'proximal': (
            np.array(PXifp),
            np.array(PYifp)
        ),

        'medial': (
            np.array(PXifd),
            np.array(PYifd)
        ),

        'tip': (
            np.array(PXtip),
            np.array(PYtip)
        )
    }

# ============================================================
# MOCAP ARTIFICIAL BIOMECÁNICO
# ============================================================

N = 120

theta_input = np.linspace(
    0,
    np.deg2rad(85),
    N
)

# FLEXIÓN BIOMECÁNICA

theta_mcp = np.deg2rad(
    55*np.sin(theta_input)
)

theta_pip = np.deg2rad(
    75*np.sin(theta_input)**1.2
)

theta_dip = 0.67*theta_pip

# LONGITUDES REALES

Fp_real = 0.050
Fm_real = 0.030
Fd_real = 0.022

# CINEMÁTICA DIRECTA

x_ifp = Fp_real*np.cos(theta_mcp)
y_ifp = Fp_real*np.sin(theta_mcp)

x_ifd = (
    x_ifp
    +
    Fm_real*np.cos(
        theta_mcp + theta_pip
    )
)

y_ifd = (
    y_ifp
    +
    Fm_real*np.sin(
        theta_mcp + theta_pip
    )
)

x_tip = (
    x_ifd
    +
    Fd_real*np.cos(
        theta_mcp +
        theta_pip +
        theta_dip
    )
)

y_tip = (
    y_ifd
    +
    Fd_real*np.sin(
        theta_mcp +
        theta_pip +
        theta_dip
    )
)

mocap = {

    'proximal': (
        x_ifp,
        y_ifp
    ),

    'medial': (
        x_ifd,
        y_ifd
    ),

    'tip': (
        x_tip,
        y_tip
    )
}

# ============================================================
# FITNESS
# ============================================================

def fitness(p):

    res = full_model(
        p,
        theta_input
    )

    if res is None:
        return 1e6

    e1 = rms_error(
        mocap['proximal'][0],
        mocap['proximal'][1],
        res['proximal'][0],
        res['proximal'][1]
    )

    e2 = rms_error(
        mocap['medial'][0],
        mocap['medial'][1],
        res['medial'][0],
        res['medial'][1]
    )

    e3 = rms_error(
        mocap['tip'][0],
        mocap['tip'][1],
        res['tip'][0],
        res['tip'][1]
    )

    penalty = 0.001*np.sum(np.abs(p))

    return (
        0.2*e1 +
        0.3*e2 +
        0.5*e3 +
        penalty
    )

# ============================================================
# BOUNDS
# ============================================================

bounds = [

    (0.015,0.08),
    (0.015,0.08),

    (0.015,0.08),
    (0.015,0.08),
    (0.015,0.08),
    (0.015,0.08),
    (0.015,0.08),
    (0.015,0.08),
    (0.015,0.08),
    (0.015,0.08),

    (0.005,0.03),
    (0.005,0.03),

    (0.03,0.07),
    (0.02,0.05),
    (0.01,0.03),

    (0,np.pi/2),
    (0,np.pi/4),

    (1,4),

    (-np.pi,np.pi)
]

# ============================================================
# OPTIMIZACIÓN
# ============================================================

result = differential_evolution(

    fitness,

    bounds,

    strategy='best1bin',

    popsize=20,

    mutation=(0.5,1),

    recombination=0.7,

    maxiter=120,

    polish=True,

    disp=True
)

best = result.x

print("\n================================")
print("PARÁMETROS ÓPTIMOS")
print("================================")

for i,val in enumerate(best):

    print(f"P{i+1}: {val:.5f}")

print(f"\nFitness: {result.fun:.6f}")

# ============================================================
# RESULTADOS
# ============================================================

res = full_model(
    best,
    theta_input
)

# ============================================================
# TRAYECTORIAS
# ============================================================

plt.figure(figsize=(9,9))

plt.plot(
    mocap['proximal'][0],
    mocap['proximal'][1],
    'r--',
    linewidth=3,
    label='MOCAP IFP'
)

plt.plot(
    res['proximal'][0],
    res['proximal'][1],
    'r'
)

plt.plot(
    mocap['medial'][0],
    mocap['medial'][1],
    'g--',
    linewidth=3,
    label='MOCAP IFD'
)

plt.plot(
    res['medial'][0],
    res['medial'][1],
    'g'
)

plt.plot(
    mocap['tip'][0],
    mocap['tip'][1],
    'b--',
    linewidth=3,
    label='MOCAP TIP'
)

plt.plot(
    res['tip'][0],
    res['tip'][1],
    'b'
)

plt.grid(True)

plt.axis('equal')

plt.legend()

plt.title(
    'Comparación MOCAP vs Optimización'
)

plt.show()

# ============================================================
# ANIMACIÓN
# ============================================================

plt.figure(figsize=(8,8))

for i in range(0,N,3):

    plt.cla()

    plt.plot(
        mocap['tip'][0],
        mocap['tip'][1],
        'r--',
        alpha=0.3
    )

    plt.plot(
        res['tip'][0],
        res['tip'][1],
        'b-',
        alpha=0.3
    )

    x = [
        0,
        res['proximal'][0][i],
        res['medial'][0][i],
        res['tip'][0][i]
    ]

    y = [
        0,
        res['proximal'][1][i],
        res['medial'][1][i],
        res['tip'][1][i]
    ]

    plt.plot(
        x,
        y,
        '-o',
        linewidth=4,
        markersize=8
    )

    plt.axis('equal')

    plt.grid(True)

    plt.xlim(-0.12,0.12)
    plt.ylim(-0.02,0.12)

    plt.pause(0.03)

plt.show()

# ============================================================
# EXPORTAR
# ============================================================

df = pd.DataFrame({

    'x_ifp': res['proximal'][0],
    'y_ifp': res['proximal'][1],

    'x_ifd': res['medial'][0],
    'y_ifd': res['medial'][1],

    'x_tip': res['tip'][0],
    'y_tip': res['tip'][1]
})

df.to_csv(
    'Resultados_Optimizacion_Exoesqueleto.csv',
    index=False
)

print("\nCSV EXPORTADO")
print("FIN")