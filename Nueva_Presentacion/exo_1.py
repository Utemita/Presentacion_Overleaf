# ============================================================
# FRAMEWORK COMPLETO
# OPTIMIZACIÓN DE EXOESQUELETO DE MANO
# 4 BARRAS + 5 BARRAS + MOCAP
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import differential_evolution

# ============================================================
# 1. FUNCIONES GEOMÉTRICAS
# ============================================================

def center_curve(x, y):

    return (
        x - np.mean(x),
        y - np.mean(y)
    )


def rotate_curve(x, y, theta):

    xr = (
        x * np.cos(theta)
        -
        y * np.sin(theta)
    )

    yr = (
        x * np.sin(theta)
        +
        y * np.cos(theta)
    )

    return xr, yr


def optimal_rotation(
    x_ref,
    y_ref,
    x_mec,
    y_mec
):

    xr, yr = center_curve(
        x_ref,
        y_ref
    )

    xm, ym = center_curve(
        x_mec,
        y_mec
    )

    num = np.sum(
        xm * yr - ym * xr
    )

    den = np.sum(
        xm * xr + ym * yr
    )

    theta = np.arctan2(
        num,
        den
    )

    return theta


def rms_procrustes(
    x_ref,
    y_ref,
    x_mec,
    y_mec
):

    xr, yr = center_curve(
        x_ref,
        y_ref
    )

    xm, ym = center_curve(
        x_mec,
        y_mec
    )

    theta = optimal_rotation(
        x_ref,
        y_ref,
        x_mec,
        y_mec
    )

    xm_r, ym_r = rotate_curve(
        xm,
        ym,
        theta
    )

    error = np.sqrt(
        np.mean(
            (xr - xm_r)**2
            +
            (yr - ym_r)**2
        )
    )

    return error

# ============================================================
# 2. CINEMÁTICA DEL MECANISMO DE 5 BARRAS
# ============================================================

def first_five_bar(
    r1,
    r2,
    r3,
    r4,
    r5,
    theta1,
    theta2
):

    e = (
        r1 * np.sin(theta1)
        -
        r4 * np.sin(theta2)
    ) / (
        r4 * np.cos(theta2)
        -
        r1 * np.cos(theta1)
        +
        2 * r3
    )

    temp = (
        2 * (
            r1 * r3 * np.cos(theta1)
            +
            r3 * r4 * np.cos(theta2)
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

    f = temp / (
        2 * (
            r4 * np.cos(theta2)
            -
            r1 * np.cos(theta1)
            +
            2 * r3
        )
    )

    d = e**2 + 1

    g = 2 * (
        e * f
        -
        e * r1 * np.cos(theta1)
        +
        e * r3
        -
        r1 * np.sin(theta1)
    )

    h = (
        f**2
        -
        2 * f * (
            r1 * np.cos(theta1)
            -
            r3
        )
        -
        2 * r1 * r3 * np.cos(theta1)
        +
        r1**2
        +
        r3**2
        -
        r2**2
    )

    disc = g**2 - 4 * d * h

    if disc < 0:
        return None

    py = (
        -g + np.sqrt(disc)
    ) / (2 * d)

    px = e * py + f

    return px, py

# ============================================================
# 3. CINEMÁTICA DEL MECANISMO DE 4 BARRAS
# ============================================================

def first_four_bar(
    a,
    b,
    c,
    d,
    theta1,
    theta14B
):

    k1 = (
        d * np.cos(theta14B)
        +
        a * np.cos(theta1)
    )

    k2 = (
        d * np.sin(theta14B)
        +
        a * np.sin(theta1)
    )

    k3 = (
        k1**2
        +
        k2**2
        +
        c**2
        -
        b**2
    )

    A1 = -k3 - 2 * k1 * c

    B1 = 4 * k2 * c

    C1 = 2 * k1 * c - k3

    disc = B1**2 - 4 * A1 * C1

    if disc < 0:
        return None

    tan_theta4 = (
        -B1 - np.sqrt(disc)
    ) / (2 * A1)

    theta4 = 2 * np.arctan(
        tan_theta4
    )

    return theta4

# ============================================================
# 4. MODELO CINEMÁTICO COMPLETO
# ============================================================

def full_exoskeleton_kinematics(
    params,
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

        Fp,
        Fm,
        Fd,

        theta_aux_fm,
        theta_aux_fd,

        gear_ratio

    ) = params

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
    c = Link6
    d = Bancada2

    # ========================================================
    # SALIDAS
    # ========================================================

    PXf = []
    PYf = []

    PXifp = []
    PYifp = []

    PXifd = []
    PYifd = []

    THfp = []
    THfm = []
    THfd = []

    # ========================================================
    # ITERACIÓN
    # ========================================================

    for th2 in theta_input:

        th1 = th2 / gear_ratio

        # ====================================================
        # 5 BARRAS
        # ====================================================

        res5 = first_five_bar(
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
        # 4 BARRAS
        # ====================================================

        theta4 = first_four_bar(
            a,
            b,
            c,
            d,
            th1,
            np.pi/2
        )

        if theta4 is None:
            return None

        # ====================================================
        # FALANGE PROXIMAL
        # ====================================================

        theta_fp = theta4

        px_ifp = (
            Fp * np.cos(theta_fp)
            -
            d * np.cos(th1)
            -
            r3
        )

        py_ifp = (
            Fp * np.sin(theta_fp)
            -
            d * np.sin(th1)
        )

        # ====================================================
        # FALANGE MEDIAL
        # ====================================================

        theta_fm = (
            theta_fp
            +
            theta_aux_fm
        )

        px_ifd = (
            Fm * np.cos(theta_fm)
            +
            px_ifp
        )

        py_ifd = (
            Fm * np.sin(theta_fm)
            +
            py_ifp
        )

        # ====================================================
        # FALANGE DISTAL
        # ====================================================

        theta_fd = (
            theta_fm
            +
            theta_aux_fd
        )

        px_f = (
            Fd * np.cos(theta_fd)
            +
            px_ifd
        )

        py_f = (
            Fd * np.sin(theta_fd)
            +
            py_ifd
        )

        PXifp.append(px_ifp)
        PYifp.append(py_ifp)

        PXifd.append(px_ifd)
        PYifd.append(py_ifd)

        PXf.append(px_f)
        PYf.append(py_f)

        THfp.append(theta_fp)
        THfm.append(theta_fm)
        THfd.append(theta_fd)

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
            np.array(PXf),
            np.array(PYf)
        ),

        'angles': (

            np.array(THfp),
            np.array(THfm),
            np.array(THfd)
        )
    }

# ============================================================
# 5. FITNESS GLOBAL
# ============================================================

def global_fitness(
    params,
    mocap,
    theta_input
):

    result = full_exoskeleton_kinematics(
        params,
        theta_input
    )

    if result is None:
        return 1e6

    # ========================================================
    # TRAYECTORIAS EXO
    # ========================================================

    exo_prox = result['proximal']
    exo_med = result['medial']
    exo_tip = result['tip']

    # ========================================================
    # MOCAP
    # ========================================================

    mocap_prox = mocap['proximal']
    mocap_med = mocap['medial']
    mocap_tip = mocap['tip']

    # ========================================================
    # ERRORES PROCRUSTES
    # ========================================================

    error_prox = rms_procrustes(
        mocap_prox[0],
        mocap_prox[1],
        exo_prox[0],
        exo_prox[1]
    )

    error_med = rms_procrustes(
        mocap_med[0],
        mocap_med[1],
        exo_med[0],
        exo_med[1]
    )

    error_tip = rms_procrustes(
        mocap_tip[0],
        mocap_tip[1],
        exo_tip[0],
        exo_tip[1]
    )

    # ========================================================
    # PENALIZACIÓN DE COMPACTACIÓN
    # ========================================================

    compact_penalty = (
        0.002 * np.sum(params)
    )

    # ========================================================
    # PENALIZACIÓN BIOMECÁNICA
    # ========================================================

    THfp, THfm, THfd = result['angles']

    biomech_penalty = np.mean(
        np.abs(
            THfd - (2/3)*THfm
        )
    )

    # ========================================================
    # FITNESS TOTAL
    # ========================================================

    fitness = (

        0.30 * error_prox
        +
        0.30 * error_med
        +
        0.40 * error_tip
        +
        compact_penalty
        +
        0.1 * biomech_penalty
    )

    return fitness

# ============================================================
# 6. OPTIMIZACIÓN GLOBAL
# ============================================================

def optimize_exoskeleton(
    mocap,
    theta_input
):

    bounds = [

        # Bancadas
        (0.01, 0.08),
        (0.01, 0.08),

        # Links
        (0.01, 0.08),
        (0.01, 0.08),
        (0.01, 0.08),
        (0.01, 0.08),
        (0.01, 0.08),
        (0.01, 0.08),
        (0.01, 0.08),
        (0.01, 0.08),

        # Falanges
        (0.03, 0.08),
        (0.02, 0.06),
        (0.01, 0.05),

        # Ángulos auxiliares
        (0, np.pi/2),
        (0, np.pi/2),

        # Gear ratio
        (1, 5)
    ]

    result = differential_evolution(

        lambda x:
            global_fitness(
                x,
                mocap,
                theta_input
            ),

        bounds=bounds,

        strategy='best1bin',

        maxiter=250,

        popsize=25,

        mutation=(0.5, 1),

        recombination=0.7,

        polish=True,

        disp=True
    )

    return result

# ============================================================
# 7. DATOS MOCAP
# ============================================================

N = 120

theta_input = np.linspace(
    0,
    np.pi/2,
    N
)

# ============================================================
# TRAYECTORIAS OBJETIVO
# ============================================================

x1 = 0.04 * np.cos(theta_input)
y1 = 0.04 * np.sin(theta_input)

x2 = 0.06 * np.cos(theta_input)
y2 = 0.06 * np.sin(theta_input)

x3 = 0.08 * np.cos(theta_input)
y3 = 0.08 * np.sin(theta_input)

mocap = {

    'proximal': (
        x1,
        y1
    ),

    'medial': (
        x2,
        y2
    ),

    'tip': (
        x3,
        y3
    )
}

# ============================================================
# MOSTRAR MOCAP
# ============================================================

plt.figure(figsize=(7,7))

plt.plot(
    x1,
    y1,
    label='Proximal'
)

plt.plot(
    x2,
    y2,
    label='Medial'
)

plt.plot(
    x3,
    y3,
    label='Tip'
)

plt.axis('equal')

plt.grid(True)

plt.legend()

plt.title('Trayectorias MOCAP')

plt.show()

# ============================================================
# 8. OPTIMIZACIÓN
# ============================================================

result = optimize_exoskeleton(
    mocap,
    theta_input
)

best = result.x

print('\n====================================')
print('RESULTADO FINAL')
print('====================================')

for i, val in enumerate(best):

    print(
        f'Parametro {i+1}: {val:.5f}'
    )

print(
    f'\nFitness final: {result.fun:.6f}'
)

# ============================================================
# 9. RESULTADOS CINEMÁTICOS
# ============================================================

res = full_exoskeleton_kinematics(
    best,
    theta_input
)

# ============================================================
# 10. GRAFICAR RESULTADOS
# ============================================================

plt.figure(figsize=(8,8))

# MOCAP
plt.plot(
    mocap['tip'][0],
    mocap['tip'][1],
    'r--',
    linewidth=3,
    label='MOCAP'
)

# EXO
plt.plot(
    res['tip'][0],
    res['tip'][1],
    'b-',
    linewidth=3,
    label='Exoesqueleto'
)

plt.axis('equal')

plt.grid(True)

plt.legend()

plt.title(
    'Comparación Final'
)

plt.show()

# ============================================================
# 11. EXPORTAR CSV
# ============================================================

df_tip = pd.DataFrame({

    'x_tip': res['tip'][0],
    'y_tip': res['tip'][1]
})

df_tip.to_csv(
    'trayectoria_tip.csv',
    index=False
)

df_prox = pd.DataFrame({

    'x_prox': res['proximal'][0],
    'y_prox': res['proximal'][1]
})

df_prox.to_csv(
    'trayectoria_proximal.csv',
    index=False
)

df_med = pd.DataFrame({

    'x_med': res['medial'][0],
    'y_med': res['medial'][1]
})

df_med.to_csv(
    'trayectoria_medial.csv',
    index=False
)

print('\nCSV exportados correctamente')

# ============================================================
# 12. ANIMACIÓN SIMPLE
# ============================================================

plt.figure(figsize=(8,8))

for i in range(0, N, 5):

    plt.cla()

    plt.plot(
        mocap['tip'][0],
        mocap['tip'][1],
        'r--',
        alpha=0.5
    )

    plt.plot(
        res['tip'][0],
        res['tip'][1],
        'b-',
        alpha=0.5
    )

    plt.plot(
        res['tip'][0][i],
        res['tip'][1][i],
        'ko',
        markersize=10
    )

    plt.axis('equal')

    plt.grid(True)

    plt.pause(0.05)

plt.show()

# ============================================================
# FIN
# ============================================================