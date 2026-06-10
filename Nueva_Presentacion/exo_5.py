
# ============================================================
# FRAMEWORK DEFINITIVO: EVOLUCIÓN DIFERENCIAL
# OPTIMIZACIÓN COMPLETA DE EXOESQUELETO DE MANO CON ALINEACIÓN
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution

# ============================================================
# PARÁMETROS GLOBALES Y ANTROPOMÉTRICOS (ADULTO PROMEDIO)
# ============================================================
FP_REAL = 0.045  # Falange proximal (45 mm)
FM_REAL = 0.028  # Falange medial (28 mm)
FD_REAL = 0.018  # Falange distal (18 mm)

N_PUNTOS = 120
theta_input = np.linspace(0, np.deg2rad(85), N_PUNTOS)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def wrap_to_pi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def rms_error(x1, y1, x2, y2):
    return np.sqrt(np.mean((x1 - x2)**2 + (y1 - y2)**2))

# ============================================================
# SOLUCIONES CINEMÁTICAS (LAZOS CERRADOS)
# ============================================================
def sol_5_barras(r1, r2, r3, r4, r5, theta1, theta2):
    den = r4*np.cos(theta2) - r1*np.cos(theta1) + 2*r3
    if np.abs(den) < 1e-10:
        return None

    e = (r1*np.sin(theta1) - r4*np.sin(theta2)) / den
    temp = (2*(r1*r3*np.cos(theta1) + r3*r4*np.cos(theta2)) - r1**2 + r2**2 + r4**2 - r5**2)
    f = temp / (2*den)
    d = e**2 + 1
    g = 2*(e*f - e*r1*np.cos(theta1) + e*r3 - r1*np.sin(theta1))
    h = (f**2 - 2*f*(r1*np.cos(theta1)-r3) - 2*r1*r3*np.cos(theta1) + r1**2 + r3**2 - r2**2)
    
    disc = g**2 - 4*d*h
    if disc < 0:
        return None

    py = (-g + np.sqrt(disc)) / (2*d)
    px = e*py + f
    return px, py

def solve_four_bar(a, b, c, d, theta2, theta1):
    k1 = a*np.cos(theta2) + d*np.cos(theta1)
    k2 = a*np.sin(theta2) + d*np.sin(theta1)
    k3 = k1**2 + k2**2 + c**2 - b**2
    
    A1 = -2*k1*c - k3
    B1 = 4*k2*c
    C1 = 2*k1*c - k3
    disc = B1**2 - 4*A1*C1
    
    if disc < 0:
        return None

    tan_theta4 = (-B1 - np.sqrt(disc)) / (2*A1)
    return 2*np.arctan(tan_theta4)

# ============================================================
# MODELO CINEMÁTICO COMPLETO (18 VARIABLES DE DISEÑO)
# ============================================================
def full_model(p, th_input):
    (
        Bancada1, Bancada2,
        Link1, Link2, Link3, Link4, Link5,
        Link6, Link7, Link8, Link9, Link10,
        hsp, dsp,
        theta_aux_fm, theta_aux_fd,
        gear_ratio, theta_offset
    ) = p

    Fp = FP_REAL
    Fm = FM_REAL
    Fd = FD_REAL

    if gear_ratio <= 0 or min(p[:14]) <= 0: # Penalizar longitudes <= 0
        return None

    PXifp, PYifp = [], []
    PXifd, PYifd = [], []
    PXtip, PYtip = [], []

    c1 = np.sqrt(hsp**2 + dsp**2)
    theta14B = np.pi/2
    theta_ps1 = np.arctan2(hsp, dsp)
    
    rs2 = np.sqrt(hsp**2 + (Fp-dsp)**2)
    theta_aux_s2 = np.arctan2(hsp, Fp-dsp)

    for th2 in th_input:
        th1 = (th2 / gear_ratio) + theta_offset

        # 1er 5 barras
        res5 = sol_5_barras(Link4, Link3, Bancada1/2, Link1, Link2, th1, th2)
        if res5 is None: return None
        pxP, pyP = res5

        # 1er 4 barras
        theta4 = solve_four_bar(Link4, Link5, c1, Bancada2, th1, theta14B)
        if theta4 is None: return None

        # Posiciones IFP y Soportes
        theta_fp = theta4 + theta_ps1
        px_ifp = Fp*np.cos(theta_fp) - Bancada2*np.cos(th1) - Bancada1/2
        py_ifp = Fp*np.sin(theta_fp) - Bancada2*np.sin(th1)

        pxs1 = c1*np.cos(theta4) - Bancada2*np.cos(th1) - Bancada1/2
        pys1 = c1*np.sin(theta4) - Bancada2*np.sin(th1)

        theta_ps2 = theta_fp - theta_aux_s2
        pxs2 = rs2*np.cos(theta_ps2) - Bancada2*np.cos(th1) - Bancada1/2
        pys2 = rs2*np.sin(theta_ps2) - Bancada2*np.sin(th1)

        pxm4 = Link4*np.cos(th2) - Bancada1/2
        pym4 = Link4*np.sin(th2)

        # Sistema Secundario
        theta_roll = np.arctan2(pym4-pys1, pxm4-pxs1)
        theta1m2 = np.arctan2(pys2-pys1, pxs2-pxs1) - theta_roll
        theta2m2 = np.arctan2(pyP-pym4, pxP-pxm4) - theta_roll

        # 2do 5 barras
        res5_2 = sol_5_barras(Link6, Link7, Link5/2, Link8, Link9, theta1m2, theta2m2)
        if res5_2 is None: return None
        px_local, py_local = res5_2

        # Transformación Global
        mag = np.sqrt(px_local**2 + py_local**2)
        theta_local = np.arctan2(py_local, px_local)
        px_aux, py_aux = (pxs1+pxm4)/2, (pys1+pym4)/2

        pxP2 = mag*np.cos(theta_local + theta_roll) + px_aux
        pyP2 = mag*np.sin(theta_local + theta_roll) + py_aux

        # 2do 4 barras
        theta1m42 = np.arctan2(pys2-py_ifp, pxs2-px_ifp)
        theta2m42 = np.arctan2(pyP2-pys2, pxP2-pxs2)
        theta4m2 = solve_four_bar(Link7, Link8, Link10, c1, theta2m42, theta1m42)
        if theta4m2 is None: return None

        # Falanges Medial y Distal
        theta_fm = theta4m2 + theta_aux_fm
        px_ifd = Fm*np.cos(theta_fm) + px_ifp
        py_ifd = Fm*np.sin(theta_fm) + py_ifp

        theta_fd = theta_fm + theta_aux_fd
        px_tip = Fd*np.cos(theta_fd) + px_ifd
        py_tip = Fd*np.sin(theta_fd) + py_ifd

        if np.isnan(px_tip) or abs(px_tip) > 1: return None

        PXifp.append(px_ifp); PYifp.append(py_ifp)
        PXifd.append(px_ifd); PYifd.append(py_ifd)
        PXtip.append(px_tip); PYtip.append(py_tip)

    return {
        'proximal': (np.array(PXifp), np.array(PYifp)),
        'medial': (np.array(PXifd), np.array(PYifd)),
        'tip': (np.array(PXtip), np.array(PYtip))
    }

# ============================================================
# MOCAP BIOMECÁNICO (TRAYECTORIAS OBJETIVO)
# ============================================================
theta_mcp = np.deg2rad(55*np.sin(theta_input))
theta_pip = np.deg2rad(75*np.sin(theta_input)**1.2)
theta_dip = 0.67*theta_pip

x_ifp = FP_REAL * np.cos(theta_mcp)
y_ifp = FP_REAL * np.sin(theta_mcp)

x_ifd = x_ifp + FM_REAL * np.cos(theta_mcp + theta_pip)
y_ifd = y_ifp + FM_REAL * np.sin(theta_mcp + theta_pip)

x_tip = x_ifd + FD_REAL * np.cos(theta_mcp + theta_pip + theta_dip)
y_tip = y_ifd + FD_REAL * np.sin(theta_mcp + theta_pip + theta_dip)

mocap = {
    'proximal': (x_ifp, y_ifp),
    'medial': (x_ifd, y_ifd),
    'tip': (x_tip, y_tip)
}

# ============================================================
# FUNCIÓN DE FITNESS (CON ALINEACIÓN DINÁMICA DE OFFSET)
# ============================================================
def fitness(p):
    result = full_model(p, theta_input)
    if result is None: return 1e6

    # 1. Calcular Offset (Diferencia entre orígenes)
    ox = mocap['proximal'][0][0] - result['proximal'][0][0]
    oy = mocap['proximal'][1][0] - result['proximal'][1][0]

    # 2. Trasladar trayectoria del Exoesqueleto
    exo_ifp_x = result['proximal'][0] + ox
    exo_ifp_y = result['proximal'][1] + oy
    exo_ifd_x = result['medial'][0] + ox
    exo_ifd_y = result['medial'][1] + oy
    exo_tip_x = result['tip'][0] + ox
    exo_tip_y = result['tip'][1] + oy

    # 3. Evaluar RMS Error de la FORMA de la trayectoria
    e1 = rms_error(mocap['proximal'][0], mocap['proximal'][1], exo_ifp_x, exo_ifp_y)
    e2 = rms_error(mocap['medial'][0], mocap['medial'][1], exo_ifd_x, exo_ifd_y)
    e3 = rms_error(mocap['tip'][0], mocap['tip'][1], exo_tip_x, exo_tip_y)

    compact_penalty = 0.001 * np.sum(np.abs(p))
    return 0.2*e1 + 0.3*e2 + 0.5*e3 + compact_penalty

# ============================================================
# LÍMITES DE BÚSQUEDA (18 VARIABLES)
# ============================================================
bounds = [
    (0.015, 0.08), (0.015, 0.08),                              # 0-1: Bancada1, Bancada2
    (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), # 2-6: Link1 a Link5
    (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), # 7-11: Link6 a Link10
    (0.005, 0.03), (0.005, 0.03),                              # 12-13: hsp, dsp
    (0, np.pi/2), (0, np.pi/4),                                # 14-15: theta_aux_fm, theta_aux_fd
    (1, 4),                                                    # 16: gear_ratio
    (-np.pi, np.pi)                                            # 17: theta_offset
]

# ============================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ============================================================
if __name__ == '__main__':
    print('INICIANDO OPTIMIZACIÓN GLOBAL (EVOLUCIÓN DIFERENCIAL)...')
    
    result = differential_evolution(
        fitness,
        bounds,
        strategy='best2bin',
        maxiter=300,
        popsize=20,
        mutation=(0.5, 1.2),
        recombination=0.8,
        tol=0.0001,
        polish=True,
        disp=True,
        updating='deferred',
        workers=-1 # Cambiar a 1 si tienes problemas de ejecución en Jupyter/Spyder
    )

    best = result.x
    print('\nOPTIMIZACIÓN COMPLETADA')
    print(f'Error RMS Combinado (Fitness): {result.fun:.6f}')
    
    # Simular con los mejores parámetros
    res = full_model(best, theta_input)

    # ========================================================
    # CÁLCULO DEL OFFSET PARA VISUALIZACIÓN Y EXPORTACIÓN
    # ========================================================
    ox = mocap['proximal'][0][0] - res['proximal'][0][0]
    oy = mocap['proximal'][1][0] - res['proximal'][1][0]
    
    print(f'\n-> PARÁMETRO DE MONTAJE (Offset respecto a MCF humano):')
    print(f'   X: {ox*1000:.2f} mm  |  Y: {oy*1000:.2f} mm')

    # Aplicar offset a los datos generados para sobreponerlos perfectamente
    res_aligned = {
        'proximal': (res['proximal'][0] + ox, res['proximal'][1] + oy),
        'medial': (res['medial'][0] + ox, res['medial'][1] + oy),
        'tip': (res['tip'][0] + ox, res['tip'][1] + oy)
    }

    # ========================================================
    # GRÁFICA COMPARATIVA ALINEADA
    # ========================================================
    plt.figure(figsize=(10, 8))
    
    plt.plot(mocap['proximal'][0], mocap['proximal'][1], 'r--', linewidth=3, label='MOCAP IFP')
    plt.plot(res_aligned['proximal'][0], res_aligned['proximal'][1], 'r', linewidth=1.5, label='EXO IFP')

    plt.plot(mocap['medial'][0], mocap['medial'][1], 'g--', linewidth=3, label='MOCAP IFD')
    plt.plot(res_aligned['medial'][0], res_aligned['medial'][1], 'g', linewidth=1.5, label='EXO IFD')

    plt.plot(mocap['tip'][0], mocap['tip'][1], 'b--', linewidth=3, label='MOCAP TIP')
    plt.plot(res_aligned['tip'][0], res_aligned['tip'][1], 'b', linewidth=1.5, label='EXO TIP')

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Comparación de Trayectorias (Alineadas)')
    plt.xlabel('Eje X (m)')
    plt.ylabel('Eje Y (m)')
    plt.show()

    # ========================================================
    # ANIMACIÓN ALINEADA
    # ========================================================
    plt.figure(figsize=(8, 8))
    for i in range(0, N_PUNTOS, 3):
        plt.cla()
        # Graficar trayectoria MOCAP (Fondo)
        plt.plot(mocap['tip'][0], mocap['tip'][1], 'b--', alpha=0.3, label='Ruta Ideal Punta')
        plt.plot(res_aligned['tip'][0], res_aligned['tip'][1], 'r-', alpha=0.3, label='Ruta Exo Punta')

        # Graficar Eslabones del exoesqueleto (Alineados)
        x_links = [0, res_aligned['proximal'][0][i], res_aligned['medial'][0][i], res_aligned['tip'][0][i]]
        y_links = [0, res_aligned['proximal'][1][i], res_aligned['medial'][1][i], res_aligned['tip'][1][i]]
        
        plt.plot(x_links, y_links, '-ko', linewidth=4, markersize=8)

        plt.axis('equal')
        plt.grid(True)
        plt.xlim(-0.02, 0.12)
        plt.ylim(-0.02, 0.12)
        plt.title('Animación del Exoesqueleto Optimizado')
        plt.legend()
        plt.pause(0.01)
    
    plt.show()

    # ========================================================
    # EXPORTACIÓN CSV
    # ========================================================
    df = pd.DataFrame({
        'x_ifp_mocap': mocap['proximal'][0], 'y_ifp_mocap': mocap['proximal'][1],
        'x_ifp_exo': res_aligned['proximal'][0], 'y_ifp_exo': res_aligned['proximal'][1],
        
        'x_ifd_mocap': mocap['medial'][0], 'y_ifd_mocap': mocap['medial'][1],
        'x_ifd_exo': res_aligned['medial'][0], 'y_ifd_exo': res_aligned['medial'][1],
        
        'x_tip_mocap': mocap['tip'][0], 'y_tip_mocap': mocap['tip'][1],
        'x_tip_exo': res_aligned['tip'][0], 'y_tip_exo': res_aligned['tip'][1]
    })

    df.to_csv('Resultados_Exoesqueleto_Alineados.csv', index=False)
    print('\n¡Datos exportados con éxito a "Resultados_Exoesqueleto_Alineados.csv"!')
    
    