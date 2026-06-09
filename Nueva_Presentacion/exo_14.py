# VERSION DEFINITIVA - BUGS CINEMÁTICOS CORREGIDOS SEGÚN EL PDF ANALÍTICO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time

# ==============================================================================
# --- 1. PARÁMETROS DE DISEÑO Y ANTROPOMETRÍA ---
# ==============================================================================
FP_REAL = 0.049  # Longitud Falange Proximal (metros)
FM_REAL = 0.026  # Longitud Falange Media (metros)
FD_REAL = 0.024  # Longitud Falange Distal (metros)

# ==============================================================================
# --- 2. CARGA Y PROCESAMIENTO DE MOCAP REAL ---
# ==============================================================================
try:
    print(">> Cargando base de datos MOCAP 'mocap_indice_120pts.csv'...")
    datos_mocap = pd.read_csv("mocap_indice_120pts.csv")
    
    theta_mcp = np.deg2rad(datos_mocap['Theta_MCP'].values)
    theta_pip = np.deg2rad(datos_mocap['Theta_PIP'].values)
    theta_dip = np.deg2rad(datos_mocap['Theta_DIP'].values)
    N_PUNTOS = len(datos_mocap)
except FileNotFoundError:
    print("\n>> ERROR CRÍTICO: No se encontró el archivo 'mocap_indice_120pts.csv'.")
    exit()

# Cinemática Directa del Dedo Humano (Ruta de Referencia)
angulo_pp = theta_mcp
angulo_mp = theta_mcp + theta_pip
angulo_dp = theta_mcp + theta_pip + theta_dip

pxIFP_mocap = FP_REAL * np.cos(angulo_pp)
pyIFP_mocap = FP_REAL * np.sin(angulo_pp)
pxIFD_mocap = pxIFP_mocap + FM_REAL * np.cos(angulo_mp)
pyIFD_mocap = pyIFP_mocap + FM_REAL * np.sin(angulo_mp)
pxPF_mocap = pxIFD_mocap + FD_REAL * np.cos(angulo_dp)
pyPF_mocap = pyIFD_mocap + FD_REAL * np.sin(angulo_dp)

mocap_pts = {
    'ifp': np.column_stack((pxIFP_mocap, pyIFP_mocap)),
    'ifd': np.column_stack((pxIFD_mocap, pyIFD_mocap)),
    'tip': np.column_stack((pxPF_mocap, pyPF_mocap))
}

# ==============================================================================
# --- 3. FUNCIONES DE EVALUACIÓN DE CURVAS ---
# ==============================================================================
def chamfer_distance(curve_target, curve_sim):
    dists = cdist(curve_target, curve_sim)
    dist_t_to_s = np.mean(np.min(dists, axis=1))
    dist_s_to_t = np.mean(np.min(dists, axis=0))
    return dist_t_to_s + dist_s_to_t

def optimal_rigid_transform(target, sim):
    centroid_target = np.mean(target, axis=0)
    centroid_sim = np.mean(sim, axis=0)
    
    target_centered = target - centroid_target
    sim_centered = sim - centroid_sim
    
    H = sim_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
        
    t = centroid_target.T - R @ centroid_sim.T
    return R, t

def apply_transform(points, R, t):
    return (R @ points.T).T + t

# ==============================================================================
# --- 4. SOLUCIONES MATEMÁTICAS ---
# ==============================================================================
def sol_5_barras(r1, r2, r3, r4, r5, theta1, theta2):
    den = r4*np.cos(theta2) - r1*np.cos(theta1) + 2*r3
    if np.abs(den) < 1e-10: return None

    e = (r1*np.sin(theta1) - r4*np.sin(theta2)) / den
    temp = (2*(r1*r3*np.cos(theta1) + r3*r4*np.cos(theta2)) - r1**2 + r2**2 + r4**2 - r5**2)
    f = temp / (2*den)
    d = e**2 + 1
    g = 2*(e*f - e*r1*np.cos(theta1) + e*r3 - r1*np.sin(theta1))
    h = (f**2 - 2*f*(r1*np.cos(theta1)-r3) - 2*r1*r3*np.cos(theta1) + r1**2 + r3**2 - r2**2)
    
    disc = g**2 - 4*d*h
    if disc < 0: return None

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
    if disc < 0: return None

    tan_theta4 = (-B1 - np.sqrt(disc)) / (2*A1)
    return 2*np.arctan(tan_theta4)

# ==============================================================================
# --- 5. MODELO CINEMÁTICO CORREGIDO FÍSICAMENTE ---
# ==============================================================================
def run_kinematics(p, num_puntos):
    (Bancada1, Bancada2, Link1, Link2, Link3, Link4, Link5,
     Link6, Link7, Link8, Link9, Link10, hsp, dsp,
     theta_aux_fm, theta_aux_fd, gear_ratio, theta_offset,
     theta_ini, theta_fin) = p

    if gear_ratio <= 0 or min(p[:12]) <= 0.005: 
        return None

    th_input = np.linspace(theta_ini, theta_fin, num_puntos)
    PXifp, PYifp, PXifd, PYifd, PXtip, PYtip = [], [], [], [], [], []

    # CORRECCIÓN 1: La bancada del 1er 4-barras es fija y vertical (pi/2) según el PDF
    theta14B = np.pi/2
    c1 = np.sqrt(hsp**2 + dsp**2)
    theta_ps1 = np.arctan2(hsp, dsp)
    rs2 = np.sqrt(hsp**2 + (FP_REAL-dsp)**2)
    theta_aux_s2 = np.arctan2(hsp, FP_REAL-dsp)

    # CORRECCIÓN 2: La manivela izquierda del 2do 5-barras es estrictamente la distancia en la falange entre S1 y S2
    crank_left_2 = FP_REAL - 2*dsp
    if crank_left_2 <= 0: return None

    for th2 in th_input:
        th1 = (th2 / gear_ratio) + theta_offset

        # 1er 5-barras
        res5 = sol_5_barras(Link4, Link3, Bancada1/2, Link1, Link2, th1, th2)
        if not res5: return None
        pxP, pyP = res5

        # 1er 4-barras
        theta4 = solve_four_bar(Link4, Link5, c1, Bancada2, th1, theta14B)
        if theta4 is None: return None

        # CORRECCIÓN 3: Anclajes de falange calculados con respecto a la Bancada Fija (NO usando th1)
        theta_fp = theta4 + theta_ps1
        px_ifp = FP_REAL*np.cos(theta_fp) - Bancada1/2
        py_ifp = FP_REAL*np.sin(theta_fp) - Bancada2

        pxs1 = c1*np.cos(theta4) - Bancada1/2
        pys1 = c1*np.sin(theta4) - Bancada2

        theta_ps2 = theta_fp - theta_aux_s2
        pxs2 = rs2*np.cos(theta_ps2) - Bancada1/2
        pys2 = rs2*np.sin(theta_ps2) - Bancada2

        # CORRECCIÓN 4: pxm4 depende del lado izquierdo (th1), no de th2
        pxm4 = Link4*np.cos(th1) - Bancada1/2
        pym4 = Link4*np.sin(th1)

        theta_roll = np.arctan2(pym4-pys1, pxm4-pxs1)
        theta1m2 = np.arctan2(pys2-pys1, pxs2-pxs1) - theta_roll
        theta2m2 = np.arctan2(pyP-pym4, pxP-pxm4) - theta_roll

        # 2do 5-barras (Usando la distancia real crank_left_2 en lugar de adivinar)
        res5_2 = sol_5_barras(crank_left_2, Link7, Link5/2, Link8, Link9, theta1m2, theta2m2)
        if not res5_2: return None
        px_local, py_local = res5_2

        mag = np.sqrt(px_local**2 + py_local**2)
        theta_local = np.arctan2(py_local, px_local)
        px_aux, py_aux = (pxs1+pxm4)/2, (pys1+pym4)/2

        pxP2 = mag*np.cos(theta_local + theta_roll) + px_aux
        pyP2 = mag*np.sin(theta_local + theta_roll) + py_aux

        # 2do 4-barras
        theta1m42 = np.arctan2(py_ifp - pys2, px_ifp - pxs2)
        theta2m42 = np.arctan2(pyP2 - pys2, pxP2 - pxs2)
        
        # CORRECCIÓN 5: Link6 ahora se aprovecha correctamente como el Acoplador del 2do 4-barras
        theta4m2 = solve_four_bar(Link7, Link6, Link10, c1, theta2m42, theta1m42)
        if theta4m2 is None: return None

        theta_fm = theta4m2 + theta_aux_fm
        px_ifd = FM_REAL*np.cos(theta_fm) + px_ifp
        py_ifd = FM_REAL*np.sin(theta_fm) + py_ifp

        theta_fd = theta_fm + theta_aux_fd
        px_tip = FD_REAL*np.cos(theta_fd) + px_ifd
        py_tip = FD_REAL*np.sin(theta_fd) + py_ifd

        PXifp.append(px_ifp); PYifp.append(py_ifp)
        PXifd.append(px_ifd); PYifd.append(py_ifd)
        PXtip.append(px_tip); PYtip.append(py_tip)

        # Filtro físico contra saltos de ensamblaje matemáticos
        if len(PXtip) > 1:
            if (PXtip[-1] - PXtip[-2])**2 + (PYtip[-1] - PYtip[-2])**2 > 0.0004:
                return None

    return {
        'ifp': np.column_stack((PXifp, PYifp)),
        'ifd': np.column_stack((PXifd, PYifd)),
        'tip': np.column_stack((PXtip, PYtip))
    }

# ==============================================================================
# --- 6. FUNCIÓN OBJETIVO ---
# ==============================================================================
def fitness_function(p):
    sim_data = run_kinematics(p, N_PUNTOS)
    if sim_data is None: 
        return 1e6

    all_mocap = np.vstack((mocap_pts['ifp'], mocap_pts['ifd'], mocap_pts['tip']))
    all_sim = np.vstack((sim_data['ifp'], sim_data['ifd'], sim_data['tip']))

    R, t = optimal_rigid_transform(all_mocap, all_sim)

    sim_ifp_aligned = apply_transform(sim_data['ifp'], R, t)
    sim_ifd_aligned = apply_transform(sim_data['ifd'], R, t)
    sim_tip_aligned = apply_transform(sim_data['tip'], R, t)

    err_ifp = chamfer_distance(mocap_pts['ifp'], sim_ifp_aligned)
    err_ifd = chamfer_distance(mocap_pts['ifd'], sim_ifd_aligned)
    err_tip = chamfer_distance(mocap_pts['tip'], sim_tip_aligned)

    return (0.20 * err_ifp) + (0.30 * err_ifd) + (0.50 * err_tip)

# Límites protegidos para evitar geometrías que rompan la falange (20 Variables)
bounds = [
    (0.015, 0.10), (0.015, 0.10),                                # Bancada1, Bancada2
    (0.01, 0.10), (0.01, 0.10), (0.01, 0.10), (0.01, 0.10), (0.01, 0.10), # Links 1-5
    (0.01, 0.10), (0.01, 0.10), (0.01, 0.10), (0.01, 0.10), (0.01, 0.10), # Links 6-10 (Link6 ahora es Acoplador 2)
    (0.01, 0.05), (0.005, 0.02),                                 # hsp, dsp (Offsets base estrictamente positivos)
    (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2),                    # Theta Auxiliares
    (0.5, 5.0),                                                  # Gear Ratio
    (-np.pi, np.pi),                                             # Theta Offset
    (-0.5, 0.5),                                                 # Theta_ini Motor (rad)
    (np.deg2rad(60), np.deg2rad(140))                            # Theta_fin Motor (rad)
]

# ==============================================================================
# --- 7. META-OPTIMIZACIÓN CON OPTUNA ---
# ==============================================================================
def objective_optuna(trial):
    strategy = trial.suggest_categorical('strategy', ['best1bin', 'best2bin'])
    popsize = trial.suggest_int('popsize', 20, 35) 
    mut_min = trial.suggest_float('mut_min', 0.4, 0.7)
    mut_max = trial.suggest_float('mut_max', mut_min + 0.2, 1.3)
    recombination = trial.suggest_float('recombination', 0.7, 0.95)

    res = differential_evolution(
        fitness_function, bounds, strategy=strategy, popsize=popsize,
        mutation=(mut_min, mut_max), recombination=recombination,
        maxiter=30, tol=1e-3, polish=False, updating='immediate', workers=1 
    )
    return res.fun

# ==============================================================================
# --- 8. EJECUCIÓN PRINCIPAL ---
# ==============================================================================
if __name__ == '__main__':
    print('\n====================================================')
    print('>> ETAPA 1: Meta-Optimización Bayesiana')
    print('====================================================')
    
    study = optuna.create_study(direction='minimize')
    t0 = time.time()
    study.optimize(objective_optuna, n_trials=15) 
    
    best_hp = study.best_params
    print(f'\n>> Búsqueda Optuna finalizada en {(time.time() - t0)/60:.2f} min.')
    
    print('\n====================================================')
    print('>> ETAPA 2: Optimización Cinemática Profunda')
    print('====================================================')
    
    resultado = differential_evolution(
        fitness_function, bounds,
        strategy=best_hp['strategy'], popsize=best_hp['popsize'],
        mutation=(best_hp['mut_min'], best_hp['mut_max']),
        recombination=best_hp['recombination'],
        maxiter=600, tol=1e-5, polish=True, disp=True,
        updating='immediate', workers=1 
    )

    p_opt = resultado.x
    
    # --- Extracción y Reporte Final ---
    best_sim = run_kinematics(p_opt, N_PUNTOS)
    
    all_mocap = np.vstack((mocap_pts['ifp'], mocap_pts['ifd'], mocap_pts['tip']))
    all_sim = np.vstack((best_sim['ifp'], best_sim['ifd'], best_sim['tip']))
    
    R_opt, t_opt = optimal_rigid_transform(all_mocap, all_sim)
    angulo_montaje = np.rad2deg(np.arctan2(R_opt[1,0], R_opt[0,0]))

    sim_aligned = {
        'ifp': apply_transform(best_sim['ifp'], R_opt, t_opt),
        'ifd': apply_transform(best_sim['ifd'], R_opt, t_opt),
        'tip': apply_transform(best_sim['tip'], R_opt, t_opt)
    }

    err_ifp_mm = chamfer_distance(mocap_pts['ifp'], sim_aligned['ifp']) * 1000
    err_ifd_mm = chamfer_distance(mocap_pts['ifd'], sim_aligned['ifd']) * 1000
    err_tip_mm = chamfer_distance(mocap_pts['tip'], sim_aligned['tip']) * 1000
    error_global_mm = (err_ifp_mm + err_ifd_mm + err_tip_mm) / 3.0

    print('\n====================================================')
    print('>> REPORTE FINAL DE RESULTADOS COMPLETO')
    print('====================================================')
    print(f">> ERROR PROMEDIO GLOBAL: {error_global_mm:.4f} mm")
    print(f"   - Error IFP:   {err_ifp_mm:.4f} mm")
    print(f"   - Error IFD:   {err_ifd_mm:.4f} mm")
    print(f"   - Error Punta: {err_tip_mm:.4f} mm\n")
    
    nombres_parametros = [
        "Bancada1 (m)", "Bancada2 (m)", "Link1 (m)", "Link2 (m)", 
        "Link3 (m)", "Link4 (m)", "Link5 (m)", "Link6 (Acoplador 2) (m)", 
        "Link7 (m)", "Link8 (m)", "Link9 (m)", "Link10 (m)", 
        "Offset Alto - hsp (m)", "Offset Ancho - dsp (m)",
        "Theta Aux FM (rad)", "Theta Aux FD (rad)", 
        "Gear Ratio (Transmisión)", "Theta Offset (rad)",
        "Theta Inicial Motor (rad)", "Theta Final Motor (rad)"
    ]
    for nombre, valor in zip(nombres_parametros, p_opt):
        if "rad" in nombre:
            print(f"   {nombre:28s} : {valor:.6f} rad ({np.rad2deg(valor):.2f}°)")
        else:
            print(f"   {nombre:28s} : {valor:.6f}")
        
    print('\n>> CONFIGURACIÓN DE MONTAJE KABSCH')
    print(f'   Traslación X: {t_opt[0]*1000:.2f} mm')
    print(f'   Traslación Y: {t_opt[1]*1000:.2f} mm')
    print(f'   Rotación Base: {angulo_montaje:.2f} grados')

    # --- Gráficas ---
    plt.figure(figsize=(10, 8))
    plt.plot(mocap_pts['ifp'][:,0]*1000, mocap_pts['ifp'][:,1]*1000, 'r--', lw=2.5, alpha=0.6, label='MOCAP IFP')
    plt.plot(mocap_pts['ifd'][:,0]*1000, mocap_pts['ifd'][:,1]*1000, 'g--', lw=2.5, alpha=0.6, label='MOCAP IFD')
    plt.plot(mocap_pts['tip'][:,0]*1000, mocap_pts['tip'][:,1]*1000, 'b--', lw=2.5, alpha=0.6, label='MOCAP Punta')
    
    plt.plot(sim_aligned['ifp'][:,0]*1000, sim_aligned['ifp'][:,1]*1000, 'r-', lw=2, label='EXO IFP (Físico)')
    plt.plot(sim_aligned['ifd'][:,0]*1000, sim_aligned['ifd'][:,1]*1000, 'g-', lw=2, label='EXO IFD (Físico)')
    plt.plot(sim_aligned['tip'][:,0]*1000, sim_aligned['tip'][:,1]*1000, 'b-', lw=2, label='EXO Punta (Físico)')

    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.title(f'Síntesis de Biofidelidad Exitosa\nError Global: {error_global_mm:.3f} mm', fontweight='bold')
    plt.xlabel('Eje X (mm)'); plt.ylabel('Eje Y (mm)')
    plt.show()