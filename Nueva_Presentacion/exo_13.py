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
FP_REAL = 0.049  
FM_REAL = 0.026  
FD_REAL = 0.024  

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

theta_input = np.linspace(0, np.deg2rad(85), N_PUNTOS)

# Cinemática Directa del Dedo Humano
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
# --- 5. MODELO CINEMÁTICO ---
# ==============================================================================
def run_kinematics(p, th_input):
    (Bancada1, Bancada2, Link1, Link2, Link3, Link4, Link5,
     Link6, Link7, Link8, Link9, Link10, hsp, dsp,
     theta_aux_fm, theta_aux_fd, gear_ratio, theta_offset) = p

    if gear_ratio <= 0 or min(p[:14]) <= 0.005: 
        return None

    PXifp, PYifp, PXifd, PYifd, PXtip, PYtip = [], [], [], [], [], []

    c1 = np.sqrt(hsp**2 + dsp**2)
    theta14B = np.pi/2
    theta_ps1 = np.arctan2(hsp, dsp)
    
    rs2 = np.sqrt(hsp**2 + (FP_REAL-dsp)**2)
    theta_aux_s2 = np.arctan2(hsp, FP_REAL-dsp)

    for th2 in th_input:
        th1 = (th2 / gear_ratio) + theta_offset

        res5 = sol_5_barras(Link4, Link3, Bancada1/2, Link1, Link2, th1, th2)
        if not res5: return None
        pxP, pyP = res5

        theta4 = solve_four_bar(Link4, Link5, c1, Bancada2, th1, theta14B)
        if theta4 is None: return None

        theta_fp = theta4 + theta_ps1
        px_ifp = FP_REAL*np.cos(theta_fp) - Bancada2*np.cos(th1) - Bancada1/2
        py_ifp = FP_REAL*np.sin(theta_fp) - Bancada2*np.sin(th1)

        pxs1 = c1*np.cos(theta4) - Bancada2*np.cos(th1) - Bancada1/2
        pys1 = c1*np.sin(theta4) - Bancada2*np.sin(th1)

        theta_ps2 = theta_fp - theta_aux_s2
        pxs2 = rs2*np.cos(theta_ps2) - Bancada2*np.cos(th1) - Bancada1/2
        pys2 = rs2*np.sin(theta_ps2) - Bancada2*np.sin(th1)

        pxm4 = Link4*np.cos(th2) - Bancada1/2
        pym4 = Link4*np.sin(th2)

        theta_roll = np.arctan2(pym4-pys1, pxm4-pxs1)
        theta1m2 = np.arctan2(pys2-pys1, pxs2-pxs1) - theta_roll
        theta2m2 = np.arctan2(pyP-pym4, pxP-pxm4) - theta_roll

        res5_2 = sol_5_barras(Link6, Link7, Link5/2, Link8, Link9, theta1m2, theta2m2)
        if not res5_2: return None
        px_local, py_local = res5_2

        mag = np.sqrt(px_local**2 + py_local**2)
        theta_local = np.arctan2(py_local, px_local)
        px_aux, py_aux = (pxs1+pxm4)/2, (pys1+pym4)/2

        pxP2 = mag*np.cos(theta_local + theta_roll) + px_aux
        pyP2 = mag*np.sin(theta_local + theta_roll) + py_aux

        theta1m42 = np.arctan2(pys2-py_ifp, pxs2-px_ifp)
        theta2m42 = np.arctan2(pyP2-pys2, pxP2-pxs2)
        theta4m2 = solve_four_bar(Link7, Link8, Link10, c1, theta2m42, theta1m42)
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

    return {
        'ifp': np.column_stack((PXifp, PYifp)),
        'ifd': np.column_stack((PXifd, PYifd)),
        'tip': np.column_stack((PXtip, PYtip))
    }

# ==============================================================================
# --- 6. FUNCIÓN OBJETIVO GLOBAL ---
# ==============================================================================
def fitness_function(p):
    sim_data = run_kinematics(p, theta_input)
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

    penalty_size = 0.002 * np.sum(p[:12]) 
    return (0.33 * err_ifp) + (0.33 * err_ifd) + (0.34 * err_tip) + penalty_size

# --- Espacio de Búsqueda (MÁXIMOS de 8cm para cuidar la ergonomía) ---
bounds = [
    (0.015, 0.15), (0.015, 0.15),                                 # Bancada1, Bancada2 (Se amplió hasta 15 cm para dar más base)
    (0.01, 0.12), (0.01, 0.12), (0.01, 0.12), (0.01, 0.12), (0.01, 0.12),  # Link 1-5 (Se amplió hasta 12 cm)
    (0.01, 0.12), (0.01, 0.12), (0.01, 0.12), (0.01, 0.12), (0.01, 0.12),  # Link 6-10 (Se amplió hasta 12 cm)
    (-0.02, 0.08), (0.01, 0.08),                                  # Offsets de montaje base (hsp: permite anclaje 2cm hacia atrás del nudillo; dsp: hasta 8cm de altura)
    (0, np.pi/1.2), (0, np.pi/1.5),                               # Theta Aux (Mantenemos positivo para evitar el ensamble "hacia atrás", pero abrimos el ángulo máximo)
    (0.5, 8.0),                                                   # Gear Ratio (Permite más libertad en el acoplamiento motor-mecanismo)
    (-np.pi, np.pi)                                               # Theta Offset (Libertad total)
]

# ==============================================================================
# --- 7. OPTIMIZACIÓN BAYESIANA (OPTUNA) ---
# ==============================================================================
def objective_optuna(trial):
    strategy = trial.suggest_categorical('strategy', ['best1bin', 'best2bin', 'rand1bin'])
    popsize = trial.suggest_int('popsize', 15, 50) 
    mut_min = trial.suggest_float('mut_min', 0.2, 0.9)
    mut_max = trial.suggest_float('mut_max', mut_min + 0.1, 1.5)
    recombination = trial.suggest_float('recombination', 0.5, 0.95)

    res = differential_evolution(
        fitness_function,
        bounds,
        strategy=strategy,
        popsize=popsize,
        mutation=(mut_min, mut_max),
        recombination=recombination,
        maxiter=50,  
        tol=1e-3,
        polish=False,
        updating='deferred',
        workers=-1 
    )
    return res.fun

# ==============================================================================
# --- 8. EJECUCIÓN PRINCIPAL ---
# ==============================================================================
if __name__ == '__main__':
    print('\n====================================================')
    print('>> ETAPA 1: Búsqueda de hiperparámetros con Optuna')
    print('====================================================')
    
    study = optuna.create_study(direction='minimize')
    
    t0 = time.time()
    study.optimize(objective_optuna, n_trials=25) 
    t_optuna = time.time() - t0
    
    best_hp = study.best_params
    print(f'\n>> Búsqueda Optuna finalizada en {t_optuna/60:.2f} min.')
    print('>> Mejores hiperparámetros encontrados:')
    for k, v in best_hp.items():
        print(f'   {k}: {v}')
        
    print('\n====================================================')
    print('>> ETAPA 2: Optimización Cinemática Final (Pesada)')
    print('====================================================')
    
    resultado = differential_evolution(
        fitness_function,
        bounds,
        strategy=best_hp['strategy'],
        popsize=best_hp['popsize'],
        mutation=(best_hp['mut_min'], best_hp['mut_max']),
        recombination=best_hp['recombination'],
        maxiter=1000, 
        tol=1e-4,
        polish=True, 
        disp=True,
        updating='deferred',
        workers=-1 
    )

    p_opt = resultado.x
    print('\n>> Optimización Finalizada')
    print(f'Fitness alcanzado: {resultado.fun:.6f}')
    
    # -------------------------------------------------------------------------
    # --- NUEVO BLOQUE: IMPRESIÓN ORDENADA DE LOS 18 PARÁMETROS FINALES ---
    # -------------------------------------------------------------------------
    nombres_parametros = [
        "Bancada1 (m)", "Bancada2 (m)", "Link1 (m)", "Link2 (m)", 
        "Link3 (m)", "Link4 (m)", "Link5 (m)", "Link6 (m)", 
        "Link7 (m)", "Link8 (m)", "Link9 (m)", "Link10 (m)", 
        "Offset X MOCAP - hsp (m)", "Offset Y MOCAP - dsp (m)",
        "Theta Aux FM (rad)", "Theta Aux FD (rad)", 
        "Gear Ratio (Transmisión)", "Theta Offset (rad)"
    ]
    
    print('\n====================================================')
    print('>> PARÁMETROS FINALES DEL MECANISMO (18 VARIABLES)')
    print('====================================================')
    for nombre, valor in zip(nombres_parametros, p_opt):
        # Imprime con un formato alineado para que parezca una tabla limpia
        print(f"   {nombre:28s} : {valor:.6f}")
    print('====================================================\n')
    
    # --- Evaluación del mejor modelo ---
    best_sim = run_kinematics(p_opt, theta_input)
    
    all_mocap = np.vstack((mocap_pts['ifp'], mocap_pts['ifd'], mocap_pts['tip']))
    all_sim = np.vstack((best_sim['ifp'], best_sim['ifd'], best_sim['tip']))
    
    R_opt, t_opt = optimal_rigid_transform(all_mocap, all_sim)
    angulo_montaje = np.rad2deg(np.arctan2(R_opt[1,0], R_opt[0,0]))

    print(f'--- Parámetros de Calibración / Montaje ---')
    print(f'Traslación X: {t_opt[0]*1000:.2f} mm')
    print(f'Traslación Y: {t_opt[1]*1000:.2f} mm')
    print(f'Rotación de la base: {angulo_montaje:.2f} grados')

    sim_aligned = {
        'ifp': apply_transform(best_sim['ifp'], R_opt, t_opt),
        'ifd': apply_transform(best_sim['ifd'], R_opt, t_opt),
        'tip': apply_transform(best_sim['tip'], R_opt, t_opt)
    }

    plt.figure(figsize=(9, 7))
    plt.plot(mocap_pts['ifp'][:,0], mocap_pts['ifp'][:,1], 'r--', lw=2.5, label='MOCAP IFP (Real)')
    plt.plot(sim_aligned['ifp'][:,0], sim_aligned['ifp'][:,1], 'r-', lw=1.5, label='EXO IFP (Simulado)')

    plt.plot(mocap_pts['ifd'][:,0], mocap_pts['ifd'][:,1], 'g--', lw=2.5, label='MOCAP IFD (Real)')
    plt.plot(sim_aligned['ifd'][:,0], sim_aligned['ifd'][:,1], 'g-', lw=1.5, label='EXO IFD (Simulado)')

    plt.plot(mocap_pts['tip'][:,0], mocap_pts['tip'][:,1], 'b--', lw=2.5, label='MOCAP Punta (Real)')
    plt.plot(sim_aligned['tip'][:,0], sim_aligned['tip'][:,1], 'b-', lw=1.5, label='EXO Punta (Simulado)')

    plt.axis('equal')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.title('Comparación de Trayectorias: EXO vs MOCAP Real')
    plt.xlabel('Eje X (m)')
    plt.ylabel('Eje Y (m)')
    plt.show()

    np.savetxt("Parametros_Optimizados_Mecanismo.txt", p_opt, header="Eslabones y offsets resultantes")

    df = pd.DataFrame({
        'x_ifp_mocap': mocap_pts['ifp'][:,0], 'y_ifp_mocap': mocap_pts['ifp'][:,1],
        'x_ifp_exo': sim_aligned['ifp'][:,0], 'y_ifp_exo': sim_aligned['ifp'][:,1],
        'x_ifd_mocap': mocap_pts['ifd'][:,0], 'y_ifd_mocap': mocap_pts['ifd'][:,1],
        'x_ifd_exo': sim_aligned['ifd'][:,0], 'y_ifd_exo': sim_aligned['ifd'][:,1],
        'x_tip_mocap': mocap_pts['tip'][:,0], 'y_tip_mocap': mocap_pts['tip'][:,1],
        'x_tip_exo': sim_aligned['tip'][:,0], 'y_tip_exo': sim_aligned['tip'][:,1]
    })
    df.to_csv('Resultados_Alineados.csv', index=False)
    print(">> Datos guardados en Resultados_Alineados.csv y Parametros_Optimizados_Mecanismo.txt")

