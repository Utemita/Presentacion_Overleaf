import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist

# --- Parámetros de Diseño y Antropometría ---
# Valores reales obtenidos del análisis cinemático
FP_REAL = 0.049  
FM_REAL = 0.026  
FD_REAL = 0.024  

N_PUNTOS = 120
theta_input = np.linspace(0, np.deg2rad(85), N_PUNTOS)

# --- Funciones para evaluar el error ---
def chamfer_distance(curve_target, curve_sim):
    # Evalúa qué tan parecidas son las formas de las curvas
    dists = cdist(curve_target, curve_sim)
    dist_t_to_s = np.mean(np.min(dists, axis=1))
    dist_s_to_t = np.mean(np.min(dists, axis=0))
    return dist_t_to_s + dist_s_to_t

def optimal_rigid_transform(target, sim):
    # Encuentra cómo rotar y mover el exoesqueleto para que cuadre con la mano
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

# --- Ecuaciones Cinemáticas ---
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

# --- Simulación del Exoesqueleto ---
def run_kinematics(p, th_input):
    (Bancada1, Bancada2, Link1, Link2, Link3, Link4, Link5,
     Link6, Link7, Link8, Link9, Link10, hsp, dsp,
     theta_aux_fm, theta_aux_fd, gear_ratio, theta_offset) = p

    # Descartar medidas ilógicas
    if gear_ratio <= 0 or min(p[:14]) <= 0.005: 
        return None

    PXifp, PYifp = [], []
    PXifd, PYifd = [], []
    PXtip, PYtip = [], []

    c1 = np.sqrt(hsp**2 + dsp**2)
    theta14B = np.pi/2
    theta_ps1 = np.arctan2(hsp, dsp)
    
    rs2 = np.sqrt(hsp**2 + (FP_REAL-dsp)**2)
    theta_aux_s2 = np.arctan2(hsp, FP_REAL-dsp)

    for th2 in th_input:
        th1 = (th2 / gear_ratio) + theta_offset

        # Primer lazo
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

        # Sistema secundario
        theta_roll = np.arctan2(pym4-pys1, pxm4-pxs1)
        theta1m2 = np.arctan2(pys2-pys1, pxs2-pxs1) - theta_roll
        theta2m2 = np.arctan2(pyP-pym4, pxP-pxm4) - theta_roll

        # Segundo lazo
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

        # Posiciones finales de los dedos
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

# --- Curvas Ideales (MOCAP) ---
theta_mcp = np.deg2rad(55*np.sin(theta_input))
theta_pip = np.deg2rad(75*np.sin(theta_input)**1.2)
theta_dip = 0.67*theta_pip

x_ifp = FP_REAL * np.cos(theta_mcp)
y_ifp = FP_REAL * np.sin(theta_mcp)

x_ifd = x_ifp + FM_REAL * np.cos(theta_mcp + theta_pip)
y_ifd = y_ifp + FM_REAL * np.sin(theta_mcp + theta_pip)

x_tip = x_ifd + FD_REAL * np.cos(theta_mcp + theta_pip + theta_dip)
y_tip = y_ifd + FD_REAL * np.sin(theta_mcp + theta_pip + theta_dip)

mocap_pts = {
    'ifp': np.column_stack((x_ifp, y_ifp)),
    'ifd': np.column_stack((x_ifd, y_ifd)),
    'tip': np.column_stack((x_tip, y_tip))
}

# --- Algoritmo de Optimización ---
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

    # Calculamos el error de forma
    err_ifp = chamfer_distance(mocap_pts['ifp'], sim_ifp_aligned)
    err_ifd = chamfer_distance(mocap_pts['ifd'], sim_ifd_aligned)
    err_tip = chamfer_distance(mocap_pts['tip'], sim_tip_aligned)

    # AJUSTE NUEVO: Castigar fuertemente si hay un punto muy separado en la punta
    dists_tip = cdist(mocap_pts['tip'], sim_tip_aligned)
    max_err_tip = np.max(np.min(dists_tip, axis=1))

    penalty_size = 0.005 * np.sum(p[:12]) 

    # Cambié los pesos: Ahora la punta vale el 70% del error (50% promedio + 20% máximo)
    return (0.1 * err_ifp) + (0.2 * err_ifd) + (0.5 * err_tip) + (0.2 * max_err_tip) + penalty_size

# --- Límites de las variables ---
bounds = [
    (0.015, 0.08), (0.015, 0.08),                              
    (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), 
    (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), (0.01, 0.08), 
    (0.005, 0.03), (0.005, 0.03),                              
    (0, np.pi/2), (0, np.pi/4),                                
    (1, 4),                                                    
    (-np.pi, np.pi)                                            
]

if __name__ == '__main__':
    print('Corriendo optimización, esto puede tardar un poco...')
    
    resultado = differential_evolution(
        fitness_function,
        bounds,
        strategy='best2bin',
        maxiter=300,
        popsize=25,
        mutation=(0.5, 1.2),
        recombination=0.8,
        tol=1e-4,
        polish=True,
        disp=True,
        updating='deferred',
        workers=-1 
    )

    p_opt = resultado.x
    print('\nListo, optimización terminada.')
    print(f'Mejor valor de fitness: {resultado.fun:.6f}')
    
    best_sim = run_kinematics(p_opt, theta_input)
    
    all_mocap = np.vstack((mocap_pts['ifp'], mocap_pts['ifd'], mocap_pts['tip']))
    all_sim = np.vstack((best_sim['ifp'], best_sim['ifd'], best_sim['tip']))
    
    R_opt, t_opt = optimal_rigid_transform(all_mocap, all_sim)
    angulo_montaje = np.rad2deg(np.arctan2(R_opt[1,0], R_opt[0,0]))

    print(f'\nDatos para el ensamblaje final:')
    print(f'Mover en X: {t_opt[0]*1000:.2f} mm')
    print(f'Mover en Y: {t_opt[1]*1000:.2f} mm')
    print(f'Girar mecanismo: {angulo_montaje:.2f} grados')

    sim_aligned = {
        'ifp': apply_transform(best_sim['ifp'], R_opt, t_opt),
        'ifd': apply_transform(best_sim['ifd'], R_opt, t_opt),
        'tip': apply_transform(best_sim['tip'], R_opt, t_opt)
    }

    # Gráficas
    plt.figure(figsize=(9, 7))
    
    plt.plot(mocap_pts['ifp'][:,0], mocap_pts['ifp'][:,1], 'r--', lw=2.5, label='Objetivo IFP')
    plt.plot(sim_aligned['ifp'][:,0], sim_aligned['ifp'][:,1], 'r-', lw=1.5, label='Resultado Exo IFP')

    plt.plot(mocap_pts['ifd'][:,0], mocap_pts['ifd'][:,1], 'g--', lw=2.5, label='Objetivo IFD')
    plt.plot(sim_aligned['ifd'][:,0], sim_aligned['ifd'][:,1], 'g-', lw=1.5, label='Resultado Exo IFD')

    plt.plot(mocap_pts['tip'][:,0], mocap_pts['tip'][:,1], 'b--', lw=2.5, label='Objetivo Punta')
    plt.plot(sim_aligned['tip'][:,0], sim_aligned['tip'][:,1], 'b-', lw=1.5, label='Resultado Exo Punta')

    plt.axis('equal')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.title('Cierre de trayectorias (Alineado)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()

    # Guardar resultados
    df = pd.DataFrame({
        'x_ifp_mocap': mocap_pts['ifp'][:,0], 'y_ifp_mocap': mocap_pts['ifp'][:,1],
        'x_ifp_exo': sim_aligned['ifp'][:,0], 'y_ifp_exo': sim_aligned['ifp'][:,1],
        'x_ifd_mocap': mocap_pts['ifd'][:,0], 'y_ifd_mocap': mocap_pts['ifd'][:,1],
        'x_ifd_exo': sim_aligned['ifd'][:,0], 'y_ifd_exo': sim_aligned['ifd'][:,1],
        'x_tip_mocap': mocap_pts['tip'][:,0], 'y_tip_mocap': mocap_pts['tip'][:,1],
        'x_tip_exo': sim_aligned['tip'][:,0], 'y_tip_exo': sim_aligned['tip'][:,1]
    })
    df.to_csv('Resultados_Ajustados.csv', index=False)