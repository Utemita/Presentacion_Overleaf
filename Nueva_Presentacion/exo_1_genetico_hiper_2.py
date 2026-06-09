# ============================================================
# OPTIMIZACIÓN INICIAL DEL EXOESQUELETO
# MEDIANTE ALGORITMOS GENÉTICOS Y HIPERHEURÍSTICA
# ============================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from typing import Tuple, Optional, List

# ============================================================
# PARÁMETROS GENERALES Y OBJETIVO
# ============================================================

POB_SIZE= 60
GENERACIONES = 120
TASA_MUTACION = 0.25
TASA_CRUCE = 0.8
N_P = 120
FP_REAL = 0.050  # Longitud falange proximal (m)

# Trayectoria objetivo
theta_input = np.linspace(0, np.deg2rad(85), N_P)
theta_mcp = np.deg2rad(55 * np.sin(theta_input))

x_obj = FP_REAL * np.cos(theta_mcp)
y_obj = FP_REAL * np.sin(theta_mcp)

# Límites de búsqueda (LIMITES)
LIMITES = [
    (0.015, 0.08),  # 0: Bancada1
    (0.015, 0.08),  # 1: Bancada2
    (0.01, 0.08),   # 2: Link1
    (0.01, 0.08),   # 3: Link2
    (0.01, 0.08),   # 4: Link3
    (0.01, 0.08),   # 5: Link4
    (0.01, 0.08),   # 6: Link5
    (0.005, 0.03),  # 7: hsp
    (0.005, 0.03),  # 8: dsp
    (0.04, 0.07),   # 9: Fp (Variable o Fija en optimización)
    (1.0, 4.0),     # 10: gear_ratio
    (-np.pi, np.pi) # 11: theta_offset
]

# ============================================================
# FUNCIONES MATEMÁTICAS Y CINEMÁTICAS
# ============================================================

def rms_error(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> float:
    return np.sqrt(np.mean((x1 - x2)**2 + (y1 - y2)**2))

def sol_5_barras(r1, r2, r3, r4, r5, theta1, theta2) -> Optional[Tuple[float, float]]:
    den = r4 * np.cos(theta2) - r1 * np.cos(theta1) + 2 * r3
    if abs(den) < 1e-8:
        return None

    e = (r1 * np.sin(theta1) - r4 * np.sin(theta2)) / den
    temp = (2 * (r1 * r3 * np.cos(theta1) + r3 * r4 * np.cos(theta2)) 
            - r1**2 + r2**2 + r4**2 - r5**2)
    f = temp / (2 * den)
    
    d = e**2 + 1
    g = 2 * (e * f - e * r1 * np.cos(theta1) + e * r3 - r1 * np.sin(theta1))
    h = (f**2 - 2 * f * (r1 * np.cos(theta1) - r3) 
         - 2 * r1 * r3 * np.cos(theta1) + r1**2 + r3**2 - r2**2)
    
    disc = g**2 - 4 * d * h
    if disc < 0:
        return None

    py = (-g + np.sqrt(disc)) / (2 * d)
    px = e * py + f
    return px, py

def sol_4_barras(a, b, c, d, theta2, theta1) -> Optional[float]:
    k1 = a * np.cos(theta2) + d * np.cos(theta1)
    k2 = a * np.sin(theta2) + d * np.sin(theta1)
    k3 = k1**2 + k2**2 + c**2 - b**2
    
    A1 = -2 * k1 * c - k3
    B1 = 4 * k2 * c
    C1 = 2 * k1 * c - k3
    
    disc = B1**2 - 4 * A1 * C1
    if disc < 0:
        return None

    tan_theta4 = (-B1 - np.sqrt(disc)) / (2 * A1)
    return 2 * np.arctan(tan_theta4)

def modelo_cinematico(p: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if np.any(p <= 0) and p[11] >= 0: # Ajuste: permite offset negativo, pero rechaza distancias <= 0
        pass # La validación precisa requiere revisar índices específicos

    B1, B2, L1, L2, L3, L4, L5, hsp, dsp, Fp, gr, th_off = p

    # Penalización básica para longitudes (índices 0 a 9)
    if np.any(p[:10] <= 0):
        return None

    PX, PY = [], []
    c1 = np.sqrt(hsp**2 + dsp**2)
    theta14B = np.pi / 2
    theta_ps1 = np.arctan2(hsp, dsp)

    for th2 in theta_input:
        th1 = (th2 / gr) + th_off
        
        # Lazo 5 barras
        res5 = sol_5_barras(L4, L3, B1/2, L1, L2, th1, th2)
        if res5 is None: return None
        
        # Lazo 4 barras
        theta4 = sol_4_barras(L4, L5, c1, B2, th1, theta14B)
        if theta4 is None: return None
        
        # Posición IFP
        theta_fp = theta4 + theta_ps1
        px_ifp = Fp * np.cos(theta_fp) - B2 * np.cos(th1) - B1/2
        py_ifp = Fp * np.sin(theta_fp) - B2 * np.sin(th1)
        
        if np.isnan(px_ifp): return None
        PX.append(px_ifp)
        PY.append(py_ifp)

    return np.array(PX), np.array(PY)

def fitness(individual: np.ndarray) -> float:
    result = modelo_cinematico(individual)
    if result is None:
        return 1e6  # Penalización alta por geometría imposible
    
    x, y = result
    error = rms_error(x_obj, y_obj, x, y)
    compact_penalty = 0.001 * np.sum(np.abs(individual)) # Fomenta exoesqueletos compactos
    
    return error + compact_penalty

# ============================================================
# OPERADORES GENÉTICOS Y HIPERHEURÍSTICA
# ============================================================

def crear_individuo() -> np.ndarray:
    return np.array([random.uniform(b[0], b[1]) for b in LIMITES])

# Mutaciones separadas
def mutar_suave(ind: np.ndarray) -> np.ndarray:
    hijo= ind.copy()
    idx = random.randint(0, len(ind) - 1)
    hijo[idx] += random.uniform(-0.005, 0.005)
    return hijo

def mutar_agresivo(ind: np.ndarray) -> np.ndarray:
    hijo= ind.copy()
    for i in range(len(ind)):
        if random.random() < 0.3:
            hijo[i] += random.uniform(-0.02, 0.02)
    return hijo

# Cruces separados
def cruce_agresivo(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    alpha = random.random()
    return alpha * p1 + (1 - alpha) * p2

def cruce_uniforme(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2)

# El "Agente" Hiperheurístico
def hyper_heuristic_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Selecciona dinámicamente la estrategia de cruce."""
    method = random.choice(['average', 'uniform'])
    if method == 'average':
        return cruce_agresivo(p1, p2)
    return cruce_uniforme(p1, p2)

def hyper_heuristic_mutation(ind: np.ndarray) -> np.ndarray:
    """Selecciona dinámicamente la estrategia de mutación."""
    method = random.choice(['soft', 'aggressive'])
    if method == 'soft':
        return mutar_suave(ind)
    return mutar_agresivo(ind)

def torneo_seleccion(pob: List[np.ndarray], fitnesses: np.ndarray, k: int = 2) -> np.ndarray:
    """Selecciona el mejor de 'k' individuos elegidos al azar."""
    indices = random.sample(range(len(pob)), k)
    best_idx = min(indices, key=lambda idx: fitnesses[idx])
    return pob[best_idx]

# ============================================================
# BUCLE PRINCIPAL (MAIN)
# ============================================================

def main():
    print('INICIANDO OPTIMIZACIÓN...')
    pob = [crear_individuo() for _ in range(POB_SIZE)]
    best_history = []
    global_best_ind = None
    global_best_fit = float('inf')

    for gen in range(GENERACIONES):
        # Evaluar población
        fitness_values = np.array([fitness(ind) for ind in pob])
        
        # Encontrar el mejor de la generación
        current_best_idx = np.argmin(fitness_values)
        current_best_fit = fitness_values[current_best_idx]
        
        # Actualizar mejor global (Elitismo puro)
        if current_best_fit < global_best_fit:
            global_best_fit = current_best_fit
            global_best_ind = pob[current_best_idx].copy()

        best_history.append(global_best_fit)
        
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f'Generación {gen+1:03d} | Mejor Fitness = {global_best_fit:.6f}')

        # ========================================================
        # CREAR NUEVA POBLACIÓN
        # ========================================================
        nueva_pob = [global_best_ind.copy()] # Elitismo: conservar el mejor

        while len(nueva_pob) < POB_SIZE:
            padre_1 = torneo_seleccion(pob, fitness_values)
            padre_2 = torneo_seleccion(pob, fitness_values)

            # Capa 1: Hiperheurística de Cruce
            if random.random() < TASA_CRUCE:
                hijo= hyper_heuristic_crossover(padre_1, padre_2)
            else:
                hijo= padre_1.copy()

            # Capa 2: Hiperheurística de Mutación
            if random.random() < TASA_MUTACION:
                hijo= hyper_heuristic_mutation(hijo)

            nueva_pob.append(hijo)

        pob = nueva_pob

    print('\n===================================')
    print('OPTIMIZACIÓN FINALIZADA')
    print('===================================')
    print(f'Fitness final: {global_best_fit:.6f}')

    # ============================================================
    # GRÁFICAS DE RESULTADOS
    # ============================================================
    final_result = modelo_cinematico(global_best_ind)
    
    if final_result is not None:
        x_best, y_best = final_result

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfica 1: Trayectorias
        ax1.plot(x_obj, y_obj, 'r--', linewidth=3, label='Trayectoria Objetivo (Base de datos)')
        ax1.plot(x_best, y_best, 'b', linewidth=3, label='Exoesqueleto Optimizado')
        ax1.set_title('Optimización Inicial de la Articulación IFP')
        ax1.axis('equal')
        ax1.grid(True)
        ax1.legend()

        # Gráfica 2: Convergencia
        ax2.plot(best_history, 'k-', linewidth=2)
        ax2.set_title('Convergencia de la Hiperheurística y AG')
        ax2.set_xlabel('Generación')
        ax2.set_ylabel('Error RMS + Penalización (Fitness)')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('imagens/resultado_hiper_AG.png', dpi=150, bbox_inches='tight')
    else:
        print("El algoritmo no logró converger a una geometría válida.")

if __name__ == '__main__':
    main()