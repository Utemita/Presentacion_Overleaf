# ============================================================
# OPTIMIZACIÓN INICIAL DEL EXOESQUELETO
# MEDIANTE ALGORITMOS GENÉTICOS
#
# ETAPA PRELIMINAR DEL PROYECTO
#
# SOLO:
# - PRIMER 5 BARRAS
# - PRIMER 4 BARRAS
# - ARTICULACIÓN IFP
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import random

# ============================================================
# PARÁMETROS GENERALES
# ============================================================

POP_SIZE = 60
GENERATIONS = 120

MUTATION_RATE = 0.25
CROSSOVER_RATE = 0.8

# ============================================================
# TRAYECTORIA OBJETIVO (BIOMECÁNICA)
# ============================================================

N = 120

theta_input = np.linspace(
    0,
    np.deg2rad(85),
    N
)

# Flexión MCP aproximada

theta_mcp = np.deg2rad(
    55*np.sin(theta_input)
)

# Longitud falange proximal

Fp_real = 0.050

# Trayectoria objetivo IFP

x_target = (
    Fp_real*np.cos(theta_mcp)
)

y_target = (
    Fp_real*np.sin(theta_mcp)
)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def rms_error(x1,y1,x2,y2):

    return np.sqrt(
        np.mean(
            (x1-x2)**2 +
            (y1-y2)**2
        )
    )

# ============================================================
# SOLUCIÓN 5 BARRAS
# ============================================================

def solve_five_bar(
    r1,r2,r3,r4,r5,
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

    if abs(den) < 1e-8:
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

    return px,py

# ============================================================
# SOLUCIÓN 4 BARRAS
# ============================================================

def solve_four_bar(
    a,b,c,d,
    theta2,
    theta1
):

    k1 = (
        a*np.cos(theta2)
        +
        d*np.cos(theta1)
    )

    k2 = (
        a*np.sin(theta2)
        +
        d*np.sin(theta1)
    )

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
# MODELO CINEMÁTICO SIMPLIFICADO
# ============================================================

def kinematic_model(p):

    (
        Bancada1,
        Bancada2,

        Link1,
        Link2,
        Link3,
        Link4,
        Link5,

        hsp,
        dsp,

        Fp,

        gear_ratio,
        theta_offset
    ) = p

    # Penalización geometría imposible

    if min(p) <= 0:
        return None

    PX = []
    PY = []

    for th2 in theta_input:

        th1 = (
            th2/gear_ratio
            +
            theta_offset
        )

        # ====================================================
        # 5 BARRAS
        # ====================================================

        res5 = solve_five_bar(

            Link4,
            Link3,
            Bancada1/2,
            Link1,
            Link2,

            th1,
            th2
        )

        if res5 is None:
            return None

        pxP,pyP = res5

        # ====================================================
        # 4 BARRAS
        # ====================================================

        c1 = np.sqrt(
            hsp**2 +
            dsp**2
        )

        theta14B = np.pi/2

        theta4 = solve_four_bar(

            Link4,
            Link5,
            c1,
            Bancada2,

            th1,
            theta14B
        )

        if theta4 is None:
            return None

        # ====================================================
        # POSICIÓN IFP
        # ====================================================

        theta_ps1 = np.arctan2(
            hsp,
            dsp
        )

        theta_fp = (
            theta4 +
            theta_ps1
        )

        px_ifp = (
            Fp*np.cos(theta_fp)
            -
            Bancada2*np.cos(th1)
            -
            Bancada1/2
        )

        py_ifp = (
            Fp*np.sin(theta_fp)
            -
            Bancada2*np.sin(th1)
        )

        if np.isnan(px_ifp):
            return None

        PX.append(px_ifp)
        PY.append(py_ifp)

    return np.array(PX),np.array(PY)

# ============================================================
# FITNESS
# ============================================================

def fitness(individual):

    result = kinematic_model(individual)

    if result is None:
        return 1e6

    x,y = result

    error = rms_error(
        x_target,
        y_target,
        x,
        y
    )

    # Penalización compacta

    compact_penalty = (
        0.001*np.sum(np.abs(individual))
    )

    return error + compact_penalty

# ============================================================
# INICIALIZACIÓN
# ============================================================

bounds = [

    (0.015,0.08), # Bancada1
    (0.015,0.08), # Bancada2

    (0.01,0.08),
    (0.01,0.08),
    (0.01,0.08),
    (0.01,0.08),
    (0.01,0.08),

    (0.005,0.03),
    (0.005,0.03),

    (0.04,0.07),

    (1,4),

    (-np.pi,np.pi)
]

# ============================================================
# CREAR INDIVIDUO
# ============================================================

def create_individual():

    return np.array([

        random.uniform(b[0],b[1])

        for b in bounds
    ])

# ============================================================
# HIPERHEURÍSTICA
# ============================================================

def hyper_heuristic_mutation(ind):

    method = random.choice([1,2])

    child = ind.copy()

    # Mutación suave

    if method == 1:

        idx = random.randint(0,len(ind)-1)

        child[idx] += random.uniform(
            -0.005,
            0.005
        )

    # Mutación agresiva

    else:

        for i in range(len(ind)):

            if random.random() < 0.3:

                child[i] += random.uniform(
                    -0.02,
                    0.02
                )

    return child

# ============================================================
# CROSSOVER
# ============================================================

def crossover(p1,p2):

    child = np.zeros(len(p1))

    method = random.choice([1,2])

    # Promedio

    if method == 1:

        alpha = random.random()

        child = (
            alpha*p1 +
            (1-alpha)*p2
        )

    # Uniforme

    else:

        for i in range(len(p1)):

            if random.random() < 0.5:

                child[i] = p1[i]

            else:

                child[i] = p2[i]

    return child

# ============================================================
# POBLACIÓN INICIAL
# ============================================================

population = [

    create_individual()

    for _ in range(POP_SIZE)
]

# ============================================================
# OPTIMIZACIÓN GENÉTICA
# ============================================================

best_history = []

for gen in range(GENERATIONS):

    fitness_values = np.array([

        fitness(ind)

        for ind in population
    ])

    idx = np.argmin(fitness_values)

    best = population[idx]

    best_fit = fitness_values[idx]

    best_history.append(best_fit)

    print(
        f'Generacion {gen+1} '
        f'| Fitness = {best_fit:.6f}'
    )

    # ========================================================
    # ELITISMO
    # ========================================================

    new_population = [best.copy()]

    while len(new_population) < POP_SIZE:

        # Torneo

        i1,i2 = random.sample(
            range(POP_SIZE),
            2
        )

        parent1 = (
            population[i1]
            if fitness_values[i1]
            <
            fitness_values[i2]
            else population[i2]
        )

        i3,i4 = random.sample(
            range(POP_SIZE),
            2
        )

        parent2 = (
            population[i3]
            if fitness_values[i3]
            <
            fitness_values[i4]
            else population[i4]
        )

        # Crossover

        if random.random() < CROSSOVER_RATE:

            child = crossover(
                parent1,
                parent2
            )

        else:

            child = parent1.copy()

        # Mutación hiperheurística

        if random.random() < MUTATION_RATE:

            child = hyper_heuristic_mutation(
                child
            )

        new_population.append(child)

    population = new_population

# ============================================================
# RESULTADO FINAL
# ============================================================

final_result = kinematic_model(best)

x_best,y_best = final_result

# ============================================================
# GRÁFICAS
# ============================================================

plt.figure(figsize=(8,8))

plt.plot(
    x_target,
    y_target,
    'r--',
    linewidth=3,
    label='Trayectoria Objetivo'
)

plt.plot(
    x_best,
    y_best,
    'b',
    linewidth=3,
    label='Exoesqueleto Optimizado'
)

plt.grid(True)

plt.axis('equal')

plt.legend()

plt.title(
    'Optimización Inicial de la Articulación IFP'
)

plt.show()

# ============================================================
# HISTORIAL FITNESS
# ============================================================

plt.figure(figsize=(8,4))

plt.plot(best_history)

plt.grid(True)

plt.xlabel('Generación')

plt.ylabel('Fitness')

plt.title(
    'Convergencia del Algoritmo Genético'
)

plt.show()

print('\n===================================')
print('OPTIMIZACIÓN FINALIZADA')
print('===================================')

print(f'\nFitness final: {best_fit:.6f}')