"""
Ubicacion de centros de transformacion (CT) y trazado de acometidas sobre red vial.
 
Objetivo optimizado de forma explicita y coherente en todo el codigo:
 
    minimizar  SUM_i  P_i * d_red(casa_i, CT(casa_i))          [kW*m]
    sujeto a   SUM_{i en CT_j} P_i * ks / cos_phi  <=  cap_max  para todo j
 
donde d_red es la distancia recorrida por las calles (no la euclidea) y los CT
solo pueden ubicarse sobre la red vial.
 
Diferencias de fondo frente a la version basada en k-means ponderado:
 
  * La actualizacion de la ubicacion del CT es la 1-mediana discreta sobre el
    grafo (argmin sobre nodos de calle de SUM P_i*d_red). El centro de masas
    ponderado minimiza SUM P_i*d^2 euclidea, que es otro problema y sesga el CT
    hacia la nube de casas en vez de hacia el optimo de momento electrico.
  * El grafo se construye UNA sola vez y las distancias casa->nodo se
    precalculan con un Dijkstra por punto de acometida, por lo que iterar es
    aritmetica vectorizada y no miles de shortest_path.
  * La asignacion respeta capacidad con un greedy por arrepentimiento
    (regret) seguido de busqueda local (reubicacion + intercambio), en vez de
    un greedy de una pasada por potencia descendente.
  * Nunca se asigna coste 0 a un CT inalcanzable (era el fallo que provocaba
    los cableados absurdos): la distancia es inf y el CT queda descartado.
"""
 
from __future__ import annotations
 
import math
from collections import defaultdict
 
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
 
POT_CTS = (250.0, 400.0, 630.0, 800.0)
KS = 0.4          # coeficiente de simultaneidad
COS_PHI = 0.9     # factor de potencia

import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd

def get_power(pandas_df):
    potencias = []
    positions = []
    for index, row in pandas_df.iterrows():
        str_potencia = str(row['Contenido']).lower()
        if "kw" in str_potencia:
            potencia = str_potencia.split("kw")[0].strip()
            position = [row['Posición X'], row['Posición Y']]

            print(f"Potencia: {potencia} kW, Posición: {position}")
            potencias.append(float(potencia.replace(",", ".")))  # Convertimos a float y reemplazamos coma por punto si es necesario
            positions.append(position)
    return potencias, positions

def get_parcelas(pandas_df):
    parcelas = []
    for index, row in pandas_df.iterrows():
        if not pd.isna(row['Inicial X']):
            x_coords = [row['Inicial X'], row['Fin X']]
            y_coords = [row['Inicial Y'], row['Fin Y']]
            line = LineString(zip(x_coords, y_coords))
            parcelas.append(line)
        elif not pd.isna(row['Centro X']):
            cx = row['Centro X']
            cy = row['Centro Y']
            ang_inicio = row['Ángulo inicial']
            ang_total = row['Ángulo total']
            long_arc = row['Longitud']
            
            # Generamos 20 puntos intermedios para que la curva se vea suave
            angulos_rad = np.radians(ang_total)
            if angulos_rad > np.pi/36:
                r = long_arc / angulos_rad # Radio del arco

                angulos_rad = np.linspace(np.radians(ang_inicio), np.radians(ang_inicio + ang_total), 20)
                
                # Trigonometría básica: X = Cx + R*cos(θ), Y = Cy + R*sin(θ)
                x_coords = cx + r * np.cos(angulos_rad)
                y_coords = cy + r * np.sin(angulos_rad)

                line = LineString(zip(x_coords, y_coords))
                parcelas.append(line)
    return parcelas

def get_calles(pandas_df):
    calles = []
    for index, row in pandas_df.iterrows():
        if not pd.isna(row['Inicial X']):
            # Forzamos float nativo de Python para que Shapely y NetworkX no hereden objetos NumPy
            x_coords = [float(row['Inicial X']), float(row['Fin X'])]
            y_coords = [float(row['Inicial Y']), float(row['Fin Y'])]
            line = LineString(zip(x_coords, y_coords))
            calles.append(line)
    return calles

# ---------------------------------------------------------------------------
# Construccion de la red vial
# ---------------------------------------------------------------------------
 
def _snapper(tol):
    """Soldadura de nodos por proximidad con hash espacial (O(1) por consulta)."""
    celdas = defaultdict(list)
 
    def snap(p):
        cx, cy = int(math.floor(p[0] / tol)), int(math.floor(p[1] / tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in celdas[(cx + dx, cy + dy)]:
                    if math.hypot(q[0] - p[0], q[1] - p[1]) <= tol:
                        return q
        celdas[(cx, cy)].append(p)
        return p
 
    return snap
 
 
def construir_red(calles, positions, paso_candidatos=10.0, tolerancia=1.5):
    """Devuelve (G, nodos_casa, nodos_calle).
 
    - Cada calle se trocea en sus vertices, en las proyecciones de las casas y
      en puntos equiespaciados cada `paso_candidatos` metros (ubicaciones
      candidatas para los CT).
    - El peso de cada tramo es la longitud REAL recorrida sobre la polilinea.
    - Cada casa cuelga de su proyeccion con el peso de su acometida.
    """
    G = nx.Graph()
    snap = _snapper(tolerancia)
    cortes = defaultdict(set)
    proy_casa = []
 
    for pos in positions:
        p = Point(float(pos[0]), float(pos[1]))
        mejor = min(range(len(calles)), key=lambda k: p.distance(calles[k]))
        calle = calles[mejor]
        q = nearest_points(calle, p)[0]
        s = float(calle.project(q))
        cortes[mejor].add(s)
        proy_casa.append((mejor, s, float(p.distance(calle))))
 
    for idx, calle in enumerate(calles):
        L = float(calle.length)
        cortes[idx].update({0.0, L})
        acc = 0.0
        coords = list(calle.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            acc += math.dist(a, b)
            cortes[idx].add(min(acc, L))
        if paso_candidatos and paso_candidatos > 0:
            n = int(L // paso_candidatos)
            cortes[idx].update(k * paso_candidatos for k in range(1, n + 1))
 
    nodo_de = {}          # (idx_calle, s) -> nodo
    for idx, calle in enumerate(calles):
        ss = sorted(cortes[idx])
        nodos = []
        for s in ss:
            xy = calle.interpolate(s)
            n = snap((float(xy.x), float(xy.y)))
            if n not in G:
                G.add_node(n, pos=n, tipo="calle")
            nodo_de[(idx, s)] = n
            nodos.append(n)
        for (s0, n0), (s1, n1) in zip(zip(ss, nodos), zip(ss[1:], nodos[1:])):
            if n0 != n1:
                w = s1 - s0
                if not G.has_edge(n0, n1) or G[n0][n1]["weight"] > w:
                    G.add_edge(n0, n1, weight=w, tipo="calle")
 
    _conectar_islas(G)
 
    nodos_casa = []
    for i, (idx, s, d) in enumerate(proy_casa):
        nc = ("CASA", i)
        G.add_node(nc, pos=(float(positions[i][0]), float(positions[i][1])), tipo="casa")
        G.add_edge(nc, nodo_de[(idx, s)], weight=d, tipo="acometida")
        nodos_casa.append(nc)
 
    nodos_calle = [n for n, d in G.nodes(data=True) if d["tipo"] == "calle"]
    return G, nodos_casa, nodos_calle
 
 
def _conectar_islas(G):
    """Une componentes desconectadas por el par de nodos de CALLE mas cercano."""
    while not nx.is_connected(G):
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        principal = np.array([n for n in comps[0] if G.nodes[n]["tipo"] == "calle"], dtype=float)
        ids_p = [n for n in comps[0] if G.nodes[n]["tipo"] == "calle"]
        mejor = (float("inf"), None, None)
        for comp in comps[1:]:
            ids_h = [n for n in comp if G.nodes[n]["tipo"] == "calle"]
            if not ids_h or len(principal) == 0:
                continue
            H = np.array(ids_h, dtype=float)
            d = np.linalg.norm(principal[:, None, :] - H[None, :, :], axis=2)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] < mejor[0]:
                mejor = (float(d[i, j]), ids_p[i], ids_h[j])
        if mejor[1] is None:
            break
        G.add_edge(mejor[1], mejor[2], weight=mejor[0], tipo="enlace")
 
 
# ---------------------------------------------------------------------------
# Matriz de distancias casa -> nodo de calle (una sola vez)
# ---------------------------------------------------------------------------
 
def matriz_distancias(G, nodos_casa, nodos_calle):
    idx_nodo = {n: k for k, n in enumerate(nodos_calle)}
    D = np.full((len(nodos_casa), len(nodos_calle)), np.inf)
 
    cache = {}
    for i, nc in enumerate(nodos_casa):
        (raiz, acom), = ((v, dd["weight"]) for v, dd in G[nc].items())
        if raiz not in cache:
            cache[raiz] = nx.single_source_dijkstra_path_length(G, raiz, weight="weight")
        for n, d in cache[raiz].items():
            k = idx_nodo.get(n)
            if k is not None:
                D[i, k] = d + acom
    return D
 
 
# ---------------------------------------------------------------------------
# Asignacion capacitada
# ---------------------------------------------------------------------------
 
def _asignar(costes, carga_sim, cap):
    """Greedy por arrepentimiento + busqueda local. costes[i, j] = P_i*d_ij."""
    n, k = costes.shape
    labels = np.full(n, -1)
    usado = np.zeros(k)
 
    orden_pref = np.argsort(costes, axis=1)
    pendientes = set(range(n))
    while pendientes:
        mejor_i, mejor_j, mejor_regret = None, None, -np.inf
        for i in pendientes:
            factibles = [j for j in orden_pref[i]
                         if np.isfinite(costes[i, j]) and usado[j] + carga_sim[i] <= cap]
            if not factibles:
                continue
            c1 = costes[i, factibles[0]]
            c2 = costes[i, factibles[1]] if len(factibles) > 1 else c1 * 2 + 1.0
            regret = c2 - c1
            if regret > mejor_regret:
                mejor_i, mejor_j, mejor_regret = i, factibles[0], regret
        if mejor_i is None:
            return None, np.inf  # no cabe: hacen falta mas CT
        labels[mejor_i] = mejor_j
        usado[mejor_j] += carga_sim[mejor_i]
        pendientes.discard(mejor_i)
 
    labels, usado = _busqueda_local(costes, carga_sim, cap, labels, usado)
    return labels, float(costes[np.arange(n), labels].sum())
 
 
def _busqueda_local(costes, carga_sim, cap, labels, usado, max_pasadas=30):
    n, k = costes.shape
    for _ in range(max_pasadas):
        mejora = False
        for i in range(n):
            a = labels[i]
            for j in range(k):
                if j == a or not np.isfinite(costes[i, j]):
                    continue
                if usado[j] + carga_sim[i] <= cap and costes[i, j] < costes[i, a] - 1e-9:
                    labels[i] = j
                    usado[a] -= carga_sim[i]
                    usado[j] += carga_sim[i]
                    a, mejora = j, True
        for i in range(n):
            for j in range(i + 1, n):
                a, b = labels[i], labels[j]
                if a == b:
                    continue
                delta = (costes[i, b] + costes[j, a]) - (costes[i, a] + costes[j, b])
                if not np.isfinite(delta) or delta >= -1e-9:
                    continue
                ua = usado[a] - carga_sim[i] + carga_sim[j]
                ub = usado[b] - carga_sim[j] + carga_sim[i]
                if ua <= cap and ub <= cap:
                    labels[i], labels[j] = b, a
                    usado[a], usado[b] = ua, ub
                    mejora = True
        if not mejora:
            break
    return labels, usado
 
 
# ---------------------------------------------------------------------------
# Algoritmo principal
# ---------------------------------------------------------------------------
 
def place_CTs(potencias, positions, parcelas, calles, paso_candidatos=10.0,
              tolerancia=1.5, utilizacion_max=1.0, max_iter=50, verbose=True):
    potencias = np.asarray(potencias, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
 
    carga_sim = potencias * KS / COS_PHI
    cap = POT_CTS[-1] * utilizacion_max
 
    G, nodos_casa, nodos_calle = construir_red(calles, positions, paso_candidatos, tolerancia)
    D = matriz_distancias(G, nodos_casa, nodos_calle)          # [n x m] metros
    C = D * potencias[:, None]                                  # [n x m] kW*m
 
    k = max(1, int(np.ceil(carga_sim.sum() / cap)))
    while True:
        res = _resolver_k(D, C, carga_sim, cap, k, potencias, max_iter)
        if res is not None:
            break
        k += 1
        if verbose:
            print(f"Capacidad insuficiente o geometria incompatible: probando con {k} CT")
 
    labels, idx_cts, coste = res
    centros = np.array([nodos_calle[j] for j in idx_cts], dtype=float)
    nodos_ct = [nodos_calle[j] for j in idx_cts]
 
    resumen = _informe(labels, potencias, carga_sim, D, idx_cts, coste, centros, verbose)
    return centros, labels, resumen, G, nodos_casa, nodos_ct
 
 
def _resolver_k(D, C, carga_sim, cap, k, potencias, max_iter):
    """k-mediana capacitada sobre la red: init k-means++ ponderado + Lloyd discreto."""
    n, m = C.shape
    rng = np.random.default_rng(42)
    nodo_de_casa = np.argmin(D, axis=1)          # nodo de acometida de cada casa
 
    centros = [int(np.argmin(np.where(np.isfinite(C), C, np.inf).sum(axis=0)))]
    while len(centros) < k:
        dmin = np.min(D[:, centros], axis=1)
        w = np.nan_to_num(potencias * dmin ** 2, posinf=0.0, nan=0.0)
        i = int(rng.choice(n, p=w / w.sum())) if w.sum() > 0 else int(rng.integers(n))
        cand = int(nodo_de_casa[i])
        if cand in centros:
            cand = int(rng.integers(m))
        centros.append(cand)
 
    mejor = None
    for _ in range(max_iter):
        labels, coste = _asignar(C[:, centros], carga_sim, cap)
        if labels is None:
            return None if mejor is None else mejor
        if mejor is None or coste < mejor[2] - 1e-6:
            mejor = (labels.copy(), list(centros), coste)
 
        # 1-mediana discreta por cluster sobre TODOS los nodos de calle
        nuevos = []
        for j in range(k):
            mask = labels == j
            if not mask.any():
                nuevos.append(centros[j])
                continue
            col = np.where(np.isfinite(C[mask]), C[mask], np.inf).sum(axis=0)
            nuevos.append(int(np.argmin(col)))
        if nuevos == centros:
            break
        centros = nuevos
 
    return mejor
 
 
def _informe(labels, potencias, carga_sim, D, idx_cts, coste, centros, verbose):
    filas = []
    for j in range(len(idx_cts)):
        mask = labels == j
        sim = float(carga_sim[mask].sum())
        nominal = next((p for p in POT_CTS if p >= sim), POT_CTS[-1])
        momento = float((D[mask, idx_cts[j]] * potencias[mask]).sum())
        filas.append({
            "ct": j + 1,
            "parcelas": int(mask.sum()),
            "x": float(centros[j][0]),
            "y": float(centros[j][1]),
            "pot_instalada_kW": float(potencias[mask].sum()),
            "carga_simultanea_kVA": sim,
            "ct_normalizado_kVA": nominal,
            "utilizacion": sim / nominal if nominal else 0.0,
            "momento_kWm": momento,
            "long_media_m": float(D[mask, idx_cts[j]].mean()) if mask.any() else 0.0,
            "long_max_m": float(D[mask, idx_cts[j]].max()) if mask.any() else 0.0,
        })
    if verbose:
        print("\n--- CT sobre red vial | objetivo SUM P*d_red ---")
        for f in filas:
            print(f"CT {f['ct']}: {f['parcelas']:3d} parcelas | X={f['x']:.2f} Y={f['y']:.2f} | "
                  f"{f['carga_simultanea_kVA']:7.1f} kVA -> CT {f['ct_normalizado_kVA']:.0f} "
                  f"({f['utilizacion']*100:.0f}%) | momento {f['momento_kWm']:.0f} kW.m | "
                  f"L media {f['long_media_m']:.0f} m / max {f['long_max_m']:.0f} m")
        print(f"\nMomento total: {coste:.2f} kW.m")
    return {"ct": filas, "momento_total_kWm": float(coste)}
 
 
# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
 
def plot_graph(potencias, positions, parcelas, centros, labels, calles, G,
               nodos_casa, nodos_ct, ruta=None):
    if ruta:
        matplotlib.use("Agg")
 
    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = plt.get_cmap("tab10")
 
    for calle in calles:
        x, y = calle.xy
        ax.plot(x, y, color="gray", linewidth=1.5, alpha=0.4, zorder=1)
    for parcela in parcelas or []:
        x, y = parcela.xy
        ax.plot(x, y, color="red", linewidth=1.0, alpha=0.4, zorder=1)
 
    for i, (pos, pot, lab) in enumerate(zip(positions, potencias, labels)):
        color = cmap(int(lab) % 10)
        try:
            camino = nx.shortest_path(G, nodos_casa[i], nodos_ct[int(lab)], weight="weight")
            xy = np.array([G.nodes[n]["pos"] for n in camino], dtype=float)
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2, alpha=0.85, zorder=3)
        except (nx.NetworkXNoPath, KeyError):
            ax.plot([pos[0], centros[int(lab)][0]], [pos[1], centros[int(lab)][1]],
                    color=color, linestyle="--", linewidth=1.2, alpha=0.4, zorder=2)
        ax.scatter(pos[0], pos[1], marker="x", color=color, s=40, zorder=4)
 
    for j, centro in enumerate(centros):
        color = cmap(j % 10)
        sim = float(np.array(potencias)[np.array(labels) == j].sum()) * KS / COS_PHI
        ax.scatter(centro[0], centro[1], marker="s", color=color, s=150,
                   edgecolor="black", linewidth=1.5, zorder=5)
        ax.annotate(f"CT {j+1}\n{sim:.0f} kVA", (centro[0], centro[1]),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=9, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))
 
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.title("CT sobre red vial - minimizacion de SUM P*d", weight="bold")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta, dpi=110)
    else:
        plt.show()
    return ruta