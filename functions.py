"""
Ubicacion de centros de transformacion (CT) y trazado de acometidas sobre red vial.
 
Objetivo optimizado de forma explicita y coherente en todo el codigo:
 
    minimizar  SUM_i  P_i * c_red(casa_i, CT(casa_i))          [kW*m equivalentes]
    sujeto a   SUM_{i en CT_j} P_i * ks / cos_phi  <=  cap_max  para todo j
 
donde c_red es la longitud recorrida por las calles afectada por el factor de
penalizacion de cada calle (1.0 en calle principal, `penalizacion` en
secundaria). La longitud REAL de cada linea se sigue calculando y reportando
aparte (es la que importa para caida de tension y para el presupuesto).
 
Restricciones de implantacion:
 
  * Los CT se ubican DENTRO de una parcela, retranqueados `retranqueo` metros
    de un lindero que da a calle (nunca sobre la calzada ni en el centro de la
    manzana).
  * Una calle es PRINCIPAL si la separacion entre su eje y las parcelas
    colindantes es >= `umbral_principal` (20 m por defecto); en caso contrario
    es secundaria y sus tramos se penalizan, de modo que el trazado y los CT
    se apoyan en viario principal salvo que el rodeo salga caro.
 
Frente a la version basada en k-means ponderado:
 
  * La recolocacion del CT es la 1-mediana discreta sobre el grafo (argmin de
    SUM P_i*c_red sobre los candidatos), no el baricentro ponderado (que
    minimiza SUM P*d^2 euclidea, otro problema distinto).
  * El grafo se construye UNA sola vez y las distancias casa->candidato se
    precalculan con un Dijkstra por punto de acometida.
  * La asignacion respeta capacidad con greedy por arrepentimiento + busqueda
    local (reubicacion e intercambio).
  * Un CT inalcanzable tiene coste inf, nunca 0.
"""
 
from __future__ import annotations
 
import math
from collections import defaultdict
 
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString, LinearRing, Polygon
from shapely.ops import nearest_points, unary_union
 
POT_CTS = (250.0, 400.0, 630.0, 800.0)
KS = 0.4          # coeficiente de simultaneidad
COS_PHI = 0.9     # factor de potencia
 
 
# ---------------------------------------------------------------------------
# Utilidades geometricas

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
# Utilidades geometricas
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
 
 
def _poligono(parcela):
    """Devuelve el Polygon de una parcela dada como ring/linea cerrada, o None."""
    if isinstance(parcela, Polygon):
        return parcela
    coords = list(parcela.coords)
    if len(coords) >= 4 and math.dist(coords[0], coords[-1]) < 1e-6:
        try:
            p = Polygon(LinearRing(coords))
            return p if p.is_valid and p.area > 0 else None
        except Exception:
            return None
    return None
 
 
def clasificar_calles(calles, parcelas, umbral_principal=20.0, paso_muestreo=5.0):
    """True = calle principal.
 
    Criterio del proyecto: la calle es principal cuando la separacion entre el
    eje y las parcelas colindantes es de al menos `umbral_principal` metros.
    Se usa la MEDIANA de la separacion muestreada a lo largo de la calle, para
    que un encuentro puntual con una esquina no degrade la clasificacion.
    """
    if not parcelas:
        return [True] * len(calles)
    union = unary_union([p.boundary if isinstance(p, Polygon) else p for p in parcelas])
    principales = []
    for calle in calles:
        L = calle.length
        n = max(2, int(L // paso_muestreo) + 1)
        seps = [union.distance(calle.interpolate(k * L / (n - 1))) for k in range(n)]
        principales.append(float(np.median(seps)) >= umbral_principal)
    return principales
 
 
def candidatos_en_parcelas(parcelas, calles, retranqueo=3.0, paso=10.0,
                           dist_max_calle=40.0):
    """Puntos candidatos para CT: dentro de la parcela, pegados a un lindero
    que da a calle.
 
    Para cada lindero se muestrea cada `paso` metros y el punto se desplaza
    `retranqueo` metros hacia el interior de la parcela (si la parcela no es un
    poligono cerrado, hacia el lado opuesto a la calle). Se descartan los
    linderos que no dan a calle (> `dist_max_calle`).
 
    Devuelve tuplas (x, y, j) donde `j` es la calle a la que da frente ese
    lindero: es la que decide si el CT queda a pie de calle principal.
    """
    red = unary_union(calles)
    candidatos = []
    for parcela in parcelas:
        poly = _poligono(parcela)
        borde = parcela.exterior if isinstance(parcela, Polygon) else parcela
        L = borde.length
        if L <= 0:
            continue
        n = max(1, int(L // paso))
        for k in range(n):
            s = (k + 0.5) * L / n
            p = borde.interpolate(s)
            d_calle = red.distance(p)
            if d_calle > dist_max_calle:
                continue
            j_frente = min(range(len(calles)), key=lambda q: p.distance(calles[q]))
            # normal al lindero en ese punto
            eps = min(1.0, L / 100.0)
            a = borde.interpolate(max(0.0, s - eps))
            b = borde.interpolate(min(L, s + eps))
            tx, ty = b.x - a.x, b.y - a.y
            norm = math.hypot(tx, ty)
            if norm < 1e-9:
                continue
            nx_, ny_ = -ty / norm, tx / norm
            c1 = Point(p.x + nx_ * retranqueo, p.y + ny_ * retranqueo)
            c2 = Point(p.x - nx_ * retranqueo, p.y - ny_ * retranqueo)
            if poly is not None:
                dentro = [c for c in (c1, c2) if poly.contains(c)]
                if not dentro:
                    continue
                c = dentro[0]
            else:
                # sin poligono: el interior es el lado contrario a la calle
                c = c1 if red.distance(c1) > red.distance(c2) else c2
            candidatos.append((float(c.x), float(c.y), j_frente))
    return candidatos
 
 
# ---------------------------------------------------------------------------
# Construccion de la red
# ---------------------------------------------------------------------------
 
def construir_red(calles, positions, parcelas=(), retranqueo=3.0,
                  paso_candidatos=10.0, tolerancia=1.5,
                  umbral_principal=20.0, penalizacion=1.6,
                  calles_principales=None, dist_max_calle=40.0):
    """Grafo unico con casas, candidatos a CT y viario.
 
    Cada arista lleva:
      weight -> longitud real en metros
      coste  -> longitud penalizada (metros * factor de la calle)
    """
    if calles_principales is None:
        calles_principales = clasificar_calles(calles, list(parcelas), umbral_principal)
    factor = [1.0 if pr else float(penalizacion) for pr in calles_principales]
 
    puntos_ct = candidatos_en_parcelas(list(parcelas), calles, retranqueo,
                                       paso_candidatos, dist_max_calle) if parcelas else []
 
    G = nx.Graph()
    snap = _snapper(tolerancia)
    cortes = defaultdict(set)
 
    def proyectar(xy, j=None):
        p = Point(float(xy[0]), float(xy[1]))
        if j is None:
            j = min(range(len(calles)), key=lambda k: p.distance(calles[k]))
        q = nearest_points(calles[j], p)[0]
        return j, float(calles[j].project(q)), float(p.distance(calles[j]))
 
    proy_casa = [proyectar(pos) for pos in positions]
    # el CT se engancha a SU calle de frente, no a la mas cercana en linea recta
    proy_ct = [proyectar(c[:2], c[2]) for c in puntos_ct]
    for j, s, _ in proy_casa + proy_ct:
        cortes[j].add(s)
 
    for idx, calle in enumerate(calles):
        L = float(calle.length)
        cortes[idx].update({0.0, L})
        acc = 0.0
        coords = list(calle.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            acc += math.dist(a, b)
            cortes[idx].add(min(acc, L))
        if paso_candidatos and paso_candidatos > 0:
            cortes[idx].update(k * paso_candidatos for k in range(1, int(L // paso_candidatos) + 1))
 
    nodo_de = {}
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
            if n0 == n1:
                continue
            w = s1 - s0
            c = w * factor[idx]
            if not G.has_edge(n0, n1) or G[n0][n1]["coste"] > c:
                G.add_edge(n0, n1, weight=w, coste=c, tipo="calle",
                           principal=calles_principales[idx])
 
    _conectar_islas(G, penalizacion)
 
    nodos_casa = []
    for i, (idx, s, d) in enumerate(proy_casa):
        nc = ("CASA", i)
        G.add_node(nc, pos=(float(positions[i][0]), float(positions[i][1])), tipo="casa")
        G.add_edge(nc, nodo_de[(idx, s)], weight=d, coste=d, tipo="acometida")
        nodos_casa.append(nc)
 
    nodos_ct = []
    vistos = set()
    for i, (idx, s, d) in enumerate(proy_ct):
        clave = nodo_de[(idx, s)], round(puntos_ct[i][0], 2), round(puntos_ct[i][1], 2)
        if clave in vistos:
            continue
        vistos.add(clave)
        nct = ("CT", i)
        G.add_node(nct, pos=puntos_ct[i][:2], tipo="ct_cand",
                   principal=calles_principales[idx], calle=idx)
        # el enlace CT-calle atraviesa el retranqueo: se penaliza como la calle
        G.add_edge(nct, nodo_de[(idx, s)], weight=d, coste=d * factor[idx], tipo="enlace_ct")
        nodos_ct.append(nct)
 
    return G, nodos_casa, nodos_ct, calles_principales
 
 
def _conectar_islas(G, penalizacion):
    """Une componentes por el par de nodos de CALLE mas cercano."""
    while not nx.is_connected(G):
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        ids_p = [n for n in comps[0] if G.nodes[n]["tipo"] == "calle"]
        if not ids_p:
            break
        principal = np.array(ids_p, dtype=float)
        mejor = (float("inf"), None, None)
        for comp in comps[1:]:
            ids_h = [n for n in comp if G.nodes[n]["tipo"] == "calle"]
            if not ids_h:
                continue
            H = np.array(ids_h, dtype=float)
            d = np.linalg.norm(principal[:, None, :] - H[None, :, :], axis=2)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] < mejor[0]:
                mejor = (float(d[i, j]), ids_p[i], ids_h[j])
        if mejor[1] is None:
            break
        G.add_edge(mejor[1], mejor[2], weight=mejor[0],
                   coste=mejor[0] * penalizacion, tipo="enlace")
 
 
# ---------------------------------------------------------------------------
# Matriz de distancias casa -> candidato (una sola vez)
# ---------------------------------------------------------------------------
 
def matriz_distancias(G, nodos_casa, candidatos, peso="coste"):
    idx_nodo = {n: k for k, n in enumerate(candidatos)}
    D = np.full((len(nodos_casa), len(candidatos)), np.inf)
    cache = {}
    for i, nc in enumerate(nodos_casa):
        (raiz, dd), = ((v, data) for v, data in G[nc].items())
        acom = dd[peso]
        if raiz not in cache:
            cache[raiz] = nx.single_source_dijkstra_path_length(G, raiz, weight=peso)
        for n, d in cache[raiz].items():
            k = idx_nodo.get(n)
            if k is not None:
                D[i, k] = d + acom
    return D
 
 
# ---------------------------------------------------------------------------
# Asignacion capacitada
# ---------------------------------------------------------------------------
 
def _asignar(costes, carga_sim, cap):
    """Greedy por arrepentimiento + busqueda local. costes[i, j] = P_i*c_ij."""
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
            if c2 - c1 > mejor_regret:
                mejor_i, mejor_j, mejor_regret = i, factibles[0], c2 - c1
        if mejor_i is None:
            return None, np.inf
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
 
def place_CTs(potencias, positions, parcelas, calles, retranqueo=3.0,
              paso_candidatos=10.0, tolerancia=1.5, umbral_principal=20.0,
              penalizacion=1.6, calles_principales=None, utilizacion_max=1.0,
              ct_solo_principal=True, radio_fallback_principal=150.0,
              max_iter=50, verbose=True):
    """`ct_solo_principal`: el CT se implanta en parcela con frente a calle
    principal; solo cae a un frente secundario cuando ese grupo no tiene
    ninguna parcela con frente principal a menos de `radio_fallback_principal`
    metros de red (None = sin escapatoria, principal siempre).
    """
    potencias = np.asarray(potencias, dtype=float)
    positions = np.asarray(positions, dtype=float)
    carga_sim = potencias * KS / COS_PHI
    cap = POT_CTS[-1] * utilizacion_max
 
    G, nodos_casa, cands, principales = construir_red(
        calles, positions, parcelas, retranqueo, paso_candidatos, tolerancia,
        umbral_principal, penalizacion, calles_principales)
    if not cands:
        raise ValueError("No hay candidatos de CT: revisa las parcelas o dist_max_calle")
 
    pr_cand = np.array([bool(G.nodes[n].get("principal")) for n in cands])
    if ct_solo_principal and not pr_cand.any():
        ct_solo_principal = False
        if verbose:
            print("Aviso: ninguna parcela tiene frente a calle principal; "
                  "se admiten frentes secundarios")
    if verbose:
        n_pr = sum(principales)
        print(f"{len(cands)} ubicaciones candidatas en parcela "
              f"({int(pr_cand.sum())} con frente principal) | "
              f"{n_pr}/{len(calles)} calles principales (separacion >= {umbral_principal} m) | "
              f"penalizacion secundaria x{penalizacion}")
 
    D_real = matriz_distancias(G, nodos_casa, cands, peso="weight")
    C_pen = matriz_distancias(G, nodos_casa, cands, peso="coste") * potencias[:, None]
    permitidos = _permiso_principal(pr_cand, D_real, ct_solo_principal,
                                    radio_fallback_principal)
 
    k = max(1, int(np.ceil(carga_sim.sum() / cap)))
    while True:
        res = _resolver_k(C_pen, carga_sim, cap, k, potencias, max_iter, permitidos)
        if res is not None:
            break
        k += 1
        if verbose:
            print(f"Sin solucion factible con {k-1} CT: probando con {k}")
 
    labels, idx_cts, coste_pen = res
    nodos_ct = [cands[j] for j in idx_cts]
    centros = np.array([G.nodes[n]["pos"] for n in nodos_ct], dtype=float)
 
    largos, pen_tramo = _longitudes_reales(G, nodos_casa, nodos_ct, labels)
    resumen = _informe(labels, potencias, carga_sim, largos, pen_tramo, coste_pen,
                       centros, nodos_ct, G, verbose)
    return centros, labels, resumen, G, nodos_casa, nodos_ct
 
 
def _permiso_principal(pr_cand, D_real, ct_solo_principal, radio):
    """Devuelve una funcion grupo -> candidatos admisibles para su CT.
 
    Con `ct_solo_principal` solo se admiten parcelas con frente a calle
    principal; se abre a frentes secundarios unicamente cuando el grupo no
    tiene ninguna parcela con frente principal a menos de `radio` metros de
    red (es decir, cuando la opcion de calle principal no existe).
    """
    idx_pr = np.flatnonzero(pr_cand)
    todos = np.arange(len(pr_cand))
    if not ct_solo_principal:
        return lambda mask: todos
    if radio is None:
        return lambda mask: idx_pr
 
    cerca = D_real[:, idx_pr] <= float(radio)      # casa x candidato principal
 
    def permitidos(mask):
        if not mask.any():
            return idx_pr
        alcanzables = idx_pr[cerca[mask].any(axis=0)]
        return alcanzables if alcanzables.size else todos
 
    return permitidos
 
 
def _resolver_k(C, carga_sim, cap, k, potencias, max_iter, permitidos=None):
    """k-mediana capacitada: init k-means++ ponderado + Lloyd discreto."""
    n, m = C.shape
    rng = np.random.default_rng(42)
    if permitidos is None:
        todos = np.arange(m)
        permitidos = lambda mask: todos
    global_adm = permitidos(np.ones(n, dtype=bool))
 
    def _mejor(mask):
        """1-mediana discreta del grupo entre sus candidatos admisibles."""
        adm = permitidos(mask)
        col = np.where(np.isfinite(C[mask]), C[mask], np.inf).sum(axis=0)[adm]
        return int(adm[int(np.argmin(col))])
 
    cand_de_casa = np.empty(n, dtype=int)
    for i in range(n):
        una = np.zeros(n, dtype=bool)
        una[i] = True
        cand_de_casa[i] = _mejor(una)      # mejor candidato admisible de cada casa
 
    centros = [_mejor(np.ones(n, dtype=bool))]
    while len(centros) < k:
        cmin = np.min(C[:, centros], axis=1)
        w = np.nan_to_num(cmin ** 2, posinf=0.0, nan=0.0)
        i = int(rng.choice(n, p=w / w.sum())) if w.sum() > 0 else int(rng.integers(n))
        cand = int(cand_de_casa[i])
        if cand in centros:
            libres = [c for c in global_adm if c not in centros]
            cand = int(rng.choice(libres)) if libres else int(rng.integers(m))
        centros.append(cand)
 
    mejor = None
    for _ in range(max_iter):
        labels, coste = _asignar(C[:, centros], carga_sim, cap)
        if labels is None:
            return None if mejor is None else mejor
        if mejor is None or coste < mejor[2] - 1e-6:
            mejor = (labels.copy(), list(centros), coste)
 
        nuevos = []
        for j in range(k):
            mask = labels == j
            if not mask.any():
                nuevos.append(centros[j])
                continue
            nuevos.append(_mejor(mask))
        if nuevos == centros:
            break
        centros = nuevos
 
    return mejor
 
 
def _longitudes_reales(G, nodos_casa, nodos_ct, labels):
    """Longitud real (m) de cada linea por el camino elegido, y % por secundaria."""
    largos = np.zeros(len(nodos_casa))
    sec = np.zeros(len(nodos_casa))
    caches = {}
    for j, nct in enumerate(nodos_ct):
        caches[j] = nx.single_source_dijkstra_path(G, nct, weight="coste")
    for i, nc in enumerate(nodos_casa):
        camino = caches[int(labels[i])].get(nc)
        if not camino:
            largos[i] = np.inf
            continue
        L = Ls = 0.0
        for u, v in zip(camino[:-1], camino[1:]):
            d = G[u][v]
            L += d["weight"]
            if d.get("tipo") == "calle" and not d.get("principal", True):
                Ls += d["weight"]
        largos[i], sec[i] = L, Ls
    return largos, sec
 
 
def _informe(labels, potencias, carga_sim, largos, sec, coste_pen, centros,
             nodos_ct, G, verbose):
    filas = []
    momento_real = float((largos * potencias).sum())
    for j in range(len(centros)):
        mask = labels == j
        sim = float(carga_sim[mask].sum())
        nominal = next((p for p in POT_CTS if p >= sim), POT_CTS[-1])
        filas.append({
            "ct": j + 1,
            "parcelas": int(mask.sum()),
            "x": float(centros[j][0]),
            "y": float(centros[j][1]),
            "en_calle_principal": bool(G.nodes[nodos_ct[j]].get("principal", True)),
            "pot_instalada_kW": float(potencias[mask].sum()),
            "carga_simultanea_kVA": sim,
            "ct_normalizado_kVA": nominal,
            "utilizacion": sim / nominal if nominal else 0.0,
            "momento_real_kWm": float((largos[mask] * potencias[mask]).sum()),
            "long_media_m": float(largos[mask].mean()) if mask.any() else 0.0,
            "long_max_m": float(largos[mask].max()) if mask.any() else 0.0,
            "pct_secundaria": float(100 * sec[mask].sum() / max(largos[mask].sum(), 1e-9)),
        })
    if verbose:
        print("\n--- CT en parcela | objetivo SUM P*longitud penalizada ---")
        for f in filas:
            print(f"CT {f['ct']}: {f['parcelas']:3d} parcelas | X={f['x']:.2f} Y={f['y']:.2f} "
                  f"({'principal' if f['en_calle_principal'] else 'secundaria'}) | "
                  f"{f['carga_simultanea_kVA']:7.1f} kVA -> CT {f['ct_normalizado_kVA']:.0f} "
                  f"({f['utilizacion']*100:.0f}%) | momento real {f['momento_real_kWm']:.0f} kW.m | "
                  f"L media {f['long_media_m']:.0f} m / max {f['long_max_m']:.0f} m | "
                  f"{f['pct_secundaria']:.0f}% por calle secundaria")
        print(f"\nMomento real total : {momento_real:.2f} kW.m")
        print(f"Momento penalizado : {coste_pen:.2f} (funcion objetivo)")
    return {"ct": filas, "momento_real_kWm": momento_real,
            "momento_penalizado": float(coste_pen),
            "pct_longitud_secundaria": float(100 * sec.sum() / max(largos.sum(), 1e-9))}
 
 
# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
 
def plot_graph(potencias, positions, parcelas, centros, labels, calles, G,
               nodos_casa, nodos_ct, calles_principales=None, ruta=None,
               titulo="CT en parcela - viario principal priorizado"):
    import matplotlib
    if ruta:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = plt.get_cmap("tab10")
 
    for idx, calle in enumerate(calles):
        pr = True if calles_principales is None else calles_principales[idx]
        x, y = calle.xy
        ax.plot(x, y, color="dimgray" if pr else "lightgray",
                linewidth=4.0 if pr else 1.5, alpha=0.6, zorder=1,
                solid_capstyle="round")
    for parcela in parcelas or []:
        x, y = (parcela.exterior.xy if hasattr(parcela, "exterior") else parcela.xy)
        ax.plot(x, y, color="indianred", linewidth=1.0, alpha=0.5, zorder=1)
 
    for i, (pos, lab) in enumerate(zip(positions, labels)):
        color = cmap(int(lab) % 10)
        try:
            camino = nx.shortest_path(G, nodos_casa[i], nodos_ct[int(lab)], weight="coste")
            xy = np.array([G.nodes[n]["pos"] for n in camino], dtype=float)
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2, alpha=0.85, zorder=3)
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            ax.plot([pos[0], centros[int(lab)][0]], [pos[1], centros[int(lab)][1]],
                    color=color, linestyle="--", linewidth=1.2, alpha=0.4, zorder=2)
        ax.scatter(pos[0], pos[1], marker="x", color=color, s=40, zorder=4)
 
    pot = np.asarray(potencias, dtype=float)
    lab_arr = np.asarray(labels)
    for j, centro in enumerate(centros):
        color = cmap(j % 10)
        sim = float(pot[lab_arr == j].sum()) * KS / COS_PHI
        ax.scatter(centro[0], centro[1], marker="s", color=color, s=160,
                   edgecolor="black", linewidth=1.5, zorder=5)
        ax.annotate(f"CT {j+1}\n{sim:.0f} kVA", (centro[0], centro[1]),
                    textcoords="offset points", xytext=(0, 13), ha="center",
                    fontsize=9, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))
 
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.title(titulo, weight="bold")
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta, dpi=110)
        plt.close(fig)
    else:
        plt.show()
    return ruta