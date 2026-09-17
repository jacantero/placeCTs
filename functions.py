import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, nearest_points
from sklearn.cluster import KMeans

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

def generar_grafo_red(calles, posiciones_puntos, potencias_o_ceros, tolerancia=1.5, grafo_inicial=None):
    """
    Usa tu algoritmo original exacto para proyectar puntos (casas o CTs) 
    sobre las calles, segmentar los tramos y devolver el grafo conectado.
    Si se le pasa 'grafo_inicial', clona ese grafo y añade los puntos sobre él.
    """
    # Si no nos dan un grafo base, empezamos con uno vacío (caso del PASO 1)
    if grafo_inicial is None:
        G = nx.Graph()
    else:
        # Clonamos el grafo de forma instantánea (caso del PASO 2)
        G = grafo_inicial.copy()

    proyecciones_por_calle = {idx: [] for idx in range(len(calles))}
    
    # Proyección ortogonal exacta (Tu lógica original)
    for pos, pot in zip(posiciones_puntos, potencias_o_ceros):
        pos_limpia = tuple(map(float, pos))
        geom_punto = Point(pos_limpia)
        
        min_dist = float('inf')
        idx_calle_optima = None
        punto_impacto = None
        
        for idx, calle in enumerate(calles):
            d = geom_punto.distance(calle)
            if d < min_dist:
                min_dist = d
                idx_calle_optima = idx
                # Tu sintaxis original exacta para extraer la coordenada (X, Y)
                punto_impacto = tuple(map(float, nearest_points(calle, geom_punto)[0].coords[0]))
        
        proyecciones_por_calle[idx_calle_optima].append((punto_impacto, pos_limpia, float(min_dist), float(pot)))

    # Mecanismo Snapping original para soldar esquinas por proximidad
    nodos_soldados = {}
    # Si ya hay nodos en el grafo inicial, los registramos para no duplicar esquinas
    if grafo_inicial is not None:
        for n in grafo_inicial.nodes:
            nodos_soldados[n] = n

    def obtener_nodo_limpio(p, tol=tolerancia):
        for n_existente in nodos_soldados:
            if np.linalg.norm(np.array(n_existente) - np.array(p)) <= tol:
                return n_existente
        nodos_soldados[p] = p
        return p

    # Construcción y segmentación original de los tramos
    for idx, calle in enumerate(calles):
        coords_originales = [tuple(map(float, c)) for c in calle.coords]
        proyecciones = proyecciones_por_calle[idx]
        
        for i in range(len(coords_originales) - 1):
            u = obtener_nodo_limpio(coords_originales[i])
            v = obtener_nodo_limpio(coords_originales[i+1])
            
            linea_tramo = LineString([u, v])
            puntos_en_tramo = []
            
            for proj, casa_o_ct, d_acometida, pot in proyecciones:
                if linea_tramo.distance(Point(proj)) < 1e-4:
                    proj_limpio = obtener_nodo_limpio(proj)
                    dist_desde_u = float(np.linalg.norm(np.array(u) - np.array(proj_limpio)))
                    puntos_en_tramo.append((dist_desde_u, proj_limpio, casa_o_ct, d_acometida))
            
            puntos_en_tramo.sort(key=lambda x: x[0])
            
            nodo_actual = u
            for dist_u, proj, casa_o_ct, d_acometida in puntos_en_tramo:
                if nodo_actual != proj:
                    G.add_edge(nodo_actual, proj, weight=float(np.linalg.norm(np.array(nodo_actual) - np.array(proj))))
                
                # Enganche físico de la acometida en el grafo
                G.add_edge(proj, casa_o_ct, weight=d_acometida)
                nodo_actual = proj
                
            if nodo_actual != v:
                G.add_edge(nodo_actual, v, weight=float(np.linalg.norm(np.array(nodo_actual) - np.array(v))))

    # Solo ejecutamos la sanación al crear el grafo base por primera vez (Paso 1)
    # para no penalizar el rendimiento del clon de la iteración.
    if grafo_inicial is None:
        while not nx.is_connected(G):
            # Extraemos todas las islas de calles desconectadas y las ordenamos por tamaño
            componentes = sorted(list(nx.connected_components(G)), key=len, reverse=True)
            isla_principal = componentes[0]
            isla_huerfana = componentes[1] # La primera isla que está aislada
            
            min_dist_puente = float('inf')
            mejor_nodo_principal = None
            mejor_nodo_huerfano = None
            
            # Buscamos los dos puntos más cercanos entre ambas islas para crear una canalización de enlace
            for np_nodo in isla_principal:
                # Evitamos calcular sobre nodos-casa (que no son esquinas viales)
                # Las casas se enviaron como tuplas, pero puedes discriminar si es necesario.
                # Una aproximación segura es calcular la distancia euclidiana entre coordenadas
                np_array_p = np.array(np_nodo)
                
                for nh_nodo in isla_huerfana:
                    dist_puente = np.linalg.norm(np_array_p - np.array(nh_nodo))
                    if dist_puente < min_dist_puente:
                        min_dist_puente = dist_puente
                        mejor_nodo_principal = np_nodo
                        mejor_nodo_huerfano = nh_nodo
            
            # Creamos el puente físico (zanja de cruce) entre las dos zonas urbanas desconectadas
            if mejor_nodo_principal and mejor_nodo_huerfano:
                G.add_edge(mejor_nodo_principal, mejor_nodo_huerfano, weight=float(min_dist_puente))
                print(f"🔗 Reparada isla urbana: Creado puente eléctrico de {min_dist_puente:.2f} metros.")

    return G


def place_CTs(potencias, positions, parcelas, calles):
    potencias = np.array(potencias, dtype=float)
    positions = np.array(positions, dtype=float)
    n_casas = len(positions)
    
    # 1. CÁLCULO DINÁMICO DE CLUSTERS
    pot_CTs = [250, 400, 630, 800]
    max_cap_ct = pot_CTs[-1]
    potencia_total_necesaria = sum(potencias) * 0.4 / 0.9
    n_clusters = int(np.ceil(potencia_total_necesaria / max_cap_ct))
    print(f"Número de clusters a generar: {n_clusters}")

    # =========================================================================
    # PASO 1: Ejecutar una vez el algoritmo para fijar las casas en las calles
    # =========================================================================
    grafo_base_casas = generar_grafo_red(calles, positions, potencias, tolerancia=1.5)

    # 2. INICIALIZACIÓN
    np.random.seed(42)
    indices_iniciales = np.argsort(-potencias)[:n_clusters]
    centros = positions[indices_iniciales].copy()
    orden_casas = np.argsort(-potencias)
    
    max_iter = 50
    labels = np.zeros(n_casas, dtype=int)
    
    # 3. BUCLE ITERATIVO
    for iteracion in range(max_iter):
        labels_antiguos = labels.copy()
        cargas_simultaneas_cts = np.zeros(n_clusters)
        cargas_nominales_cts = np.zeros(n_clusters)
        
        # =========================================================================
        # PASO 2: Clonar el grafo e inyectar dinámicamente los CTs usando la misma lógica
        # =========================================================================
        # Creamos un vector de ceros para los CTs ya que no aportan potencia al grafo, solo se conectan
        ceros_ct = np.zeros(n_clusters)
        grafo_iteracion = generar_grafo_red(calles, centros, ceros_ct, tolerancia=1.5, grafo_inicial=grafo_base_casas)

        # =========================================================================
        # PASO 3: Calcular las distancias mínimas en la red vial resultante
        # =========================================================================
        for i in orden_casas:
            casa_coord = positions[i]
            casa_pot = potencias[i]
            casa_pot_simultanea = casa_pot * 0.4 / 0.9
            
            tupla_casa = tuple(map(float, casa_coord.tolist()))
            
            distancias_viales = np.zeros(n_clusters)
            for c in range(n_clusters):
                tupla_ct = tuple(map(float, centros[c].tolist()))
                
                try:
                    # Al estar ambos perfectamente inyectados como nodos, vamos directo de punto a punto
                    distancias_viales[c] = nx.shortest_path_length(
                        grafo_iteracion, source=tupla_casa, target=tupla_ct, weight='weight'
                    )
                except (nx.NetworkXNoPath, KeyError):
                    print("Fallo")

            
            # El coste real de conectar esta casa a cada centro es Distancia * Potencia
            # Al minimizar este coste, el algoritmo prioriza asignar casas grandes a centros muy cercanos
            costes_linea = distancias_viales * casa_pot
            
            # Ordenamos los centros de menor a mayor coste de cable ponderado
            centros_ordenados_por_coste = np.argsort(costes_linea)
            
            asignado = False
            for c in centros_ordenados_por_coste:
                # Comprobamos la restricción de los 800 kW simultáneos
                if cargas_simultaneas_cts[c] + casa_pot_simultanea <= max_cap_ct:
                    labels[i] = c
                    cargas_simultaneas_cts[c] += casa_pot_simultanea
                    cargas_nominales_cts[c] += casa_pot
                    asignado = True
                    break
            
            # Caso de emergencia por si se llenan todos los centros óptimos
            if not asignado:
                c_con_espacio = np.argmin(cargas_simultaneas_cts)
                labels[i] = c_con_espacio
                cargas_simultaneas_cts[c_con_espacio] += casa_pot_simultanea
                cargas_nominales_cts[c_con_espacio] += casa_pot

        # 4. RECALCULAR CENTROS DE MASAS REALES (Tu fórmula exacta)
        nuevos_centros = np.zeros_like(centros)
        for j in range(n_clusters):
            mascara_grupo = np.where(labels == j)[0]
            potencias_grupo = potencias[mascara_grupo]
            posiciones_grupo = positions[mascara_grupo]
            potencia_total_grupo = sum(potencias_grupo)
            
            if potencia_total_grupo > 0:
                x_ponderada = posiciones_grupo[:, 0] * potencias_grupo
                y_ponderada = posiciones_grupo[:, 1] * potencias_grupo
                
                centro_x_real = sum(x_ponderada) / potencia_total_grupo
                centro_y_real = sum(y_ponderada) / potencia_total_grupo
                nuevos_centros[j] = [centro_x_real, centro_y_real]
            else:
                nuevos_centros[j] = positions[np.random.choice(n_casas)]
                
        if np.array_equal(labels, labels_antiguos):
            break
            
        centros = nuevos_centros

    # 5. IMPRESIÓN DE RESULTADOS Y CÁLCULO DEL COSTE TOTAL DEL CABLEADO
    centros_de_masas_reales = centros
    sumatoria_coste_total = 0
    
    print("\n--- RESULTADOS DE LA OPTIMIZACIÓN (MINIMIZANDO DISTANCIA * POTENCIA) ---")
    for j in range(n_clusters):
        mascara_grupo = np.where(labels == j)[0]
        potencias_grupo = potencias[mascara_grupo]
        posiciones_grupo = positions[mascara_grupo]
        potencia_total_grupo = sum(potencias_grupo)
        pot_simultanea = potencia_total_grupo * 0.4 / 0.9
        
        # Calcular el coste de cable ponderado de este cluster específico
        distancias_grupo = np.linalg.norm(posiciones_grupo - centros_de_masas_reales[j], axis=1)
        coste_cluster = sum(distancias_grupo * potencias_grupo)
        sumatoria_coste_total += coste_cluster
        
        print(f"📊 Grupo {j+1}: {len(potencias_grupo)} parcelas | "
              f"Centro real: X={centros_de_masas_reales[j,0]:.2f}, Y={centros_de_masas_reales[j,1]:.2f} | "
              f"Carga simultánea: {pot_simultanea:.2f} kVA (Máx 800) | "
              f"Momento del cableado: {coste_cluster:.2f} kW·m")

    print(f"\n⚡ Sumatorio total del momento de carga (Mínimo global alcanzado): {sumatoria_coste_total:.2f} kW·m")

    return centros_de_masas_reales, labels, potencia_total_grupo, grafo_iteracion  # Devolvemos también el grafo final para la visualización

def plot_graph(potencias, positions, parcelas, centros_transformacion_ideales, labels, calles, grafo_final):
    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = plt.get_cmap('tab10')

    # 1. DIBUJAR LAS PARCELAS / CALLES BASE (Una sola vez fuera de los bucles)
    for calle in calles:
        x_coords, y_coords = calle.xy
        ax.plot(x_coords, y_coords, color="gray", linewidth=1.5, alpha=0.4, zorder=1)

    # Dibujamos las líneas de las parcelas en color rojo (Una sola vez)
    for parcela in parcelas:
        x_coords, y_coords = parcela.xy
        ax.plot(x_coords, y_coords, color="red", linewidth=1.2, alpha=0.5, zorder=1)

    # 2. CALCULAR Y DIBUJAR LOS CAMINOS ELÉCTRICOS DE CADA CASA A SU CT
    for position, potencia, label in zip(positions, potencias, labels):
        ct_asignado = centros_transformacion_ideales[label]
        color_cluster = cmap(label)
        
        # Convertimos a tuplas puras para buscar en el grafo estructurado
        tupla_casa = tuple(map(float, position.tolist())) if hasattr(position, "tolist") else tuple(map(float, position))
        tupla_ct = tuple(map(float, ct_asignado.tolist())) if hasattr(ct_asignado, "tolist") else tuple(map(float, ct_asignado))
        
        # Recuperamos la ruta exacta nodo a nodo (incluye acometidas e infraestructura vial)
        try:
            camino_nodos = nx.shortest_path(grafo_final, source=tupla_casa, target=tupla_ct, weight='weight')
            camino_coords = np.array(camino_nodos)
            
            # Dibujar la línea de cable continua sobre la calle
            ax.plot(camino_coords[:, 0], camino_coords[:, 1], color=color_cluster, linewidth=2, alpha=0.8, zorder=3)
        except (nx.NetworkXNoPath, KeyError):
            # Línea de respaldo discontinua si hay alguna zona aislada topológicamente
            ax.plot([position[0], ct_asignado[0]], [position[1], ct_asignado[1]], 
                    color=color_cluster, linestyle="--", linewidth=1.2, alpha=0.4, zorder=2)

        # Dibujar los puntos de las casas
        ax.scatter(position[0], position[1], marker='x', color=color_cluster, s=45, zorder=4)
        ax.annotate(f"{potencia} kW", (position[0], position[1]), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8, color=color_cluster, weight='bold')

    # 3. DIBUJAR LOS CENTROS DE TRANSFORMACIÓN REALES (CTs)
    for idx, centro in enumerate(centros_transformacion_ideales):
        color_ct = cmap(idx)
        potencia_total = sum([potencias[i] for i in range(len(potencias)) if labels[i] == idx])
        pot_simultanea = potencia_total * 0.4 / 0.9
        
        # Dibujamos el CT con un marcador cuadrado destacado con borde negro
        ax.scatter(centro[0], centro[1], marker='s', color=color_ct, s=140, edgecolor='black', linewidth=1.5, zorder=5)
        ax.annotate(f"⚡ CT {idx+1}\n{pot_simultanea:.1f} kVA", (centro[0], centro[1]), textcoords="offset points", 
                    xytext=(0,12), ha='center', fontsize=9, color='black', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_ct, alpha=0.9))

    ax.set_aspect('equal')
    plt.title("Trazado Eléctrico Optimizado por Calles con Restricción de Capacidad", fontsize=12, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
