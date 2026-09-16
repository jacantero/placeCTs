import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import networkx as nx
import pulp
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize
from sklearn.cluster import KMeans
import geopandas as gpd

def get_power(pandas_df):
    potencias = []
    positions = []
    for index, row in pandas_df.iterrows():
        str_potencia = str(row['Contenido']).lower()
        if "kw" in str_potencia:
            potencia = str_potencia.split("kw")[0].strip()
            position = [row['Posición X1'], row['Posición Y1']]

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

def dict_parcelas(pandas_df, posiciones_textos):
    """
    Algoritmo de Cosido Topológico sugerido por el usuario:
    Recorre las líneas y arcos del Excel, y une sus extremos de forma secuencial
    si el principio o el final están a menos de la tolerancia establecida.
    """
    lineas_en_bruto = []
    df_viales = pandas_df[pandas_df['Capa'] == "0-REPARCELACION"].copy()
    
    # 1. Extraemos todas las líneas y arcos del Excel tal y como los tienes
    for index, row in df_viales.iterrows():
        if not pd.isna(row.get('Inicial X')):
            p1 = (row['Inicial X'], row['Inicial Y'])
            p2 = (row['Fin X'], row['Fin Y'])
            lineas_en_bruto.append((p1, p2))
        elif not pd.isna(row.get('Centro X')):
            cx, cy = row['Centro X'], row['Centro Y']
            ang_inicio, ang_total = row['Ángulo inicial'], row['Ángulo total']
            long_arc = row['Longitud']
            r = abs(long_arc / np.radians(ang_total)) if ang_total > 0 else 10
            ang_linspace = np.linspace(np.radians(ang_inicio), np.radians(ang_inicio + ang_total), 30)
            x_coords = cx + r * np.cos(ang_linspace)
            y_coords = cy + r * np.sin(ang_linspace)
            
            # Guardamos los micro-tramos del arco seguidos
            for i in range(len(x_coords) - 1):
                p1 = (x_coords[i], y_coords[i])
                p2 = (x_coords[i+1], y_coords[i+1])
                lineas_en_bruto.append((p1, p2))
   # 2. EL ALGORITMO DE COSIDO (Usando un Grafo Topológico)
    # Para encadenar líneas que se tocan o están muy cerca, creamos un grafo
    # donde los nodos son las coordenadas y las aristas son las líneas.
    G_topo = nx.Graph()
    
    # Diccionario para agrupar nodos que están extremadamente cerca (Snapping de extremos)
    nodos_consolidados = {}
    tolerancia_metros = 0.5  # Distancia máxima para considerar que dos nodos son el mismo
    def obtener_nodo_fijo(punto):
        x, y = punto
        # Buscamos si ya hemos registrado una esquina muy cercana
        for n in nodos_consolidados:
            dist = ((n[0] - x)**2 + (n[1] - y)**2)**0.5
            if dist <= tolerancia_metros:
                return n # Devolvemos la esquina existente para "coserlas" juntas
        # Si está lejos, es una esquina nueva
        nodos_consolidados[(x, y)] = True
        return (x, y)

    # Añadimos los tramos al grafo aplicando el cosido de extremos
    for p1, p2 in lineas_en_bruto:
        p1_fijo = obtener_nodo_fijo(p1)
        p2_fijo = obtener_nodo_fijo(p2)
        
        if p1_fijo != p2_fijo:
            G_topo.add_edge(p1_fijo, p2_fijo)

    # 3. CONVERTIR LAS CADENAS CERRADAS EN POLÍGONOS DE SHAPELY
    # Creamos las LineStrings limpias y ya cosidas
    lineas_cosidas = [LineString([u, v]) for u, v in G_topo.edges()]
    
    # Polygonize se encarga de agrupar las cadenas que forman anillos cerrados
    poligonos_cerrados = list(polygonize(lineas_cosidas))
    
    print(f"🧵 Algoritmo de Cosido finalizado: se han unido los extremos sueltos.")
    print(f"🎉 Se han conseguido cerrar {len(poligonos_cerrados)} parcelas perfectas sin fugas.")

    
    return poligonos_cerrados

def asignar_potencia_a_parcelas_cosidas(poligonos_cerrados, posiciones_textos, potencias):
    from shapely.geometry import Point
    
    parcelas_estructuradas = {}
    
    for idx_txt, pos in enumerate(posiciones_textos):
        punto_texto = Point(pos[0], pos[1])
        polygon_asignado = None
        
        # Buscamos en qué anillo cerrado cae este texto
        for poly in poligonos_cerrados:
            if poly.contains(punto_texto):
                polygon_asignado = poly
                break
                
        # Extraemos las líneas del contorno para tu función de dibujo por colores
        if polygon_asignado:
            ext_coords = list(polygon_asignado.exterior.coords)
            lindes_generadas = [LineString([ext_coords[i], ext_coords[i+1]]) for i in range(len(ext_coords)-1)]
        else:
            lindes_generadas = []
            
        parcelas_estructuradas[idx_txt] = {
            'posicion_texto': pos,
            'poly_obj': polygon_asignado,
            'lindes': lindes_generadas
        }
        
    return parcelas_estructuradas

def plot_graph_cosido(parcelas_estructuradas, potencias, lineas_totales_cosidas):
    """
    Dibuja el resultado del algoritmo de Cosito Topológico.
    Cada texto de potencia y su parcela cerrada tendrán el mismo color asignado.
    Las líneas que no formen ninguna parcela se pintarán en gris de fondo.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Definimos una paleta con muchos colores distintos bien diferenciados
    num_parcelas = len(parcelas_estructuradas)
    cmap = plt.get_cmap('tab20' if num_parcelas <= 20 else 'gist_nanderson')
    colores = [cmap(i / num_parcelas) for i in range(num_parcelas)]

    # 2. PINTAR EL FONDO NEUTRO: Dibujamos todas las líneas cosidas en gris flojo
    # Esto asegura que las calles y zonas públicas se vean, aunque no tengan texto
    for linea in lineas_totales_cosidas:
        print(f"Dibujando línea cosida: {linea}")
        x_coords, y_coords = linea.exterior.xy
        ax.plot(x_coords, y_coords, color="#cccccc", linewidth=1.0, zorder=1, alpha=0.5)

    # 3. PINTAR LAS PARCELAS DE COLORES (Según tu algoritmo de cosido)
    for idx, (id_txt, datos) in enumerate(parcelas_estructuradas.items()):
        color_actual = colores[idx]
        pos_texto = datos['posicion_texto']
        poligono = datos['poly_obj']
        kw = potencias[id_txt] if id_txt < len(potencias) else "?"

        # A. Si el algoritmo ha conseguido cerrar el polígono, lo rellenamos y perfilamos
        if poligono and not poligono.is_empty:
            x_coords, y_coords = poligono.exterior.xy
            # Pintamos el contorno de la parcela con su color único
            ax.plot(x_coords, y_coords, color=color_actual, linewidth=2.0, zorder=3)
            # Aplicamos un relleno transparente muy elegante para ver la propiedad
            ax.fill(x_coords, y_coords, color=color_actual, alpha=0.15, zorder=2)
        else:
            # Si la parcela falló en cerrarse por completo, pintamos al menos las líneas que tenía cerca
            for linde in datos.get('lindes', []):
                x_coords, y_coords = linde.xy
                ax.plot(x_coords, y_coords, color=color_actual, linewidth=1.5, linestyle="--", zorder=3)

        # B. Dibujamos el texto de potencia (kW) flotando en su sitio con el mismo color
        ax.annotate(
            f"P{id_txt}: {kw} kW", 
            xy=(pos_texto[0], pos_texto[1]), 
            textcoords="offset points", 
            xytext=(0, 5), 
            ha='center', 
            fontsize=8, 
            color=color_actual,
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_actual, lw=1, alpha=0.8),
            zorder=4
        )
        
        # Un punto indicador en el centro del texto
        ax.scatter(pos_texto[0], pos_texto[1], color=color_actual, s=30, edgecolors='black', zorder=5)

    ax.set_aspect('equal')
    plt.title("Verificación: Parcelas Cerradas y Segelladas por Cosito Topológico")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

def plot_parcelas(parcelas_estructuradas):
    """
    Dibuja cada texto de potencia y sus líneas asignadas con un color único
    para verificar el éxito del algoritmo de visibilidad.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Creamos una paleta con suficientes colores distintos para todas las parcelas
    num_parcelas = len(parcelas_estructuradas)
    # Usamos 'hsv' o 'tab20' mezclado para generar un abanico grande de colores
    cmap = plt.get_cmap('gist_ncar' if num_parcelas > 20 else 'tab20')
    colores_aleatorios = [cmap(i / num_parcelas) for i in range(num_parcelas)]

    # 2. Recorremos cada parcela estructurada por tu algoritmo
    for idx, (id_parcela, datos) in enumerate(parcelas_estructuradas.items()):
        color_actual = colores_aleatorios[idx]
        pos_texto = datos['posicion_texto']
        lindes = datos['lindes']

        # A. Pintamos las líneas de esta parcela con su color único
        for linde in lindes:
            x_coords, y_coords = linde.xy
            ax.plot(x_coords, y_coords, color=color_actual, linewidth=2.0, zorder=2)
        
        # Dibujamos un punto gordo donde está el ancla del texto
        ax.scatter(pos_texto[0], pos_texto[1], color=color_actual, s=30, edgecolors='black', zorder=4)

    ax.set_aspect('equal')
    plt.title("Verificación del Algoritmo: Líneas de Fachada asignadas por Visibilidad Directa")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

def place_CTs(potencias, positions, G):
    """
    Ubicación y asignación de CTs limitada a 800 kVA por máquina
    y penalizando el paso por calles estrechas a través del grafo G.
    """
    # 1. Extraemos las potencias y posiciones de las acometidas
    
    if not potencias:
        print("⚠️ No se encontraron potencias válidas.")
        return [], [], [], []

    # 2. Calcular cuántos CTs necesitamos como mínimo absoluto
    potencia_total_simultanea = sum(potencias) * 0.4 / 0.9
    limite_ct = 800.0  # kVA máximo por CT
    
    n_clusters = int(np.ceil(potencia_total_simultanea / limite_ct))
    n_clusters = max(1, n_clusters)
    print(f"⚡ Potencia simultánea total: {potencia_total_simultanea:.2f} kVA. Necesitamos mínimo {n_clusters} CT(s) de 800 kVA.")

    # 3. K-Means inicial para ubicar los centros teóricos aproximados en el espacio
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(positions, sample_weight=potencias)
    centros_teoricos = kmeans.cluster_centers_

    # REGLA DE INGENIERÍA: Mover el CT teórico al nodo del grafo de calles más cercano
    # (Un CT no puede flotar en mitad de una parcela privada)
    centros_transformacion_ideales = []
    for centro in centros_teoricos:
        nodo_calle_mas_cercano = min(G.nodes, key=lambda n: ((n[0]-centro[0])**2 + (n[1]-centro[1])**2))
        centros_transformacion_ideales.append(nodo_calle_mas_cercano)
        
    centros_transformacion_ideales = np.array(centros_transformacion_ideales)

    # ========================================================
    # 4. OPTIMIZACIÓN CON PuLP (Teniendo en cuenta pesos del Grafo)
    # ========================================================
    num_parcelas = len(positions)
    prob = pulp.LpProblem("Reparto_Capacitado_Con_Penalizacion_Calles", pulp.LpMinimize)
    X = pulp.LpVariable.dicts("Asignacion", ((i, j) for i in range(num_parcelas) for j in range(n_clusters)), cat='Binary')
    
    # Calcular la matriz de distancias PENALIZADAS utilizando el Grafo Vial
    distancias_penalizadas = np.zeros((num_parcelas, n_clusters))
    
    for i in range(num_parcelas):
        # Encontramos el nodo del grafo más cercano a la acometida de la parcela 'i'
        p_coords = (positions[i][0], positions[i][1])
        nodo_origen_calle = min(G.nodes, key=lambda n: ((n[0]-p_coords[0])**2 + (n[1]-p_coords[1])**2))
        
        for j in range(n_clusters):
            ct_coords = (centros_transformacion_ideales[j][0], centros_transformacion_ideales[j][1])
            
            try:
                # Calculamos la distancia por calles usando el atributo 'weight' penalizado (Paso 1)
                # Si pasa por calle estrecha, este valor será mucho mayor que la distancia geométrica real
                dist_vial_penalizada = nx.shortest_path_length(G, source=nodo_origen_calle, target=ct_coords, weight='weight')
                
                # Sumamos el pequeño tramo de acometida desde la parcela hasta la calle
                dist_acometida = ((p_coords[0]-nodo_origen_calle[0])**2 + (p_coords[1]-nodo_origen_calle[1])**2)**0.5
                
                distancias_penalizadas[i, j] = dist_vial_penalizada + dist_acometida
                
            except nx.NetworkXNoPath:
                # Si el grafo está desconectado por error de dibujo, aplicamos una penalización masiva
                distancias_penalizadas[i, j] = 99999.0

    # FUNCIÓN OBJETIVO: Minimizar el coste del cableado considerando las calles penalizadas
    prob += pulp.lpSum(X[i, j] * distancias_penalizadas[i, j] * potencias[i] for i in range(num_parcelas) for j in range(n_clusters))

    # RESTRICCIÓN 1: Cada parcela se asigna a un único CT
    for i in range(num_parcelas):
        prob += pulp.lpSum(X[i, j] for j in range(n_clusters)) == 1

    # RESTRICCIÓN 2: Ningún CT puede superar los 800 kVA simultáneos
    for j in range(n_clusters):
        prob += pulp.lpSum(X[i, j] * (potencias[i] * 0.4 / 0.9) for i in range(num_parcelas)) <= limite_ct

    # Resolver el modelo matemático en silencio
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 5. Extraer las nuevas etiquetas corregidas por el optimizador
    labels = np.zeros(num_parcelas, dtype=int)
    for i in range(num_parcelas):
        for j in range(n_clusters):
            if pulp.value(X[i, j]) == 1:
                labels[i] = j

    # Imprimir balance de carga real para verificar resultados
    print("\n📊 Balance de carga final (Optimizado por capacidad y tipo de calle):")
    for j in range(n_clusters):
        cargas_asignadas = [potencias[i] for i in range(num_parcelas) if labels[i] == j]
        pot_simultanea_ct = sum(cargas_asignadas) * 0.4 / 0.9
        print(f"   - CT {j+1} (Ubicado en calle): {len(cargas_asignadas)} parcelas | Potencia: {pot_simultanea_ct:.2f} kVA / {limite_ct} kVA")

    return potencias, positions, centros_transformacion_ideales, labels

def plot_graph(potencias, positions, parcelas, centros_transformacion_ideales, labels):
    fig, ax = plt.subplots(figsize=(10, 10))

    cmap = plt.get_cmap('tab10')

    for position, potencia, label in zip(positions, potencias, labels):
        ax.annotate(f"{potencia} kW", (position[0], position[1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=cmap(label))
        ax.scatter(position[0], position[1], marker='x', color=cmap(label))

    # Dibujamos las líneas una a una de forma independiente en color rojo
    for parcela in parcelas:
        x_coords, y_coords = parcela.xy
        ax.plot(x_coords, y_coords, color="red", linewidth=1.5)

    # Dibujamos los centros de masa de potencia
    for idx,centro in enumerate(centros_transformacion_ideales):
       potencia_total = sum([potencias[i] for i in range(len(potencias)) if labels[i] == idx])
       ax.scatter(centro[0], centro[1], marker='o', color=cmap(idx))
       ax.annotate(f"{potencia_total*0.4/0.9} kW", (centro[0], centro[1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=cmap(idx))

    ax.set_aspect('equal')
    plt.title("Líneas en bruto desde el Excel (Sin agrupar)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()