import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize
from sklearn.cluster import KMeans

import pulp

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

def place_CTs(potencias, positions):
    potencias = np.array(potencias)
    positions = np.array(positions)
    n_casas = len(positions)
    
    # 1. CÁLCULO DINÁMICO DE CLUSTERS
    pot_CTs = [250, 400, 630, 800]  # Potencias de los CTs en kW
    max_cap_ct = pot_CTs[-1]        # Límite estricto de 800 kW
    
    potencia_total_necesaria = sum(potencias) * 0.4 / 0.9
    n_clusters = int(np.ceil(potencia_total_necesaria / max_cap_ct))
    print(f"Número de clusters a generar: {n_clusters}")

    # 2. INICIALIZACIÓN
    np.random.seed(42)
    # Inicialización inteligente: elegimos las casas con más potencia como semillas iniciales
    indices_iniciales = np.argsort(-potencias)[:n_clusters]
    centros = positions[indices_iniciales].copy()
    
    # Ordenamos las casas de mayor a menor potencia para garantizar que las cargas
    # más críticas (que penalizan más el coste del cable) elijan primero su centro óptimo
    orden_casas = np.argsort(-potencias)
    
    max_iter = 50
    labels = np.zeros(n_casas, dtype=int)
    
    # 3. BUCLE ITERATIVO (Minimizando Sumatoria de Distancia * Potencia)
    for iteracion in range(max_iter):
        labels_antiguos = labels.copy()
        
        cargas_simultaneas_cts = np.zeros(n_clusters)
        cargas_nominales_cts = np.zeros(n_clusters)
        
        for i in orden_casas:
            casa_coord = positions[i]
            casa_pot = potencias[i]
            casa_pot_simultanea = casa_pot * 0.4 / 0.9
            
            # --- AQUÍ ESTÁ EL CAMBIO CLAVE ---
            # Calculamos la distancia geométrica a cada centro
            distancias = np.linalg.norm(centros - casa_coord, axis=1)
            
            # El coste real de conectar esta casa a cada centro es Distancia * Potencia
            # Al minimizar este coste, el algoritmo prioriza asignar casas grandes a centros muy cercanos
            costes_linea = distancias * casa_pot
            
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

    return centros_de_masas_reales, labels, potencia_total_grupo


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