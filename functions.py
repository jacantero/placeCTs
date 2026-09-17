import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize
from sklearn.cluster import KMeans

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

def place_CTs(potencias, positions):
    # Implementamos K-Means clásico adaptado con pesos en la muestra
    # n_clusters será el número de CTs que quieres colocar
    pot_CTs = [250, 400, 630, 800]  # Potencias de los CTs en kW
    n_clusters = int(np.ceil((sum(potencias)*0.4/0.9)/pot_CTs[-1]))  # Ajusta este cálculo según tus necesidades
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(positions, sample_weight=potencias)
    
    # Estos son los puntos óptimos teóricos (los centros de masa de potencia)
    centros_transformacion_ideales = kmeans.cluster_centers_
        
    return centros_transformacion_ideales, labels

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