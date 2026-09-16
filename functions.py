import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize

def plot_graph(pandas_df):

    fig, ax = plt.subplots(figsize=(10, 10))

    # Dibujamos las líneas una a una de forma independiente en color rojo
    for index, row in pandas_df.iterrows():
        if not pd.isna(row['Inicial X']):
            x_coords = [row['Inicial X'], row['Fin X']]
            y_coords = [row['Inicial Y'], row['Fin Y']]
         # CASO B: Es un Arco (No tiene Inicial X, pero tiene Centro X y Radio)
        elif not pd.isna(row['Centro X']):
            cx = row['Centro X']
            cy = row['Centro Y']
            ang_inicio = row['Ángulo inicial']
            ang_total = row['Ángulo total']
            long_arc = row['Longitud']
            
            # Generamos 20 puntos intermedios para que la curva se vea suave
            angulos_rad = np.radians(ang_total)
            r = long_arc / angulos_rad # Radio del arco

            angulos_rad = np.linspace(np.radians(ang_inicio), np.radians(ang_inicio + ang_total), 20)
            
            # Trigonometría básica: X = Cx + R*cos(θ), Y = Cy + R*sin(θ)
            x_coords = cx + r * np.cos(angulos_rad)
            y_coords = cy + r * np.sin(angulos_rad)
            print(x_coords, y_coords)

        ax.plot(x_coords, y_coords, color="red", linewidth=1.5)

    ax.set_aspect('equal')
    plt.title("Líneas en bruto desde el Excel (Sin agrupar)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()