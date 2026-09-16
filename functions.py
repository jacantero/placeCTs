import matplotlib.pyplot as plt 
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import polygonize

def load_from_df(path):
    # 1. Leer el archivo CAD directamente
    doc = ezdxf.readfile(ruta_dxf)
    msp = doc.modelspace()
    
    lineas = []
    
    # 2. Leer TODAS las polilíneas de la capa de parcelas
    # (Funciona tanto para LWPOLYLINE como para POLYLINE tradicionales)
    for entity in msp.query('LWPOLYLINE[layer=="0-REPARCELACION"]'):
        # El truco: 'distance=0.05' le dice que aproxime las curvas con tramos rectos cada 5 cm
        puntos = list(entity.flattening(distance=0.05))
        if len(puntos) > 1:
            # Creamos la geometría continua de la parcela completa
            # Sacamos tramos de dos puntos para alimentar el polygonize
            for i in range(len(puntos) - 1):
                p1 = (round(puntos[i][0], 2), round(puntos[i][1], 2))
                p2 = (round(puntos[i+1][0], 2), round(puntos[i+1][1], 2))
                lineas.append(LineString([p1, p2]))
                
    # 3. Cerrar los perímetros
    parcelas_poligonos = list(polygonize(lineas))
    print(f"Polígonos totales generados con éxito: {len(parcelas_poligonos)}")

def plot_graph(pandas_df):

    fig, ax = plt.subplots(figsize=(10, 10))

    # Dibujamos las líneas una a una de forma independiente en color rojo
    for index, row in pandas_df.iterrows():
        if not pd.isna(row['Inicial X']):
            x_coords = [row['Inicial X'], row['Fin X']]
            y_coords = [row['Inicial Y'], row['Fin Y']]
            ax.plot(x_coords, y_coords, color="red", linewidth=1.5)

    ax.set_aspect('equal')
    plt.title("Líneas en bruto desde el Excel (Sin agrupar)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()