import ezdxf
import matplotlib.pyplot as plt 
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import polygonize

def load_from_dxf(path):
    # 1. Leer el archivo CAD directamente
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    
    lineas = []

    # 2. Leer TODAS las polilíneas de la capa de parcelas
    # (Funciona tanto para LWPOLYLINE como para POLYLINE tradicionales)
    for entity in msp.query('LWPOLYLINE[layer=="0-REPARCELACION"]'):

        ruta_objeto = ezdxf.path.make_path(entity)
        # El truco: 'distance=0.05' le dice que aproxime las curvas con tramos rectos cada 5 cm
        puntos = list(ruta_objeto.flattening(distance=0.05))
        # 1. Convertimos la entidad de AutoCAD en un objeto Path genérico
        
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
    return parcelas_poligonos

def plot_graph(parcelas_poligonos):

# 4. Pintar el resultado
    fig, ax = plt.subplots(figsize=(10, 10))
    for pol in parcelas_poligonos:
        x, y = pol.exterior.xy
        ax.plot(x, y, color="blue", linewidth=1)
        ax.fill(x, y, color="lightblue", alpha=0.3)
        
    ax.set_aspect('equal')
    plt.title("Plano Completo de Parcelas (Procesado por DXF)")
    plt.show()