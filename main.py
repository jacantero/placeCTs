from functions import *

path = "16925-CMYPA-PLA-11.1-RedBajaTensionDisHojas-S2-00.xlsx"
#Nos interesan las columnas de Capa, Posición x, Posición Y, Contenido, Inicio X, Inicio Y, Fin X y Fin Y. Hay que _explode las polilineas
df = pd.read_excel(path)

capas_validas = ["0-REPARCELACION", "2-CARGAS-BT"]
selected_rows = df[df['Capa'].isin(capas_validas)]

#print(selected_rows)

potencias, positions = get_power(selected_rows)
parcelas = get_parcelas(selected_rows)
poligonos_cerrados = dict_parcelas(selected_rows, positions)
parcelas_estructuradas = asignar_potencia_a_parcelas_cosidas(poligonos_cerrados, positions, potencias)

plot_graph_cosido(parcelas_estructuradas, potencias, poligonos_cerrados)

centros_cts, labels = place_CTs(potencias, positions)
