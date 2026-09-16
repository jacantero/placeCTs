from functions import *

path = "16925-CMYPA-PLA-11.1-RedBajaTensionDisHojas-S2-00.xlsx"
#Nos interesan las columnas de Capa, Posición x, Posición Y, Contenido, Inicio X, Inicio Y, Fin X y Fin Y. Hay que _explode las polilineas
df = pd.read_excel(path)

capas_validas = ["0-REPARCELACION", "2-CARGAS-BT"]
selected_rows = df[df['Capa'].isin(capas_validas)]

#print(selected_rows)
plot_graph(selected_rows)