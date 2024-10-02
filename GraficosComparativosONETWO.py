import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Definición de los teléfonos y sus archivos asociados
telefonos = ["Pixel 6 Pro", "Pixel 3a", "Pixel 4a", "M2007J3SY"]
nombres_telefonos = [t.replace(" ", "") for t in telefonos]
# Definición de las MACs para One-Sided y Two-Sided (ajustar según sea necesario)
macs_one_sided = [
    'c4:41:1e:fa:07:db#2_value',  # L5DFS
    'c4:41:1e:fa:07:da#2_value',  # L5indoor
    'c4:41:1e:fa:07:d9#2_value',  # L2_4
    'cc:f4:11:47:ef:eb#2_value',  # G2_4
    'cc:f4:11:47:ef:e7#2_value'   # G5
]
macs_two_sided = [
    'c4:41:1e:fa:07:db#1_value',  # L5DFS
    'c4:41:1e:fa:07:da#1_value',  # L5indoor
    'c4:41:1e:fa:07:d9#1_value',  # L2_4
    'cc:f4:11:47:ef:eb#1_value',  # G2_4
    'cc:f4:11:47:ef:e7#1_value'   # G5
]
# Correspondencia MAC a Banda para One-Sided y Two-Sided
mac_to_band_one_sided = {
    'c4:41:1e:fa:07:db#2_value': 'L5DFS',
    'c4:41:1e:fa:07:da#2_value': 'L5indoor',
    'c4:41:1e:fa:07:d9#2_value': 'L2.4',
    'cc:f4:11:47:ef:eb#2_value': 'G2.4',
    'cc:f4:11:47:ef:e7#2_value': 'G5'
}
mac_to_band_two_sided = {
    'c4:41:1e:fa:07:db#1_value': 'L5DFS',
    'c4:41:1e:fa:07:da#1_value': 'L5indoor',
    'c4:41:1e:fa:07:d9#1_value': 'L2.4',
    'cc:f4:11:47:ef:eb#1_value': 'G2.4',
    'cc:f4:11:47:ef:e7#1_value': 'G5'
}
def cargar_dataframes(path_base, nombres_telefonos, macs, mac_to_band, tipo_metodo):
    dataframes = {}
    for nombre_telefono in nombres_telefonos:
        # Cargar el CSV correspondiente al teléfono
        df = pd.read_csv(f'{path_base}{nombre_telefono}.csv')
        # Crear un DataFrame para cada MAC, extrayendo las columnas relevantes
        for mac_deseada in macs:
            # Extraer las columnas relevantes
            columnas_interes = ['y', 'model', 'angle', 'sampleNumber', mac_deseada, mac_deseada.replace('_value', '_error'), mac_deseada.replace('_value', '_samples_averaged')]
            df_filtrado = df[columnas_interes]
            # Guardar el DataFrame con un nombre adecuado
            band_name = mac_to_band[mac_deseada]
            dataframe_name = f"{nombre_telefono}_{band_name}_{tipo_metodo}"
            dataframes[dataframe_name] = df_filtrado
    return dataframes
# Crear DataFrames para One-Sided
dataframes_one_sided = cargar_dataframes(
    'C:/Users/jaume/Documents/VISUAL CODES/TFG/muestras3/muestras3_', 
    nombres_telefonos, 
    macs_one_sided, 
    mac_to_band_one_sided, 
    'One-Sided'
)
# Crear DataFrames para Two-Sided
dataframes_two_sided = cargar_dataframes(
    'C:/Users/jaume/Documents/VISUAL CODES/TFG/muestrasTwoSided/muestras2TwoSided_', 
    nombres_telefonos, 
    macs_two_sided, 
    mac_to_band_two_sided, 
    'Two-Sided'
)
# Combinar ambos diccionarios de DataFrames
dataframes = {**dataframes_one_sided, **dataframes_two_sided}
# Filtrar los DataFrames específicos que necesitas
telefonos_interes = ["Pixel3a"]
bandas_interes = ['G5', 'G2.4', 'L5indoor']
metodos = ['One-Sided', 'Two-Sided']
# Seleccionar solo los DataFrames de interés
df_interes = {
    f"{telefono}_{banda}_{metodo}": dataframes[f"{telefono}_{banda}_{metodo}"]
    for telefono in telefonos_interes
    for banda in bandas_interes
    for metodo in metodos
}
# Definir los BS de interés
BSlist = list(range(2, 12))
############ --------------------------- Funcion RMSE -----------------------#####
def calcular_rmse(df, mac_deseada):
    # Filtrar solo las filas correspondientes a la distancia 5m
    df_5m = df[df['y'] == 5]
    rmse_por_BS = {}
    for BS in BSlist:
        # Filtrar las filas que cumplen con las condiciones del ángulo y muestras válidas
        filtro = (df_5m['sampleNumber'] >= 0) & (df_5m['sampleNumber'] <= 199) & (df_5m['angle'] == BS)
        df_filtrado = df_5m[filtro].copy()  # Crear una copia del DataFrame filtrado
        # Reemplazar valores de -100 por NaN en la columna del valor de MAC
        df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
        # Filtrado para eliminar outliers fuera de ±3 sigma
        df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
        # Calcular la mediana y el RMSE
        mediana = df_filtrado[mac_deseada].median()
        df_filtrado[mac_deseada] -= mediana
        df_filtrado[mac_deseada] **= 2
        rmse = (df_filtrado[mac_deseada].sum() / df_filtrado[mac_deseada].count()) ** 0.5
        rmse_por_BS[BS] = rmse
    return rmse_por_BS
# Calcular el RMSE para cada uno de los DataFrames seleccionados
rmse_resultados = {}
for key, df in df_interes.items():
    # Determinar la MAC relevante para la banda y método
    if 'One-Sided' in key:
        mac_deseada = [mac for mac, banda in mac_to_band_one_sided.items() if banda in key][0]
    else:
        mac_deseada = [mac for mac, banda in mac_to_band_two_sided.items() if banda in key][0]
    # Calcular RMSE
    rmse_resultados[key] = calcular_rmse(df, mac_deseada)
# Colores para One-Sided y Two-Sided
colores = {
    'One-Sided': 'blue',
    'Two-Sided': 'red'
}
# Estilos de línea para cada banda
estilos_linea = {
    'G5': '-',
    'G2.4': '--',
    'L5indoor': ':'
}
# Graficar los resultados del RMSE para los DataFrames seleccionados
plt.figure(figsize=(12, 8))
for key, rmse_data in rmse_resultados.items():
    # Determinar el método (One-Sided o Two-Sided) y la banda
    metodo = 'One-Sided' if 'One-Sided' in key else 'Two-Sided'
    banda = [banda for banda in bandas_interes if banda in key][0]
    # Seleccionar color y estilo según el método y la banda
    color = colores[metodo]
    estilo = estilos_linea[banda]
    # Graficar la curva con el color y estilo adecuados
    plt.plot(
        list(rmse_data.keys()), 
        list(rmse_data.values()), 
        marker='o', 
        label=f"{banda} ({metodo})", 
        color=color, 
        linestyle=estilo
    )
plt.title('RMSE para Pixel 3a - Bandas G5, G2.4 y L5Indoor (One-Sided y Two-Sided)')
plt.xlabel('Burst Size')
plt.ylabel('RMSE')
plt.grid(True)
plt.legend(title='Combinaciones')
plt.show()

############ --------------------------- Funcion Ranging Error -----------------------#####
def calcular_ranging_error(df, mac_deseada):
    # Filtrar solo las filas correspondientes a la distancia 5m
    df_5m = df[df['y'] == 5]
    ranging_error_por_BS = {}
    for BS in BSlist:
        # Filtrar las filas que cumplen con las condiciones del ángulo y muestras válidas
        filtro = (df_5m['sampleNumber'] >= 0) & (df_5m['sampleNumber'] <= 199) & (df_5m['angle'] == BS)
        df_filtrado = df_5m[filtro].copy()  # Crear una copia del DataFrame filtrado
        # Reemplazar valores de -100 por NaN en la columna del valor de MAC
        df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
        # Filtrado para eliminar outliers fuera de ±3 sigma
        df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
        # Calcular la mediana y el Ranging Error
        mediana = df_filtrado[mac_deseada].median()
        media = df_filtrado[mac_deseada].mean()
        ranging_error = media - mediana
        ranging_error_por_BS[BS] = ranging_error
    return ranging_error_por_BS
# Calcular el Ranging Error para cada uno de los DataFrames seleccionados
ranging_error_resultados = {}
for key, df in df_interes.items():
    # Determinar la MAC relevante para la banda y método
    if 'One-Sided' in key:
        mac_deseada = [mac for mac, banda in mac_to_band_one_sided.items() if banda in key][0]
    else:
        mac_deseada = [mac for mac, banda in mac_to_band_two_sided.items() if banda in key][0]
    # Calcular Ranging Error
    ranging_error_resultados[key] = calcular_ranging_error(df, mac_deseada)
# Graficar los resultados del Ranging Error para los DataFrames seleccionados
plt.figure(figsize=(12, 8))
for key, ranging_error_data in ranging_error_resultados.items():
    # Determinar el método (One-Sided o Two-Sided) y la banda
    metodo = 'One-Sided' if 'One-Sided' in key else 'Two-Sided'
    banda = [banda for banda in bandas_interes if banda in key][0]
    # Seleccionar color y estilo según el método y la banda
    color = colores[metodo]
    estilo = estilos_linea[banda]
    # Graficar la curva con el color y estilo adecuados
    plt.plot(
        list(ranging_error_data.keys()), 
        list(ranging_error_data.values()), 
        marker='o', 
        label=f"{banda} ({metodo})", 
        color=color, 
        linestyle=estilo
    )
plt.title('Ranging Error para Pixel 3a - Bandas G5, G2.4 y L5Indoor (One-Sided y Two-Sided)')
plt.xlabel('Burst Size')
plt.ylabel('Ranging Error')
plt.grid(True)
#pintamos el 0 con una linea mas gruesa
plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
plt.legend(title='Combinaciones')
plt.show()
def calcular_pdf_ranging_error(df, mac_deseada):
    # Filtrar solo las filas correspondientes a la distancia 5m
    df_5m = df[df['y'] == 5]
    ranging_errors = []
    for BS in BSlist:
        # Filtrar las filas que cumplen con las condiciones del ángulo y muestras válidas
        filtro = (df_5m['sampleNumber'] >= 0) & (df_5m['sampleNumber'] <= 199) & (df_5m['angle'] == 8)
        df_filtrado = df_5m[filtro].copy()  # Crear una copia del DataFrame filtrado
        # Reemplazar valores de -100 por NaN en la columna del valor de MAC
        df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
        # Filtrado para eliminar outliers fuera de ±3 sigma
        df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
        # Calcular la mediana y el Ranging Error
        mediana = df_filtrado[mac_deseada].median()
        media = df_filtrado[mac_deseada].mean()
        ranging_error = media - mediana
        ranging_errors.extend(df_filtrado[mac_deseada] - mediana)
    return ranging_errors
# Calcular los valores de Ranging Error para cada combinación
pdf_resultados = {}
for key, df in df_interes.items():
    # Determinar la MAC relevante para la banda y método
    if 'One-Sided' in key:
        mac_deseada = [mac for mac, banda in mac_to_band_one_sided.items() if banda in key][0]
    else:
        mac_deseada = [mac for mac, banda in mac_to_band_two_sided.items() if banda in key][0]
    # Calcular valores de Ranging Error
    pdf_resultados[key] = calcular_pdf_ranging_error(df, mac_deseada)

# Graficar las PDF para el método One-Sided
plt.figure(figsize=(14, 6))
for key, errors in pdf_resultados.items():
    if 'One-Sided' in key:
        banda = [banda for banda in bandas_interes if banda in key][0]
        color = colores['One-Sided']
        estilo = estilos_linea[banda]
        sns.kdeplot(errors, label=f"{banda} (One-Sided)", color=color, linestyle=estilo, fill=True)

plt.title('PDF del Ranging Error para Pixel 3a - One-Sided - BS 8')
plt.xlabel('Ranging Error [m]')
plt.ylabel('Densidad')
plt.grid(True)
plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
plt.legend(title='Bandas')
plt.show()
# Graficar las PDF para el método Two-Sided
plt.figure(figsize=(14, 6))
for key, errors in pdf_resultados.items():
    if 'Two-Sided' in key:
        banda = [banda for banda in bandas_interes if banda in key][0]
        color = colores['Two-Sided']
        estilo = estilos_linea[banda]
        sns.kdeplot(errors, label=f"{banda} (Two-Sided)", color=color, linestyle=estilo, fill=True)

plt.title('PDF del Ranging Error para Pixel 3a - Two-Sided - BS 8')
plt.xlabel('Ranging Error [m]')
plt.ylabel('Densidad')
plt.grid(True)
plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
plt.legend(title='Bandas')
plt.show()
