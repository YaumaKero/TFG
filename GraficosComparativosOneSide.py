import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#####------------------------CREAR DATAFRAMES DE CADA TELEFONO-MAC------------------------#####

# Definición de los teléfonos y sus archivos asociados
telefonos = ["Pixel 6 Pro", "Pixel 3a", "Pixel 4a", "M2007J3SY"]
nombres_telefonos = [t.replace(" ", "") for t in telefonos]
# Definición de las MACs (que son los nombres de las columnas)
macs = [
    'c4:41:1e:fa:07:db#2_value',  # L5DFS
    'c4:41:1e:fa:07:da#2_value',  # L5indoor
    'c4:41:1e:fa:07:d9#2_value',  # L2_4
    'cc:f4:11:47:ef:eb#2_value',  # G2_4
    'cc:f4:11:47:ef:e7#2_value'   # G5
]
# Correspondencia MAC a Banda
mac_to_band = {
    'c4:41:1e:fa:07:db#2_value': 'L5DFS',
    'c4:41:1e:fa:07:da#2_value': 'L5indoor',
    'c4:41:1e:fa:07:d9#2_value': 'L2.4',
    'cc:f4:11:47:ef:eb#2_value': 'G2.4',
    'cc:f4:11:47:ef:e7#2_value': 'G5'
}
# Diccionario para almacenar los DataFrames
dataframes = {}
# Crear DataFrames para cada combinación de teléfono y MAC
for nombre_telefono in nombres_telefonos:
    # Cargar el CSV correspondiente al teléfono
    df = pd.read_csv(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/muestras3/muestras3_{nombre_telefono}.csv')
    # Crear un DataFrame para cada MAC, extrayendo las columnas relevantes
    for mac_deseada in macs:
        # Extraer las columnas relevantes (y, model, angle, sampleNumber y columnas relacionadas con la MAC deseada)
        columnas_interes = ['y', 'model', 'angle', 'sampleNumber', mac_deseada, mac_deseada.replace('_value', '_error'), mac_deseada.replace('_value', '_samples_averaged')]
        df_filtrado = df[columnas_interes]
        # Guardar el DataFrame con un nombre adecuado
        band_name = mac_to_band[mac_deseada]
        dataframe_name = f"{nombre_telefono}_{band_name}"
        dataframes[dataframe_name] = df_filtrado

#####------------------------ANALISIS DE MUESTRAS VALIDAS------------------------#####

# Definir la lista de valores 'y' y 'BS' que quieres analizar
ys = [1,5,10,15,20,25]  # Ejemplo, cambia estos valores según lo que necesites
BSlist = list(range(2, 30))
# Diccionario para almacenar las muestras válidas
muestras_validas = {}
# Iterar sobre cada combinación de 'y' y 'BS'
for y in ys:
    for BS in BSlist:
        for nombre_telefono in nombres_telefonos:
            for mac_deseada in macs:
                # Selecciona el DataFrame filtrado
                df = dataframes[f'{nombre_telefono}_{mac_to_band[mac_deseada]}']
                # Aplica los filtros para seleccionar las muestras válidas
                filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == y) & (df['angle'] == BS)
                df_filtrado = df[filtro].copy()
                # Filtro adicional dependiendo del nombre del teléfono
                if nombre_telefono == "Pixel6Pro":
                    filtro_BS = (df_filtrado[mac_deseada.replace('_value', '_samples_averaged')] == BS - 1)
                else:
                    filtro_BS = (df_filtrado[mac_deseada.replace('_value', '_samples_averaged')] == BS)    
                df_filtrado = df_filtrado[filtro_BS]
                # Contar el número de muestras válidas
                num_muestras_validas = df_filtrado[mac_deseada].count()
                # Aplicar divisiones basadas en el número de muestras
                if num_muestras_validas > 200:
                    num_muestras_validas /= 2
                if num_muestras_validas > 400:
                    num_muestras_validas /= 4
                num_muestras_validas = round(num_muestras_validas)
                porcentaje_muestras_validas = (num_muestras_validas / 200) * 100
                # Acumular el total para cada BS
                key = f'{nombre_telefono}_{mac_to_band[mac_deseada]}_BS{BS}'
                if key not in muestras_validas:
                    muestras_validas[key] = []
                muestras_validas[key].append(porcentaje_muestras_validas)
# Calcular la media del porcentaje de muestras válidas para cada combinación
media_muestras_validas = {key: np.mean(valores) for key, valores in muestras_validas.items()}
# Convertir el diccionario en un DataFrame para facilitar su análisis
df_muestras_validas_media = pd.DataFrame(list(media_muestras_validas.items()), columns=['Combinación', 'Media Porcentaje Muestras Válidas'])
# Separar las partes de la columna 'Combinación'
df_muestras_validas_media['Telefono'] = df_muestras_validas_media['Combinación'].apply(lambda x: x.split('_')[0])
df_muestras_validas_media['MAC'] = df_muestras_validas_media['Combinación'].apply(lambda x: x.split('_')[1])
df_muestras_validas_media['Angulo'] = df_muestras_validas_media['Combinación'].apply(lambda x: x.split('BS')[1])
# Crear una nueva columna que combine el Telefono y la MAC para el pivot
df_muestras_validas_media['Telefono_MAC'] = df_muestras_validas_media['Telefono'] + '_' + df_muestras_validas_media['MAC']
# Convertir 'Angulo' a numérico para asegurar el orden correcto en la tabla pivote
df_muestras_validas_media['Angulo'] = pd.to_numeric(df_muestras_validas_media['Angulo'])
# Realizar el pivot para que 'Angulo' sea el índice y 'Telefono_MAC' las columnas
df_muestras_validas_media_pivot = df_muestras_validas_media.pivot(index='Angulo', columns='Telefono_MAC', values='Media Porcentaje Muestras Válidas')
#excluims del dataframe las columnas: Pixel6Pro_L2, Pixel6Pro_G2, Pixel6Pro_L5indoor, M2007J3SY_L5DFS
df_muestras_validas_media_pivot = df_muestras_validas_media_pivot.drop(columns=['Pixel6Pro_L2.4', 'Pixel6Pro_G2.4', 'Pixel6Pro_L5indoor', 'M2007J3SY_L5DFS'])
#redondeamos los valores del dataframe a 2 decimales
df_muestras_validas_media_pivot = df_muestras_validas_media_pivot.round(2)
# Imprimir el DataFrame pivoteado
print(df_muestras_validas_media_pivot)
#guardamos el dataframe en un archivo csv
df_muestras_validas_media_pivot.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/muestras_validas_media_pivot.csv')


#####------------------------ANALISIS DE BS MEDIO------------------------#####
# Diccionario para almacenar los resultados de BS medio logrado
BSlogrado = {}
# Iterar sobre cada combinación de teléfono y banda (MAC)
for nombre_telefono in nombres_telefonos:
    for mac_deseada in macs:
        # Nombre de la banda y DataFrame correspondiente
        band_name = mac_to_band[mac_deseada]
        df = dataframes[f"{nombre_telefono}_{band_name}"]       
        # Inicializar un diccionario para almacenar la media de BS por ángulo (BS)
        medias_por_angulo = {}
        for BS in BSlist:
            # Filtrar las filas que cumplen con las condiciones
            filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['angle'] == BS) & (df[mac_deseada.replace('_value', '_samples_averaged')] != 0)
            df_filtrado = df[filtro].copy()  # Crear una copia del DataFrame filtrado          
            # Reemplazar valores de -100 por NaN en la columna del valor de MAC
            df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)         
            # Calcular la media de la columna 'samples_averaged'
            media_angulo = df_filtrado[mac_deseada.replace('_value', '_samples_averaged')].mean()
            # Almacenar la media por ángulo
            medias_por_angulo[BS] = media_angulo 
        # Almacenar las medias por ángulo en el diccionario BSlogrado
        clave = f"{nombre_telefono}_{band_name}"
        BSlogrado[clave] = medias_por_angulo
# Convertir el diccionario BSlogrado a un DataFrame, con los ángulos como índice
df_BSlogrado = pd.DataFrame(BSlogrado)
df_BSlogrado.index.name = 'BS'
# Redondear los valores a 2 decimales
df_BSlogrado = df_BSlogrado.round(2)
# Imprimir el DataFrame resultante
print(f'\nMedia BS Logrado por Ángulo para cada Teléfono y MAC')
print(df_BSlogrado)
#guardamos el dataframe en un archivo csv
df_BSlogrado.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/BSlogrado.csv')

###----------------------------------------------GRAFICAS DE RMSE MEDIO------------------------------------------------------------###
# Inicializamos el diccionario para almacenar los resultados del RMSE por cada teléfono y la banda L2.4
rmse_resultados_L2_4 = {telefono: {} for telefono in nombres_telefonos}
# Iterar sobre cada combinación de teléfono para la banda L2.4
for nombre_telefono in nombres_telefonos:
    mac_deseada = 'c4:41:1e:fa:07:d9#2_value'  # MAC para L2.4 en One-Sided
    band_name = 'L2.4'
    dataframe_name = f"{nombre_telefono}_{band_name}"
    # Obtener el DataFrame correspondiente
    df = dataframes[dataframe_name]
    # Filtrar solo las filas correspondientes a la distancia 5m
    df_5m = df[df['y'] == 5]
    # Inicializar diccionarios para almacenar RMSE por BS
    rmse_por_BS = {}
    # Iterar sobre cada ángulo (BS)
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
        # Almacenar el RMSE calculado por ángulo
        rmse_por_BS[BS] = rmse
    # Guardar los resultados del RMSE para la combinación de teléfono y banda L2.4
    rmse_resultados_L2_4[nombre_telefono] = rmse_por_BS
# Graficar los resultados del RMSE para la banda L2.4 para cada teléfono
for nombre_telefono in nombres_telefonos:
    plt.figure(figsize=(12, 8))
    rmse_data = rmse_resultados_L2_4[nombre_telefono]
    plt.plot(
        list(rmse_data.keys()), 
        list(rmse_data.values()), 
        marker='o', 
        label=f'L2.4', 
        color='green'  # Asignar un color verde para L2.4
    )
    plt.title(f'RMSE - {nombre_telefono}')
    plt.xlabel('Burst Size')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend(title='Bandas')
    plt.show()

#####------------------------ANALISIS DEL RANGING ERROR 4 GRAFICAS ------------------------#####
# Inicializar diccionarios para almacenar los resultados de Ranging Error y offset
ranging_error_resultados = {telefono: {} for telefono in nombres_telefonos}
offset_por_telefono_banda = {telefono: {} for telefono in nombres_telefonos}
# Combinaciones a excluir
exclusiones = ['Pixel6Pro_L2.4', 'Pixel6Pro_G2.4', 'Pixel6Pro_L5indoor', 'M2007J3SY_L5DFS']
# Colores específicos para cada banda
colores_bandas = {
    'L5DFS': '#1f77b4',      # Azul
    'L5indoor': '#ff7f0e',   # Naranja
    'L2.4': '#2ca02c',       # Verde (no se mostrará)
    'G2.4': '#d62728',       # Rojo
    'G5': '#9467bd'          # Púrpura
}
# Iterar sobre cada combinación de teléfono y banda (MAC)
for nombre_telefono in nombres_telefonos:
    for mac_deseada in macs:
        # Nombre de la banda y DataFrame correspondiente
        band_name = mac_to_band[mac_deseada]
        dataframe_name = f"{nombre_telefono}_{band_name}"
        # Excluir combinaciones no deseadas y la banda L2.4
        if dataframe_name in exclusiones or band_name != 'L2.4':
            continue
        # Obtener el DataFrame correspondiente
        df = dataframes[dataframe_name]    
        # Inicializar diccionarios para almacenar los resultados por BS
        ranging_error_por_BS = {}
        offset_por_BS = {}
        # Iterar sobre cada ángulo (BS)
        for BS in BSlist:
            # Filtrar las filas que cumplen con las condiciones del ángulo y muestras válidas
            filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == 5) & (df['angle'] == BS)
            df_filtrado = df[filtro].copy()  # Crear una copia del DataFrame filtrado
            # Reemplazar valores de -100 por NaN en la columna del valor de MAC
            df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
            # Filtrado para eliminar outliers fuera de ±3 sigma
            df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
            # Calcular la desviación media y el Ranging Error
            mean_RangingError = df_filtrado[mac_deseada].mean()
            mediana = df_filtrado[mac_deseada].median()
            # Guardar los valores calculados
            ranging_error_por_BS[BS] = mean_RangingError
            offset_por_BS[BS] = mediana - 5  # Ajustar el offset
        # Guardar los resultados del Ranging Error y offset para la combinación de teléfono y banda
        ranging_error_resultados[nombre_telefono][band_name] = ranging_error_por_BS
        offset_por_telefono_banda[nombre_telefono][band_name] = offset_por_BS
# Aplicar el offset para ajustar el Ranging Error
for telefono in nombres_telefonos:
    for banda in ranging_error_resultados[telefono]:
        # Restar el offset correspondiente a cada Ranging Error
        for BS in ranging_error_resultados[telefono][banda]:
            ranging_error_resultados[telefono][banda][BS] -= offset_por_telefono_banda[telefono][banda][BS]
# Restamos la distancia real al ranging error para obtener el ranging error corregido
for telefono in nombres_telefonos:
    for banda in ranging_error_resultados[telefono]:
        for BS in ranging_error_resultados[telefono][banda]:
            ranging_error_resultados[telefono][banda][BS] -= 5
# Convertir los resultados del Ranging Error a un DataFrame en formato rectangular
df_ranging_error = pd.concat(
    {telefono: pd.DataFrame(bandas) for telefono, bandas in ranging_error_resultados.items()},
    axis=1
)
# Redondear los valores a 3 decimales
df_ranging_error = df_ranging_error.round(3)
# Imprimir el DataFrame de Ranging Error
print(df_ranging_error)
# Guardar el DataFrame en un archivo CSV
df_ranging_error.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/RangingError.csv')
# Graficar los resultados del Ranging Error para cada teléfono
for nombre_telefono in nombres_telefonos:
    plt.figure(figsize=(12, 8))
    for band_name, color in colores_bandas.items():
        if band_name in ranging_error_resultados[nombre_telefono] and band_name == 'L2.4':
            ranging_data = ranging_error_resultados[nombre_telefono][band_name]
            plt.plot(
                list(ranging_data.keys()), 
                list(ranging_data.values()), 
                marker='o', 
                label=band_name, 
                color=color
            )
    plt.title(f'Ranging Error - {nombre_telefono}')
    plt.xlabel('Burst Size')
    plt.ylabel('Ranging Error (m)')
    plt.grid(True)
    # Marcamos el eje x=0 más grueso y oscuro
    plt.axhline(y=0, color='k', linewidth=1.5)
    plt.legend(title='Bandas')
    plt.show()

#####------------------------ANALISIS DE PDF RANGING ERROR (SEPARADO PARA L2.4) ------------------------#####
# Colores específicos para cada banda
colores_bandas = {
    'L5DFS': '#1f77b4',      # Azul
    'L5indoor': '#ff7f0e',   # Naranja
    'L2.4': '#2ca02c',       # Verde
    'G2.4': '#d62728',       # Rojo
    'G5': '#9467bd'          # Púrpura
}
# Crear histogramas de Ranging Error para todas las bandas disponibles para cada terminal
for nombre_telefono in nombres_telefonos:
    # Generar una gráfica por banda para cada teléfono
    for mac_deseada, band_name in mac_to_band.items():
        dataframe_name = f"{nombre_telefono}_{band_name}"        
        # Verificar si el DataFrame existe (si el teléfono soporta esta banda)
        if dataframe_name in dataframes:
            df = dataframes[dataframe_name]          
            # Filtrar solo las filas correspondientes a la distancia 5m y BS 8
            df_filtrado = df[(df['y'] == 5) & (df['angle'] == 8)].copy()
            # Reemplazar valores de -100 por NaN
            df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)           
            # Filtrado para eliminar outliers fuera de ±3 sigma
            df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]            
            # Calcular el *Ranging Error* para cada muestra
            mediana = df_filtrado[mac_deseada].median()
            df_filtrado['ranging_error'] = df_filtrado[mac_deseada] - mediana            
            # Crear el histograma de errores de ranging
            ranging_error_data = df_filtrado['ranging_error'].dropna().tolist()
            print(df_filtrado['ranging_error'])
            # Si hay datos, graficar el histograma
            if len(ranging_error_data) > 0:
                plt.figure(figsize=(10, 6))
                plt.hist(ranging_error_data, bins=15, alpha=0.75, label=f'{band_name}', color=colores_bandas.get(band_name, 'gray'))
                #guardamos el histograma en un archivo png
                plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/Histograma_{nombre_telefono}_{band_name}.png')
                                
                # Añadir detalles a la gráfica
                plt.title(f'Histograma del Ranging Error - {nombre_telefono} (BS 8) - {band_name}')
                plt.xlabel('Ranging Error (m)')
                plt.ylabel('Frecuencia')
                plt.grid(True)
                plt.axvline(x=0, color='k', linewidth=1.5, linestyle='--')  # Línea vertical en x=0 para referencia
                plt.legend(title='Banda')
                plt.show()


#####------------------------ANALISIS DE LA DESVIACION ESTANDAR ------------------------#####
# Diccionario para almacenar los resultados de la desviación estándar por banda
desviacion_estandar_resultados = {telefono: {} for telefono in nombres_telefonos}
# Combinaciones a excluir
exclusiones = ['Pixel6Pro_L2.4', 'Pixel6Pro_G2.4', 'Pixel6Pro_L5indoor', 'M2007J3SY_L5DFS']
# Iterar sobre cada combinación de teléfono y banda (MAC)
for nombre_telefono in nombres_telefonos:
    for mac_deseada in macs:
        # Nombre de la banda y DataFrame correspondiente
        band_name = mac_to_band[mac_deseada]
        dataframe_name = f"{nombre_telefono}_{band_name}"
        # Excluir combinaciones no deseadas
        if dataframe_name in exclusiones:
            continue
        # Obtener el DataFrame correspondiente
        df = dataframes[dataframe_name]
        # Inicializar diccionario para almacenar la desviación estándar por BS
        desviacion_estandar_por_BS = {}
        # Iterar sobre cada ángulo (BS)
        for BS in BSlist:
            # Filtrar las filas que cumplen con las condiciones del ángulo y muestras válidas
            filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == 5) & (df['angle'] == BS)
            df_filtrado = df[filtro].copy()  # Crear una copia del DataFrame filtrad
            # Reemplazar valores de -100 por NaN en la columna del valor de MAC
            df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
            # Filtrado para eliminar outliers fuera de ±3 sigma
            df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
            # Calcular la desviación estándar
            desviacion_media = df_filtrado[mac_deseada].std()  
            # Guardar la desviación estándar calculada por ángulo
            desviacion_estandar_por_BS[BS] = desviacion_media
        # Guardar los resultados de la desviación estándar para la combinación de teléfono y banda
        desviacion_estandar_resultados[nombre_telefono][band_name] = desviacion_estandar_por_BS
# Convertir los resultados de la desviación estándar a un DataFrame en formato rectangular
df_desviacion_estandar = pd.concat(
    {telefono: pd.DataFrame(bandas) for telefono, bandas in desviacion_estandar_resultados.items()},
    axis=1
)
# Redondear los valores a 3 decimales
df_desviacion_estandar = df_desviacion_estandar.round(3)
# Imprimir el DataFrame de Desviación Estándar
print("\nDesviación Estándar\n")
print(df_desviacion_estandar)
# Guardar el DataFrame en un archivo CSV
df_desviacion_estandar.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/DesviacionEstandar.csv')

#####------------------------ANALISIS DE estimacion medida ------------------------#####
# Inicializar diccionarios para almacenar los resultados de Ranging Error y offset
ranging_error_resultados = {telefono: {} for telefono in nombres_telefonos}
offset_por_telefono_banda = {telefono: {} for telefono in nombres_telefonos}
# Combinaciones a excluir
exclusiones = ['Pixel6Pro_L2.4', 'Pixel6Pro_G2.4', 'Pixel6Pro_L5indoor', 'M2007J3SY_L5DFS']
# Iterar sobre cada combinación de teléfono y banda (MAC)
for nombre_telefono in nombres_telefonos:
    for mac_deseada in macs:
        # Nombre de la banda y DataFrame correspondiente
        band_name = mac_to_band[mac_deseada]
        dataframe_name = f"{nombre_telefono}_{band_name}"
        # Excluir combinaciones no deseadas
        if dataframe_name in exclusiones:
            continue
        # Obtener el DataFrame correspondiente
        df = dataframes[dataframe_name]
        # Inicializar diccionarios para almacenar los resultados por BS
        ranging_error_por_BS = {}
        offset_por_BS = {}
        # Iterar sobre cada ángulo (BS)
        for BS in BSlist:
            # Filtrar las filas que cumplen con las condiciones del ángulo y muestras válidas
            filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == 5) & (df['angle'] == BS)
            df_filtrado = df[filtro].copy()  # Crear una copia del DataFrame filtrado
            # Reemplazar valores de -100 por NaN en la columna del valor de MAC
            df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
            # Filtrado para eliminar outliers fuera de ±3 sigma
            df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
            # Calcular la desviación media y el Ranging Error
            mean_RangingError = df_filtrado[mac_deseada].mean()
            mediana = df_filtrado[mac_deseada].median()
            # Guardar los valores calculados
            ranging_error_por_BS[BS] = mean_RangingError
            offset_por_BS[BS] = mediana - 5  # Ajustar el offse
        # Guardar los resultados del Ranging Error y offset para la combinación de teléfono y banda
        ranging_error_resultados[nombre_telefono][band_name] = ranging_error_por_BS
        offset_por_telefono_banda[nombre_telefono][band_name] = offset_por_BS
# Aplicar el offset para ajustar el Ranging Error
for telefono in nombres_telefonos:
    for banda in ranging_error_resultados[telefono]:
        # Restar el offset correspondiente a cada Ranging Error
        for BS in ranging_error_resultados[telefono][banda]:
            ranging_error_resultados[telefono][banda][BS] -= offset_por_telefono_banda[telefono][banda][BS]
# Convertir los resultados del Ranging Error a un DataFrame en formato rectangular
df_ranging_error = pd.concat(
    {telefono: pd.DataFrame(bandas) for telefono, bandas in ranging_error_resultados.items()},
    axis=1
)
# Redondear los valores a 3 decimales
df_ranging_error = df_ranging_error.round(3)
# Imprimir el DataFrame de Ranging Error
print("Ranging Error")
print(df_ranging_error)
# Guardar el DataFrame en un archivo CSV
df_ranging_error.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/RangingError.csv')

#####------------------------ANALISIS DEL offset tabla de todos ------------------------#####
import pandas as pd
# Inicializar un diccionario para almacenar los resultados del offset a 5 metros
offset_5m_resultados = {telefono: {} for telefono in nombres_telefonos}
# Combinaciones a excluir
exclusiones = ['Pixel6Pro_L2.4', 'Pixel6Pro_G2.4', 'Pixel6Pro_L5indoor', 'M2007J3SY_L5DFS']
# Iterar sobre cada combinación de teléfono y banda (MAC)
for nombre_telefono in nombres_telefonos:
    for mac_deseada in macs:
        # Nombre de la banda y DataFrame correspondiente
        band_name = mac_to_band[mac_deseada]
        dataframe_name = f"{nombre_telefono}_{band_name}"
        # Excluir combinaciones no deseadas
        if dataframe_name in exclusiones:
            continue
        # Obtener el DataFrame correspondiente
        df = dataframes[dataframe_name]
        # Inicializar un diccionario para almacenar los offsets por BS
        offset_por_BS = {}
        # Iterar sobre cada ángulo (BS)
        for BS in BSlist:
            # Filtrar las filas para y = 5 metros y el BS correspondiente
            filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == 5) & (df['angle'] == BS)
            df_filtrado = df[filtro].copy()
            # Reemplazar valores de -100 por NaN en la columna del valor de MAC
            df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
            # Filtrado para eliminar outliers fuera de ±2 sigma
            df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
            # Calcular la mediana y el offset (mediana - 5 metros)
            mediana = df_filtrado[mac_deseada].median()
            offset = mediana - 5
            # Guardar el offset para el BS actual
            offset_por_BS[BS] = offset
        # Guardar los resultados del offset para la combinación de teléfono y banda
        offset_5m_resultados[nombre_telefono][band_name] = offset_por_BS
# Convertir los resultados del offset a un DataFrame en formato rectangular
df_offset_5m = pd.concat(
    {telefono: pd.DataFrame(bandas) for telefono, bandas in offset_5m_resultados.items()},
    axis=1
)
# Redondear los valores a 3 decimales
df_offset_5m = df_offset_5m.round(3)
# Imprimir el DataFrame de offset a 5 metros
print(df_offset_5m)
# Guardar el DataFrame en un archivo CSV
df_offset_5m.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/Offset_5m.csv')

#####------------------------ANALISIS DEL RMSE 5 GRAFICAS POR BANDA ------------------------#####
# Inicializamos el diccionario para almacenar los resultados del RMSE por cada teléfono y banda (MAC)
rmse_resultados = {telefono: {} for telefono in nombres_telefonos}
# Colores específicos para cada teléfono
colores_telefonos = {
    'Pixel3a': '#1f77b4',      # Azul
    'Pixel4a': '#ff7f0e',      # Naranja
    'Pixel6Pro': '#2ca02c',    # Verde
    'M2007J3SY': '#d62728'     # Rojo
}
# Combinaciones a excluir
exclusiones = ['Pixel6Pro_L2.4', 'Pixel6Pro_G2.4', 'Pixel6Pro_L5indoor', 'M2007J3SY_L5DFS']
# Iterar sobre cada combinación de teléfono y banda (MAC)
for nombre_telefono in nombres_telefonos:
    for mac_deseada in macs:
        # Nombre de la banda y DataFrame correspondiente
        band_name = mac_to_band[mac_deseada]
        dataframe_name = f"{nombre_telefono}_{band_name}"
        # Excluir combinaciones no deseadas
        if dataframe_name in exclusiones:
            continue
        # Obtener el DataFrame correspondiente
        df = dataframes[dataframe_name]
        # Filtrar solo las filas correspondientes a la distancia 5m
        df_5m = df[df['y'] == 5]
        # Inicializar diccionarios para almacenar RMSE por BS
        rmse_por_BS = {}
        # Iterar sobre cada ángulo (BS)
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
            # Almacenar el RMSE calculado por ángulo
            rmse_por_BS[BS] = rmse
        # Guardar los resultados del RMSE para la combinación de teléfono y banda
        rmse_resultados[nombre_telefono][band_name] = rmse_por_BS
# Convertir los resultados del RMSE a un DataFrame en formato rectangular
df_rmse = pd.concat(
    {telefono: pd.DataFrame(bandas) for telefono, bandas in rmse_resultados.items()}, 
    axis=1
)
# Imprimir el DataFrame de RMSE en formato tabla rectangular
print(df_rmse)
# Guardar el DataFrame en un archivo CSV
df_rmse.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/RMSE.csv')
# Graficar los resultados del RMSE para cada banda (5 gráficas en total)
for band_name in set(mac_to_band.values()):
    plt.figure(figsize=(12, 8))
    for nombre_telefono in nombres_telefonos:
        if band_name in rmse_resultados[nombre_telefono]:
            rmse_data = rmse_resultados[nombre_telefono][band_name]
            plt.plot(list(rmse_data.keys()), list(rmse_data.values()), marker='o', label=nombre_telefono, color=colores_telefonos[nombre_telefono])
    plt.title(f'RMSE - {band_name}')
    plt.xlabel('Burst Size')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend(title='Teléfonos')
    plt.show()

#####------------------------ANALISIS DEL RANGING ERROR 5 GRAFICAS POR BANDA ------------------------#####
# Inicializamos el diccionario para almacenar los resultados del Ranging Error por cada teléfono y banda (MAC)
ranging_error_resultados = {telefono: {} for telefono in nombres_telefonos}
offset_por_telefono_banda = {telefono: {} for telefono in nombres_telefonos}
# Colores específicos para cada teléfono
colores_telefonos = {
    'Pixel3a': '#1f77b4',      # Azul
    'Pixel4a': '#ff7f0e',      # Naranja
    'Pixel6Pro': '#2ca02c',    # Verde
    'M2007J3SY': '#d62728'     # Rojo
}
# Combinaciones a excluir
exclusiones = ['Pixel6Pro_L2.4', 'Pixel6Pro_G2.4', 'Pixel6Pro_L5indoor', 'M2007J3SY_L5DFS']
# Iterar sobre cada combinación de teléfono y banda (MAC)
for nombre_telefono in nombres_telefonos:
    for mac_deseada in macs:
        # Nombre de la banda y DataFrame correspondiente
        band_name = mac_to_band[mac_deseada]
        dataframe_name = f"{nombre_telefono}_{band_name}"
        # Excluir combinaciones no deseadas
        if dataframe_name in exclusiones:
            continue
        # Obtener el DataFrame correspondiente
        df = dataframes[dataframe_name]
        # Inicializar diccionarios para almacenar los resultados por BS
        ranging_error_por_BS = {}
        offset_por_BS = {}
        # Iterar sobre cada ángulo (BS)
        for BS in BSlist:
            # Filtrar las filas que cumplen con las condiciones del ángulo y muestras válidas
            filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == 5) & (df['angle'] == BS)
            df_filtrado = df[filtro].copy()  # Crear una copia del DataFrame filtrado
            # Reemplazar valores de -100 por NaN en la columna del valor de MAC
            df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
            # Filtrado para eliminar outliers fuera de ±3 sigma
            df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
            # Calcular la desviación media y el Ranging Error
            mean_RangingError = df_filtrado[mac_deseada].mean()
            mediana = df_filtrado[mac_deseada].median()
            # Guardar los valores calculados
            ranging_error_por_BS[BS] = mean_RangingError
            offset_por_BS[BS] = mediana - 5  # Ajustar el offset
        # Guardar los resultados del Ranging Error y offset para la combinación de teléfono y banda
        ranging_error_resultados[nombre_telefono][band_name] = ranging_error_por_BS
        offset_por_telefono_banda[nombre_telefono][band_name] = offset_por_BS
# Aplicar el offset para ajustar el Ranging Error
for telefono in nombres_telefonos:
    for banda in ranging_error_resultados[telefono]:
        # Restar el offset correspondiente a cada Ranging Error
        for BS in ranging_error_resultados[telefono][banda]:
            ranging_error_resultados[telefono][banda][BS] -= offset_por_telefono_banda[telefono][banda][BS]
# Restamos la distancia real al ranging error para obtener el ranging error corregido
for telefono in nombres_telefonos:
    for banda in ranging_error_resultados[telefono]:
        for BS in ranging_error_resultados[telefono][banda]:
            ranging_error_resultados[telefono][banda][BS] -= 5
# Convertir los resultados del Ranging Error a un DataFrame en formato rectangular
df_ranging_error = pd.concat(
    {telefono: pd.DataFrame(bandas) for telefono, bandas in ranging_error_resultados.items()},
    axis=1
)
# Redondear los valores a 3 decimales
df_ranging_error = df_ranging_error.round(3)
# Imprimir el DataFrame de Ranging Error
print(df_ranging_error)
# Guardar el DataFrame en un archivo CSV
df_ranging_error.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/RangingError.csv')
# Graficar los resultados del Ranging Error para cada banda (5 gráficas en total)
for band_name in set(mac_to_band.values()):
    plt.figure(figsize=(12, 8))
    for nombre_telefono in nombres_telefonos:
        if band_name in ranging_error_resultados[nombre_telefono]:
            ranging_data = ranging_error_resultados[nombre_telefono][band_name]
            plt.plot(
                list(ranging_data.keys()), 
                list(ranging_data.values()), 
                marker='o', 
                label=nombre_telefono, 
                color=colores_telefonos[nombre_telefono]
            )
    plt.title(f'Ranging Error - {band_name}')
    plt.xlabel('Burst Size')
    plt.ylabel('Ranging Error (m)')
    plt.grid(True)
    # Marcamos el eje x=0 más grueso y oscuro
    plt.axhline(y=0, color='k', linewidth=1.5)
    plt.legend(title='Teléfonos')
    plt.show()
