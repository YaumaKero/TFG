import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os 
import seaborn as sns
from scipy.stats import ttest_1samp


#####-----------------------------FILTRADO MOVILES-----------------------------------#####
telefonos = ["Pixel 6 Pro","Pixel 3a","Pixel 4a","M2007J3SY"]
print("Que telefono quieres escoger?\n")
for i in range(len(telefonos)):
    print(str(i) + " " + telefonos[i])
telefono = input()
telefono = int(telefono)
nombre_telefono = telefonos[telefono].replace(" ", "")
df = pd.read_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/muestras3/muestras3_' + nombre_telefono + '.csv')

############ --------------------------- Diccionarios -----------------------######      
BSlist = list(range(2, 30))
ys = [1, 5, 10, 15, 20, 25]
#ys = [5]
rmse_por_BS_y = {}
bias_por_BS_y = {}
offset = {}
muestras_validas = {}  
Nmuestras = {} 
BSlogrado = {}
desviaciones_por_y = {}
RangingError_por_y = {}


##### --------------------------- MACS -----------------------#####
#mac_deseada = 'c4:41:1e:fa:07:db#2_value'      #DFS
#mac_deseada = 'c4:41:1e:fa:07:da#2_value'     #indoor
#mac_deseada = 'c4:41:1e:fa:07:d9#2_value'     #L2_4
#mac_deseada = 'cc:f4:11:47:ef:eb#2_value'     #G2_4
#mac_deseada = 'cc:f4:11:47:ef:e7#2_value'     #G5

mac_dict = {
    0: {'mac': 'c4:41:1e:fa:07:db#2_value', 'band': 'L5DFS'},
    1: {'mac': 'c4:41:1e:fa:07:da#2_value', 'band': 'L5indoor'},
    2: {'mac': 'c4:41:1e:fa:07:d9#2_value', 'band': 'L2_4'},
    3: {'mac': 'cc:f4:11:47:ef:eb#2_value', 'band': 'G2_4'},
    4: {'mac': 'cc:f4:11:47:ef:e7#2_value', 'band': 'G5'}
}
print("Que mac quieres escoger?\n")
for key, value in mac_dict.items():
    print(f"{key} {value['mac']} ({value['band']})")
mac = int(input())
mac_deseada = mac_dict.get(mac)
if mac_deseada is None:
    print("Mac no valida")
    exit()
else:
    mac_deseada = mac_deseada['mac']       

mac = mac_deseada[:17]
mac_to_band = {
    'c4:41:1e:fa:07:db#2_value': 'L5DFS',
    'c4:41:1e:fa:07:da#2_value': 'L5indoor',
    'c4:41:1e:fa:07:d9#2_value': 'L2_4',
    'cc:f4:11:47:ef:eb#2_value': 'G2_4',
    'cc:f4:11:47:ef:e7#2_value': 'G5'
}
band_name = mac_to_band.get(mac_deseada, 'Desconocido')


### --------------------------- Calcular parametros -----------------------#####
for y in ys:
    desviaciones_por_BS = {}
    RangingError_por_BS = {}
    for BS in BSlist:
        filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == y) & (df['angle'] == BS)
        df_filtrado = df[filtro].copy()              
        df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
        #FILTRADO +- 3 sigma
        df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
        desviacion_media = df_filtrado[mac_deseada].std() 
        mean_RangingError = df_filtrado[mac_deseada].mean() 
        desviaciones_por_BS[BS] = desviacion_media
        RangingError_por_BS[BS] = mean_RangingError
        Nmuestras.setdefault(BS, {})[y] = df_filtrado[mac_deseada].count()
        mediana = df_filtrado[mac_deseada].median()        
        bias_por_BS_y.setdefault(BS, {})[y] = mediana
        offset.setdefault(BS, {})[y] = mediana - y        
        # Calcular el RMSE
        df_filtrado[mac_deseada] -= mediana
        df_filtrado[mac_deseada] **= 2
        rmse = (df_filtrado[mac_deseada].sum() / Nmuestras[BS][y]) ** 0.5
        rmse_por_BS_y.setdefault(BS, {})[y] = rmse
    desviaciones_por_y[y] = desviaciones_por_BS
    RangingError_por_y[y] = RangingError_por_BS

### --------------------------- Estudio Outliers -----------------------#####
# Filtro para las muestras del Pixel 4a a 5 metros y ángulo 8
filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == 5) & (df['angle'] == 8)
df_filtrado = df[filtro].copy()
# Reemplazamos los valores de -100 por NaN
df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
# Nos quedamos con la primera muestra de cada sampleNumber
df_filtrado = df_filtrado.drop_duplicates(subset='sampleNumber', keep='first')
# Ordenamos por sampleNumber
df_filtrado = df_filtrado.sort_values(by='sampleNumber')
# Gráfico de barras con los ajustes de los ejes
df_filtrado.set_index('sampleNumber')[mac_deseada].plot(kind='bar')
# Configuración del título y ejes
plt.title(f'Muestras cronologicas - {nombre_telefono} - {band_name}')
plt.xlabel('Numero de muestra')
plt.ylabel('Distancia medida [m]')
# Ajustar el rango de los ejes
plt.xlim(1, 200)
plt.ylim(0, 3800)
# Personalizar las marcas en el eje X para que solo aparezcan 50, 100, 150 y 200
plt.xticks(ticks=[49, 99, 149, 199], labels=['50', '100', '150', '200'])
# Guardar el gráfico y mostrarlo
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_MuestrasOrdenadasOutliers.png')
plt.show()


### --------------------------- Estudio muestras sin filtrar -----------------------#####
#Para todas las muestras de todos los BS en y=5 hacemos un histograma
filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == 5)
df_filtrado = df[filtro].copy()
df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
#df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
df_filtrado[mac_deseada].hist(bins=100)
plt.title(f'Histograma - {nombre_telefono} - {band_name}')
plt.xlabel('Distancia medida [m]')
plt.ylabel('Nº de muestras')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_Histograma.png')
plt.show()


### --------------------------- Estudio muestras ordenadas por samplenumber -----------------------#####
#creamos un grafico con las muestras ordenadas por samplenumber a 5 metros
filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == 5)
df_filtrado = df[filtro].copy()
df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
#df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
df_filtrado = df_filtrado.sort_values(by='sampleNumber')
df_filtrado[mac_deseada].plot()
plt.title(f'Muestras ordenadas por samplenumber - {nombre_telefono} - {band_name}')
plt.xlabel('Muestra')
plt.ylabel('Distancia medida [m]')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_MuestrasOrdenadas.png')
plt.show()

##### --------------------------- MUESTRAS VALIDAS -----------------------#####
muestras_validas = {}
for y in ys:
    for BS in BSlist:
        filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == y) & (df['angle'] == BS)
        df_filtrado = df[filtro].copy()
        # Filtrar muestras donde el ángulo en la columna correspondiente coincida con el ángulo actual
        if nombre_telefono == "Pixel6Pro":
            filtro_BS = (df_filtrado[mac + '#2_samples_averaged'] == BS - 1)
        else:
            filtro_BS = (df_filtrado[mac + '#2_samples_averaged'] == BS)    
        df_filtrado = df_filtrado[filtro_BS]
        num_muestras_validas = df_filtrado[mac_deseada].count()
        # Aplicar divisiones según el número de muestras
        if num_muestras_validas > 200:
            num_muestras_validas /= 2
        if num_muestras_validas > 400:
            num_muestras_validas /= 4
        num_muestras_validas = round(num_muestras_validas)
        num_muestras_validas = (num_muestras_validas / 200) * 100
        # Acumular el total para cada BS
        if BS not in muestras_validas:
            muestras_validas[BS] = []
        muestras_validas[BS].append(num_muestras_validas)
# Calcular la media de porcentajes de muestras válidas para cada BS
media_muestras_validas = {BS: np.mean(muestras) for BS, muestras in muestras_validas.items()}
# Convertir el diccionario a un DataFrame
df_muestras_validas_media = pd.DataFrame(list(media_muestras_validas.items()), columns=['BS', 'Media Porcentaje Muestras Válidas'])
df_muestras_validas_media.set_index('BS', inplace=True)
# Mostrar el DataFrame final
print(f'\nMedia del porcentaje de muestras que alcanzan el BS pedido - {nombre_telefono} - {band_name}')
print(df_muestras_validas_media)

#---Codigo antinguo con 5 distancias---#
# muestras_validas = {}
# for y in ys:
#     for BS in BSlist:
#         filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == y) & (df['angle'] == BS)
#         df_filtrado = df[filtro].copy()
#         #Filtrar muestras donde el ángulo en la columna correspondiente coincida con el ángulo actual en caso de ser el Pixel 6 Pro restar 1 en vez de 0
#         if nombre_telefono == "Pixel6Pro":
#             filtro_BS = (df_filtrado[mac + '#2_samples_averaged'] == BS - 1)
#         else:
#             filtro_BS = (df_filtrado[mac + '#2_samples_averaged'] == BS)        
#         df_filtrado = df_filtrado[filtro_BS]
#         num_muestras_validas = df_filtrado[mac_deseada].count()
#         if num_muestras_validas > 200:
#             num_muestras_validas /= 2
#         if num_muestras_validas > 400:
#             num_muestras_validas /= 4
#         num_muestras_validas = round(num_muestras_validas)
#         num_muestras_validas = (num_muestras_validas / 200) * 100
#         muestras_validas.setdefault(BS, {})[y] = num_muestras_validas
# df_muestras_validas = pd.DataFrame.from_dict(muestras_validas, orient='index', columns=ys)
# df_muestras_validas.index.name = 'BS'
# print(f'\nPorcentaje muestras que alcanzan el BS pedido - {nombre_telefono} - {band_name}')
# #df_muestras_validas = df_muestras_validas.transpose()
# print(df_muestras_validas)


################################################################################################
###########  TABLAS   ########################################################################################
################################################################################################

##### --------------------------- Tabla RMSE (5metros) -----------------------#####
df_rmse = pd.DataFrame.from_dict(rmse_por_BS_y, orient='index', columns=ys)
df_rmse.index.name = 'Angle'
#df_rmse = df_rmse.loc[5]
print(f"\nRMSE - {nombre_telefono} - {band_name} - 5 metros")
print(df_rmse)

##### --------------------------- Tabla Bias -----------------------#####
df_bias = pd.DataFrame.from_dict(bias_por_BS_y, orient='index', columns=ys)
df_bias.index.name = 'Angle'
print(f"\nBias - {nombre_telefono} - {band_name}")
print(df_bias)

##### --------------------------- Tabla Offset -----------------------#####
df_offset = pd.DataFrame.from_dict(offset, orient='index', columns=ys)
df_offset.index.name = 'Angle'
print(f"\nOffset - {nombre_telefono} - {band_name}")
print(df_offset)
#Media del Offset 
media_offset = df_offset.mean()
print("Media del offset:", media_offset.round(2))

##### --------------------------- Tabla BS lograddo -----------------------#####
for y in ys:
    for BS in BSlist:
        filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == y) & (df['angle'] == BS) & (df[mac + '#2_samples_averaged'] != 0)
        df_filtrado = df[filtro].copy()  # Crear una copia del DataFrame para evitar SettingWithCopyWarning
        df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
        media = df_filtrado[mac + '#2_samples_averaged'].mean()
        BSlogrado.setdefault(BS, {})[y] = media
df_BSlogrado = pd.DataFrame.from_dict(BSlogrado, orient='index', columns=ys)
df_BSlogrado.index.name = 'Angle'
print(f'\nBS Medio Logrado - {nombre_telefono} - {band_name}')
print(df_BSlogrado)


################################################################################################
#######   GRAFICAS    ########################################################################################
################################################################################################

##### --------------------------- Grafica RMSE (6 Curvas) -----------------------#####
df_rmse.plot()
plt.title(f'RMSE(BS) - {nombre_telefono} - {band_name}')
plt.xlabel('Burst Size')
plt.ylabel('RMSE')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_RMSE.png')
plt.show()

##### --------------------------- Grafica RMSE (Media) con barras error -----------------------#####
df_rmse = df_rmse.transpose()
df_rmse.mean().plot()
plt.show(block= False)
mean_rmse = df_rmse.mean()
std_rmse = df_rmse.std()  
plt.errorbar(mean_rmse.index, mean_rmse.values, yerr=std_rmse.values, fmt='o', capsize=5)
plt.title(f'Media de RMSE(BS) con barras de error {nombre_telefono} - {band_name}')
plt.xlabel('Angle (BS)')
plt.ylabel('RMSE')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_RMSE_media.png')
plt.show()
df_rmse = df_rmse.transpose()

##### --------------------------- Grafica RMSE 29 BS -----------------------#####
df_rmse = df_rmse.transpose()
df_rmse_group1 = df_rmse.iloc[:, :10]
df_rmse_group2 = df_rmse.iloc[:, 10:20]
df_rmse_group3 = df_rmse.iloc[:, 20:]
ax = df_rmse_group1.plot(marker='.', linestyle='-', figsize=(10, 6), alpha=1)
df_rmse_group2.plot(ax=ax, marker='^', linestyle='dotted', alpha=1,)
df_rmse_group3.plot(ax=ax, marker='s', linestyle='--', alpha=1)
plt.title(f'RMSE(Y) - {nombre_telefono} - {band_name}')
plt.xlabel('Distancia (Y)')
plt.ylabel('RMSE')
plt.legend(loc='upper right', title='BS')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_RMSE_Distancia.png')
plt.show()
# Número de columnas en df_rmse
num_columns = len(df_rmse.columns)
# Dividimos en dos grupos: primeras 14 columnas y las restantes
columns_first_half = df_rmse.columns[:14]
columns_second_half = df_rmse.columns[14:]

# Función para crear un gráfico de regresión lineal combinando todas las columnas
def plot_all_regressions(df_rmse, nombre_telefono, band_name):
    plt.figure(figsize=(14, 8))  # Ajustar el tamaño de la figura si es necesario
    markers = ['.', '^', 's']  # Marcadores para diferenciar grupos de columnas
    linestyles = ['-', 'dotted', '--']  # Diferentes estilos de línea

    # Contador para alternar entre marcadores y estilos
    marker_counter = 0
    linestyle_counter = 0

    # Iterar sobre cada columna del DataFrame
    for i, column in enumerate(df_rmse.columns):
        # Cambiar marcador y estilo cada 10 columnas para diferenciación visual
        if i % 10 == 0 and i != 0:
            marker_counter = (marker_counter + 1) % len(markers)
            linestyle_counter = (linestyle_counter + 1) % len(linestyles)

        # Preparar los datos para la regresión lineal
        regression_model = LinearRegression()
        df_rmse_cleaned = df_rmse.dropna(subset=[column])
        X = df_rmse_cleaned.index.values.reshape(-1, 1)
        y = df_rmse_cleaned[column].values

        # Ajustar la regresión lineal
        regression_model.fit(X, y)
        y_pred = regression_model.predict(X)

        # Trazar la regresión lineal
        plt.plot(
            X, 
            y_pred, 
            linestyle=linestyles[linestyle_counter], 
            marker=markers[marker_counter], 
            label=f'BS {column}', 
            linewidth=2
        )

    # Configurar el título y las etiquetas
    plt.title(f'Regresiones Lineales del RMSE(Y) - {nombre_telefono} - {band_name}')
    plt.xlabel('Distancia [m]')
    plt.ylabel('RMSE')
    plt.grid(True)

    # Establecer los límites del eje Y para centrar las líneas de regresión
    plt.ylim(-1, 4)  # Ajustar el rango de Y según las necesidades

    # Ajustar la leyenda dentro de los márgenes de la figura
    plt.legend(
        loc='upper right', 
        title='BS', 
        bbox_to_anchor=(1.05, 1),  # Ajustar la posición de la leyenda
        borderaxespad=0.001,  # Ajustar el espacio entre la leyenda y el gráfico
        fontsize='small'  # Reducir el tamaño de la fuente si es necesario
    )

    # Ajustar los márgenes de la gráfica para dar más espacio a la leyenda
    plt.subplots_adjust(right=0.8)

    # Guardar la gráfica
    plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_RegresionesRMSE_Completo.png', bbox_inches='tight')
    plt.show()

# Llamar a la función para crear la gráfica combinada
plot_all_regressions(df_rmse, nombre_telefono, band_name)


##### --------------------------- Grafica Bias menos offset -----------------------#####
df_bias_distancia = df_bias.transpose()
primer_valor_distancia = df_bias_distancia.iloc[0, :] - 1 
df_bias_distancia = df_bias_distancia.sub(primer_valor_distancia, axis=1)
linestyles = ['-', '--', '-.', ':']
plt.figure(figsize=(10, 6))
for i, column in enumerate(df_bias_distancia.columns):
    linestyle = linestyles[i // 10]  
    df_bias_distancia[column].plot(label=f'BS {column}', linestyle=linestyle)
plt.plot([1, 25], [1, 25], '--', linewidth=2, color='black')
plt.title(f'Bias(Distancia) - {nombre_telefono} - {band_name}')
plt.xlabel('Distancia')
plt.ylabel('Bias')
plt.legend(loc='upper right', title='BS')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_Bias.png')
plt.show()

##### --------------------------- Tabla y Grafica Desviacion Estandar (6 distancias) -----------------------#####
df_desviaciones = pd.DataFrame.from_dict(desviaciones_por_y, orient='index')
df_desviaciones.index.name = 'Y'
print(f'\nDesviacion Estandar - {nombre_telefono} - {band_name}')
df_desviaciones = df_desviaciones.transpose()
print(df_desviaciones)
#Grafica de las desviaciones por BS y Y
plt.figure(figsize=(10, 6))
for y in ys:
    plt.plot(desviaciones_por_y[y].keys(), desviaciones_por_y[y].values(), label=f'Y = {y}')
plt.title(f'Desviacion estandar - {nombre_telefono} - {band_name}')
plt.xlabel('Burst Size')
plt.ylabel('Desviación')
plt.legend(title='Y')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_DesviacionEstandar.png')
plt.show()

##----------------------------Ranging Error---------------------------------##
df_RangingError = pd.DataFrame.from_dict(RangingError_por_y, orient='index')
df_RangingError.index.name = 'Y'
print(f'\nRanging Error - {nombre_telefono} - {band_name}')
#media del offset
df_offset = df_offset.transpose()
#df_offset = df_offset.mean(axis=1)
df_RangingError = df_RangingError - df_offset - 5
df_RangingError = df_RangingError.transpose()
#df_RangingError = df_RangingError.mean(axis=1)
print(df_RangingError.round(3))
df_RangingError.plot()
plt.title(f'RangingError media - {nombre_telefono} - {band_name}')
plt.xlabel('Burst Size')
plt.ylabel('RangeError')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_RangingErrorMedia.png')
plt.show()
# print("Ranging Error medio:")
# df_RangingError = df_RangingError.round(3)
# print(df_RangingError)
df_RangingError += 5

### --------------------------- Error de medida -----------------------#####
#Al ranging error medio sin offset le restamos la distancia real
df_ErrorMedida = df_RangingError
df_ErrorMedida = df_ErrorMedida.transpose()
df_ErrorMedida = df_ErrorMedida.sub(ys, axis=0)
df_ErrorMedida = df_ErrorMedida.transpose()
df_ErrorMedida = df_ErrorMedida.abs()
print("Error de medida:")
print(df_ErrorMedida.round(3))

##### --------------------------- PDF Desviacion Estandar -----------------------#####
plt.figure(figsize=(10, 6))
for y in ys:
    sns.kdeplot(df_desviaciones[y], fill=False, bw_adjust=0.5, label=f'Y = {y}')
plt.title(f'Densidad de la Desviacion Estandar - {nombre_telefono} - {band_name}')
plt.xlabel('Range Error')
plt.ylabel('Densidad')
plt.legend(title='Y')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_pdfDesviacionEstandar.png')
plt.show()

##### ----------------------- Media de las Desviaciones Estandar (Media de las desitancias)-----------------------#####
df_desviaciones_mean = df_desviaciones.mean(axis=1)
df_desviaciones_mean.plot()
plt.title(f'Media de las 6 Desviaciones Estandar - {nombre_telefono} - {band_name}')
plt.xlabel('Ángulo')
plt.ylabel('Desviación')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_DesviacionEstandarMedia.png')
plt.show()
print("Media de las 6 Desviaciones Estandar:")
#df_desviaciones_mean = df_desviaciones_mean.transpose()
df_desviaciones_mean = df_desviaciones_mean.round(3)
print(df_desviaciones_mean)
df_desviaciones_mean = df_desviaciones_mean.to_frame().transpose()
df_desviaciones_mean.to_csv(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/tablas/stdOneSided.csv', mode='a', header=False)


# # Calcular la media de las desviaciones
# media_desviaciones = np.mean(list(desviaciones_por_y.values()))
# print("Media de las desviaciones:", media_desviaciones.round(2))
# #calculamos la t-student para n = 6 y un alpha de 0.05
# t_statistic, p_value = ttest_1samp(list(desviaciones_por_y.values()), 0)
# print("Valor p:", p_value.round(2))


#.................GRAFICA DENSIDAD DE PROBABILIDAD DE LAS DIFERENCIAS ENTRE LA DISTANCIA REAL Y LA MEDIDA.....................###
#.......................................Opcion antigua con gama colores...............#
x = np.array([1, 5, 10, 15, 20, 25])
y_45 = x  
#dataframe del ranging error pero la columna de 5 metros
#df_RangingError_5m = df_RangingError.loc[5]
#df_RangingError_5m = df_RangingError_5m.iloc[:, :20]
#df_bias_distancia = df_bias_distancia.iloc[17]
print(df_bias_distancia)
df_RangingError = df_RangingError.transpose()
df_bias_distancia = df_RangingError.loc[5]
print(df_bias_distancia)
df_bias_distancia = df_bias_distancia.transpose()
#df_bias_distancia = df_bias_distancia.iloc[:10]
#distancias = df_bias_distancia.subtract(x, axis=0) 
#---densidad ranging error a 5m---#
sns.kdeplot(df_bias_distancia - 5, fill=False, bw_adjust=0.3)
plt.title('Función de densidad del ranging error a 5m')
plt.xlabel('Ranging Error (m)')
plt.ylabel('Densidad')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_pdfRangingError5m.png')
plt.show()
#---densidad ranging error con 29 BS---#
# Transponer el DataFrame
df_ErrorMedida = df_ErrorMedida.transpose()
# Dividir el DataFrame en dos partes
df_ErrorMedida_1 = df_ErrorMedida.iloc[:, :15]  # Primeras 15 columnas
df_ErrorMedida_2 = df_ErrorMedida.iloc[:, 15:]  # Últimas 14 columnas
# Primer gráfico
sns.kdeplot(df_ErrorMedida_1, fill=False, bw_adjust=2)
plt.title('Función de densidad del ranging error 2-16 BS')
plt.xlabel('Ranging Error (m)')
plt.ylabel('Densidad')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_pdfRangingError2BS.png')
plt.show()
# Segundo gráfico
sns.kdeplot(df_ErrorMedida_2, fill=False, bw_adjust=2)
plt.title('Función de densidad del ranging error 17-29 BS')
plt.xlabel('Ranging Error (m)')
plt.ylabel('Densidad')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_pdfRangingError16BS.png')
plt.show()
# Transponer de nuevo si es necesario
df_RangingError = df_RangingError.transpose()
df_ErrorMedida = df_ErrorMedida.transpose()

### --------------------------- Coeficiente  -----------------------#####
#El coeficiente es desviacion tipica / ranging error medio
coeficiente = df_desviaciones / df_RangingError
print("Coeficiente:")
print(coeficiente.round(5))

################################################################################################

