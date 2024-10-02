import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os 
import seaborn as sns

#####-----------------------------FILTRADO MOVILES-----------------------------------#####

telefonos = ["Pixel 6 Pro","Pixel 3a","Pixel 4a","M2007J3SY"]
print("Que telefono quieres escoger?\n")
for i in range(len(telefonos)):
    print(str(i) + " " + telefonos[i])
telefono = input()
telefono = int(telefono)
nombre_telefono = telefonos[telefono].replace(" ", "")
df = pd.read_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/muestrasTwoSided/muestras2TwoSided_' + nombre_telefono + '.csv')
      
############ --------------------------- Diccionarios -----------------------#####
angulos = list(range(2, 12))
ys = [5]
rmse_por_angulo_y = {}
bias_por_angulo_y = {}
RangingError = {}
muestras_validas = {} 
kde_data = {}
desviaciones_por_y = {}
desviaciones_por_angulo = {}


##### --------------------------- MACS -----------------------#####

#mac_deseada = 'c4:41:1e:fa:07:db#1_value'      #DFS
#mac_deseada = 'c4:41:1e:fa:07:da#1_value'     #indoor
#mac_deseada = 'c4:41:1e:fa:07:d9#1_value'     #L2_4
#mac_deseada = 'cc:f4:11:47:ef:eb#1_value'     #G2_4
#mac_deseada = 'cc:f4:11:47:ef:e7#1_value'     #G5

mac_dict = {
    0: {'mac': 'c4:41:1e:fa:07:db#1_value', 'band': 'L5DFS'},
    1: {'mac': 'c4:41:1e:fa:07:da#1_value', 'band': 'L5indoor'},
    2: {'mac': 'c4:41:1e:fa:07:d9#1_value', 'band': 'L2_4'},
    3: {'mac': 'cc:f4:11:47:ef:eb#1_value', 'band': 'G2_4'},
    4: {'mac': 'cc:f4:11:47:ef:e7#1_value', 'band': 'G5'}
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
    'c4:41:1e:fa:07:db#1_value': 'L5DFS',
    'c4:41:1e:fa:07:da#1_value': 'L5indoor',
    'c4:41:1e:fa:07:d9#1_value': 'L2_4',
    'cc:f4:11:47:ef:eb#1_value': 'G2_4',
    'cc:f4:11:47:ef:e7#1_value': 'G5'
}
band_name = mac_to_band.get(mac_deseada, 'Desconocido')

##### --------------------------- Calcular RMSE, BIAS, RANGING ERROR -----------------------#####
df2 = df[df['angle'] == 11][[mac_deseada, 'y']]
offset = np.mean(df2.groupby(by='y').median().to_numpy())

for y in ys:
    for angulo in angulos:
        filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == y) & (df['angle'] == angulo)
        df_filtrado = df[filtro].copy()
        # Reemplazar los valores -100 por NaN
        df_filtrado[mac_deseada].replace(-100, np.nan, inplace=True)
        #FILTRADO +- 2 sigma
        df_filtrado = df_filtrado[np.abs(df_filtrado[mac_deseada] - df_filtrado[mac_deseada].mean()) <= (2 * df_filtrado[mac_deseada].std())]
        desviacion_media = df_filtrado[mac_deseada].std()   
        desviaciones_por_angulo[angulo] = desviacion_media
        # Contar el número de muestras válidas y almacenarlo en el diccionario de muestras válidas      
        muestras_validas.setdefault(angulo, {})[y] = df_filtrado[mac_deseada].count()
        # Calcular la mediana
        mediana = df_filtrado[mac_deseada].median()
        media = df_filtrado[mac_deseada].mean()  
        # Calcular el bias y el Ranging Error
        bias_por_angulo_y.setdefault(angulo, {})[y] = mediana
        RangingError.setdefault(angulo, {})[y] = media - y        
        # Calcular el RMSE
        df_filtrado[mac_deseada] -= y
        kde_data[angulo] = df_filtrado[mac_deseada].dropna()
        df_filtrado[mac_deseada] **= 2
        rmse = (df_filtrado[mac_deseada].sum() / muestras_validas[angulo][y]) ** 0.5
        rmse_por_angulo_y.setdefault(angulo, {})[y] = rmse
    desviaciones_por_y[y] = desviaciones_por_angulo


################################################################################################
###########  TABLAS   ########################################################################################
################################################################################################

##### --------------------------- Tabla RMSE -----------------------#####
df_rmse = pd.DataFrame.from_dict(rmse_por_angulo_y, orient='index', columns=ys)
df_rmse.index.name = 'Angle'
print(f"\nRMSE - {nombre_telefono} - {band_name}")
print(df_rmse)

##### --------------------------- Tabla y Grafica RMSE -----------------------#####
df_bias = pd.DataFrame.from_dict(bias_por_angulo_y, orient='index', columns=ys)
df_bias.index.name = 'Angle'
print(f"\nBias - {nombre_telefono} - {band_name}")
print(df_bias)
df_rmse.plot()
plt.title(f'RMSE TwoSided 5m - {nombre_telefono} - {band_name}')
plt.scatter(df_rmse.index, df_rmse.values, color='red')
plt.xticks(df_rmse.index)
plt.xlabel('Ángulo')
plt.ylabel('RMSE')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_RMSE_TwoSided.png')
plt.show()

##### --------------------------- Tabla y Grafica RanginError -----------------------#####
df_RangingError = pd.DataFrame.from_dict(RangingError, orient='index', columns=ys)
df_RangingError.index.name = 'Angle'
print(f"\nBias Normalizado - {nombre_telefono} - {band_name}")
print(df_RangingError)
df_RangingError.plot()
plt.title(f'RangingError(BS)_TwoSided_5m - {nombre_telefono} - {band_name}')
plt.scatter(df_RangingError.index, df_RangingError.values, color='red')
plt.xticks(df_RangingError.index)
plt.xlabel('Ángulo')
plt.ylabel('Ranging Error')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_RangingError_TwoSided_BS.png')
plt.show()

##### --------------------------- Grafica RanginError Density -----------------------#####
plt.figure(figsize=(10, 6))
for angulo, datos in kde_data.items():
    sns.kdeplot(data=datos, fill=False, label=f'BS {angulo}')
plt.title(f'RangingError Density TwoSided 5m - {nombre_telefono} - {band_name}')
plt.xlabel('Ranging Error')
plt.ylabel('Densidad')
plt.legend()
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_RangingError_Density_TwoSided.png')
plt.show()

##### --------------------------- Tabla Muestras Válidas -----------------------#####
df_muestras_validas = pd.DataFrame.from_dict(muestras_validas, orient='index', columns=ys)
df_muestras_validas.index.name = 'Angle'
print(f'\nNúmero de Muestras Válidas - {nombre_telefono} - {band_name}')
df_muestras_validas = df_muestras_validas.transpose()
print(df_muestras_validas)

##### --------------------------- Tabla de BS logrado -----------------------#####
BS_logrado = {}
for y in ys:
    for angulo in angulos:
        filtro = (df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199) & (df['y'] == y) & (df['angle'] == angulo)
        df_filtrado = df[filtro].copy()
        media = df_filtrado[mac + '#1_samples_averaged'].mean()
        BS_logrado.setdefault(angulo, {})[y] = media
df_BS_logrado = pd.DataFrame.from_dict(BS_logrado, orient='index', columns=ys)
df_BS_logrado.index.name = 'Angle'
print(f'\nBS Logrado - {nombre_telefono} - {band_name}')
print(df_BS_logrado)

##### --------------------------- Tabla y Grafica Desviacion Estandar -----------------------#####
df_desviaciones = pd.DataFrame.from_dict(desviaciones_por_y, orient='index')
df_desviaciones.index.name = 'Y'
print(f'\nDesviacion Estandar - {nombre_telefono} - {band_name}')
df_desviaciones = df_desviaciones.round(3)
print(df_desviaciones) 
df_desviaciones.to_csv(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/tablas/stdTwoSided.csv',mode='a', header=False)  
plt.figure(figsize=(10, 6))
for y in ys:
    plt.plot(desviaciones_por_y[y].keys(), desviaciones_por_y[y].values(), label=f'Y = {y}')
plt.title(f'Desviacion estandar - {nombre_telefono} - {band_name}')
plt.xlabel('Burst Size')
plt.ylabel('Desviación')
plt.legend(title='Y')
plt.savefig(f'C:/Users/jaume/Documents/VISUAL CODES/TFG/plots/{nombre_telefono}_{band_name}_DesviacionEstandar.png')
plt.show()    


################################################################################################
