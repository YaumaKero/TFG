import pandas as pd


#########Con este codigo pretendemos eliminar las medidas de Diogo, quedarnos con las bandas deseadas########
##############Tambien haremos el filtrado del movil que queremos estudiar#######################################

#directorio del csv inicial
csvdir = 'C:/Users/jaume/Documents/VISUAL CODES/TFG/Xiaomi - 2024-05-17--08-26-58.csv'

#en el archivo csv cambiamos todas las ";" por ","
with open(csvdir, 'r') as file:
    filedata = file.read()
    filedata = filedata.replace(';', ',')   
with open(csvdir, 'w') as file:
    file.write(filedata)

#creamos un dataframe con el csv 
df = pd.read_csv(csvdir)

#ordenamos el dataframe por la columna "date"
df = df.sort_values(by=['date'])

#Nos quedamos con las filas que su batch, los ultimos 4 caracteres sea superior a 5161
df['batch'] = df['batch'].astype(str).str[-4:].astype(int)
df = df[df['batch'] > 5160]
#Lista de telefonos
telefonos = ["Pixel 6 Pro","Pixel 3a","Pixel 4a","M2007J3SY"]
#printea por consola que telefono quiere escoger de 1 al 4
print("Que telefono quieres escoger?\n")
for i in range(len(telefonos)):
    print(str(i) + " " + telefonos[i])
telefono = input()
telefono = int(telefono)

#Nos quedamos con las filas que su modelo sea "telefono[0]"
df = df[df['model'] == telefonos[telefono]]

#ordenamos por y luego por angle y luego por SampleNumber
df = df.sort_values(by=['y','angle','sampleNumber'])

#lista macs
#mac_list_0 = ["c4:41:1e:fa:07:da#0_value","cc:f4:11:47:ef:eb#0_value","c4:41:1e:fa:07:db#0_value","c4:41:1e:fa:07:d9#0_value"]
mac_list_1_L5DFS = ["c4:41:1e:fa:07:db#2_value","c4:41:1e:fa:07:db#2_error","c4:41:1e:fa:07:db#2_samples_averaged"]
mac_list_1_L5indoor = ["c4:41:1e:fa:07:da#2_value","c4:41:1e:fa:07:da#2_error","c4:41:1e:fa:07:da#2_samples_averaged"]
mac_list_1_L2_4 = ["c4:41:1e:fa:07:d9#2_value","c4:41:1e:fa:07:d9#2_error","c4:41:1e:fa:07:d9#2_samples_averaged"]
mac_list_1_G2_4 = ["cc:f4:11:47:ef:eb#2_value","cc:f4:11:47:ef:eb#2_error","cc:f4:11:47:ef:eb#2_samples_averaged"]
mac_list_1_G5 = ["cc:f4:11:47:ef:e7#2_value","cc:f4:11:47:ef:e7#2_error","cc:f4:11:47:ef:e7#2_samples_averaged"]

#nos quedamos con las columnas de las macs y  batch	x	y	z	brand	model	angle	sampleNumber
df = df[["batch","x","y","z","brand","model","angle","sampleNumber", "date"] + mac_list_1_L5DFS + mac_list_1_L5indoor + mac_list_1_L2_4 + mac_list_1_G2_4 + mac_list_1_G5]
#guardamos el dataframe en un nuevo csv llamado csv_filtered
df.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/muestras3/muestras3_' + telefonos[telefono].replace(" ", "") + '.csv', index=False)


