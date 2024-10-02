import pandas as pd


#########Con este codigo pretendemos eliminar las medidas de Diogo, quedarnos con las bandas deseadas########
##############Tambien haremos el filtrado del movil que queremos estudiar#######################################

#directorio del csv inicial
csvdir = 'C:/Users/jaume/Documents/VISUAL CODES/TFG/datasetlast.csv'

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

#Nos quedamos con las filas que su batch, los primeros 4 caracteres sea superior a 5161
df['batch'] = df['batch'].astype(str).str[:4].str.replace(':', '').astype(int)
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
#EN la columna SampleNumber nos quedamos con los valores que sean mayores o iguales a 0 y menores o iguales a 199
df = df[(df['sampleNumber'] >= 0) & (df['sampleNumber'] <= 199)]
df = df[(df['y'] == 5) & (df['angle'].between(2, 29))]
#si en  una linea todos los valores de las columnas c4:41:1e:fa:07:db#1_value,c4:41:1e:fa:07:da#1_value, c4:41:1e:fa:07:d9#1_value, cc:f4:11:47:ef:eb#1_value, cc:f4:11:47:ef:e7#1_value son -100, eliminamos esa linea
#df = df[(df[["c4:41:1e:fa:07:da#1_value","c4:41:1e:fa:07:d9#1_value","cc:f4:11:47:ef:eb#1_value","cc:f4:11:47:ef:e7#1_value"]] != -100).all(axis=1)]

#Si en un mismo y y angle hay dos SampleNumber iguales, nos quedamos con el primero
df = df.drop_duplicates(subset=['y','angle','sampleNumber'], keep='first')

#ordenamos por y luego por angle y luego por SampleNumber
df = df.sort_values(by=['y','angle','sampleNumber'])

#lista macs
mac_list_1_L5DFS = ["c4:41:1e:fa:07:db#1_value","c4:41:1e:fa:07:db#1_error","c4:41:1e:fa:07:db#1_samples_averaged"]
mac_list_1_L5indoor = ["c4:41:1e:fa:07:da#1_value","c4:41:1e:fa:07:da#1_error","c4:41:1e:fa:07:da#1_samples_averaged"]
mac_list_1_L2_4 = ["c4:41:1e:fa:07:d9#1_value","c4:41:1e:fa:07:d9#1_error","c4:41:1e:fa:07:d9#1_samples_averaged"]
mac_list_1_G2_4 = ["cc:f4:11:47:ef:eb#1_value","cc:f4:11:47:ef:eb#1_error","cc:f4:11:47:ef:eb#1_samples_averaged"]
mac_list_1_G5 = ["cc:f4:11:47:ef:e7#1_value","cc:f4:11:47:ef:e7#1_error","cc:f4:11:47:ef:e7#1_samples_averaged"]

#nos quedamos con las columnas de las macs y  batch	x	y	z	brand	model	angle	sampleNumber
df = df[["batch","x","y","z","brand","model","angle","sampleNumber", "date"] + mac_list_1_L5DFS + mac_list_1_L5indoor + mac_list_1_L2_4 + mac_list_1_G2_4 + mac_list_1_G5]

#guardamos el dataframe en un nuevo csv llamado csv_filtered
df.to_csv('C:/Users/jaume/Documents/VISUAL CODES/TFG/muestrasTwoSided/muestras2TwoSided_' + telefonos[telefono].replace(" ", "") + '.csv', index=False)
