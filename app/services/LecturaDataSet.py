import os

class LecturaDataSet:

    def __init__(self):
        pass

    def extraccionVideo(self, ruta):
        rutas = []
        for carpeta in os.listdir(ruta):
            ruta_Carpeta = os.path.join(ruta, carpeta)
            if os.path.isdir(ruta_Carpeta):
                for archivo in os.listdir(ruta_Carpeta):
                    if archivo.endswith(".avi"):
                        ruta_archivo = os.path.join(ruta_Carpeta, archivo)
                        rutas.append([ruta_archivo, carpeta])
        return rutas
    
    def extraccionImagenes(self, ruta):
        rutas = []
        for carpeta in os.listdir(ruta):
            ruta_Carpeta = os.path.join(ruta, carpeta)
            if os.path.isdir(ruta_Carpeta):
                for archivo in os.listdir(ruta_Carpeta):
                    if archivo.endswith(".jpg"):
                        ruta_archivo = os.path.join(ruta_Carpeta, archivo)
                        rutas.append([ruta_archivo, carpeta])
        return rutas
        