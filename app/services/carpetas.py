import os
import sys
def listar_subcarpetas(ruta_raiz):
    """
    Lista todas las subcarpetas inmediatas dentro de la carpeta raíz dada.
    
    Args:
        ruta_raiz (str): Ruta de la carpeta raíz a analizar.
    
    Returns:
        dict: Diccionario con {nombre_carpeta: [subcarpetas]}
    """
    estructura = {}
    
    if not os.path.exists(ruta_raiz):
        print(f"La carpeta {ruta_raiz} no existe.")
        return estructura
    
    for nombre in os.listdir(ruta_raiz):
        ruta_completa = os.path.join(ruta_raiz, nombre)
        if os.path.isdir(ruta_completa):
            subcarpetas = []
            for subnombre in os.listdir(ruta_completa):
                subruta = os.path.join(ruta_completa, subnombre)
                if os.path.isdir(subruta):
                    subcarpetas.append(subnombre)
            estructura[nombre] = subcarpetas
    
    return estructura

def mostrar_estructura(estructura):
    """
    Muestra la estructura de carpetas de forma legible.
    """
    lista = []
    for carpeta, subcarpetas in estructura.items():
        lista.append(carpeta)
    return lista

def GetListaCarpetas(ruta):
    estructura = listar_subcarpetas("resources/DataSet/Palabras/"+ruta)
    return mostrar_estructura(estructura)