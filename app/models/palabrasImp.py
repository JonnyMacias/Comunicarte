from app.models.conexionDB import Conexion
import random

def consultaAleatoria():
    conexion = Conexion('palabrasDeletreo')
    doc_aleatorio = conexion.getColeccion().aggregate([ { "$sample": { "size": 1 } } ])
    doc_aleatorio = list(doc_aleatorio)[0]  # Convertir el cursor en lista y obtener el primero
    return doc_aleatorio["palabra"]
