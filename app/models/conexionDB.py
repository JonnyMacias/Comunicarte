from pymongo import MongoClient

class Conexion:
    def __init__(self, col):
        self.cliente = MongoClient('mongodb://localhost:27017')
        self.db = self.cliente['lsm']
        self.coleccion = self.db[col]
    
    def getColeccion(self):
        return self.coleccion
