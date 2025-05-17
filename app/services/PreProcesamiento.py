import math
import numpy as np
from scipy.fft import fft
class PreProcesamiento:
    json_P = {
        "id": 1,
        "datos_brazos": {
            "Brazo Derecho": {
                "Brazo": 0,
                "Antebrazo": 0
            },
            "Brazo Izquierdo": {
                "Brazo": 0,
                "Antebrazo": 0
            },
            "Mano Derecha": {
                "0_1": 0,
                "1_2": 0,
                "2_3": 0,
                "3_4": 0,
                "0_5": 0,
                "5_6": 0,
                "6_7": 0,
                "7_8": 0,
                "5_9": 0,
                "9_10": 0,
                "10_11": 0,
                "11_12": 0,
                "9_13": 0,
                "13_14": 0,
                "14_15": 0,
                "15_16": 0,
                "13_17": 0,
                "0_17": 0,
                "17_18": 0,
                "18_19": 0,
                "19_20": 0
            },
            "Mano Izquierda": {
                "0_1": 0,
                "1_2": 0,
                "2_3": 0,
                "3_4": 0,
                "0_5": 0,
                "5_6": 0,
                "6_7": 0,
                "7_8": 0,
                "5_9": 0,
                "9_10": 0,
                "10_11": 0,
                "11_12": 0,
                "9_13": 0,
                "13_14": 0,
                "14_15": 0,
                "15_16": 0,
                "13_17": 0,
                "0_17": 0,
                "17_18": 0,
                "18_19": 0,
                "19_20": 0
            },
            "variable_Derecha": {
                "pulgar": 0,
                "indice": 0,
                "medio": 0,
                "anular": 0,
                "menique": 0,
                "0_X": 0,
                "0_Y": 0,
                "4_X": 0,
                "4_Y": 0,
                "8_X": 0,
                "8_Y": 0,
                "12_X": 0,
                "12_Y": 0,
                "16_X": 0,
                "16_Y": 0,
                "20_X": 0,
                "20_Y": 0
            },
            "variable_Izquierda": {
                "pulgar": 0,
                "indice": 0,
                "medio": 0,
                "anular": 0,
                "menique": 0,
                "0_X": 0,
                "0_Y": 0,
                "4_X": 0,
                "4_Y": 0,
                "8_X": 0,
                "8_Y": 0,
                "12_X": 0,
                "12_Y": 0,
                "16_X": 0,
                "16_Y": 0,
                "20_X": 0,
                "20_Y": 0
            }
        }
    }
    def __init__(self):
        pass

    def movimiento(self,json_A):
        movD = []
        movI = []
        datos_A = json_A["datos_brazos"]
        datos_P = self.json_P["datos_brazos"]

        for mano in ["Mano Derecha"]:
            claves_ordenadas = sorted(datos_A[mano].keys())
            for clave in claves_ordenadas:
                #print(str(datos_A[mano][clave]) + " - " + str(datos_P[mano][clave]))
                movD.append(datos_A[mano][clave] - datos_P[mano][clave])

        for mano in ["Mano Izquierda"]:
            claves_ordenadas = sorted(datos_A[mano].keys())
            for clave in claves_ordenadas:
                movI.append(datos_A[mano][clave] - datos_P[mano][clave])

        for d in movD:
            if(abs(d) > 30):
                return "Hay movimiento: " + str(d)
        for i in movI:
            if(abs(i) > 30):
                return "Hay movimiento: " + str(i)
        return "No hay movimiento"

    def m(self, punto1, punto2): #Calcula la pendiente
        x1, y1 = punto1
        x2, y2 = punto2
        if x2 - x1 == 0:  # Evitar división por cero
            return float("inf")  # Pendiente infinita (línea vertical)
        return (y2 - y1) / (x2 - x1)
    def angulo(self, m1, m2):
        return math.atan((m2-m1)/(1 + (m1 * m2)))
    def velocidad(self, valActual, valAnterior):
        return valActual - valAnterior
    def magnitud (self, velX, velY):
        return abs(math.sqrt((velX ** 2)+(velY ** 2)))
    def movAngulo(self, x1, y1, x2, y2):
        a = np.array([x1, y1]) # A va corresponder al frame anterior
        b = np.array([x2, y2]) # B va corresponder al frame actual
        c = np.array([x2 + 10, y2]) # C va ser solo el complemento para poder sacar el angulo

        ab = b - a
        cb = b - c

        pPunto = np.dot(ab, cb)
        norm = np.linalg.norm(ab) * np.linalg.norm(cb)
        ang_rad = np.arccos(pPunto / norm)
        ang_deg = np.degrees(ang_rad)
        return ang_deg
    def pendiente(self, results, frame_data):
        # Postura (brazos)
                if results.pose_landmarks:
                    frame_data["datos_brazos"]["Brazo Derecho"] = {
                    "Brazo": self.m((results.pose_landmarks.landmark[11].x,results.pose_landmarks.landmark[11].y,),(results.pose_landmarks.landmark[13].x,results.pose_landmarks.landmark[13].y,),),
                    "Antebrazo": self.m((results.pose_landmarks.landmark[13].x,results.pose_landmarks.landmark[13].y,),(results.pose_landmarks.landmark[15].x,results.pose_landmarks.landmark[15].y,),),
                }
                    frame_data["datos_brazos"]["Brazo Izquierdo"] = {
                    "Brazo": self.m((results.pose_landmarks.landmark[12].x,results.pose_landmarks.landmark[12].y,),(results.pose_landmarks.landmark[14].x,results.pose_landmarks.landmark[14].y,),),
                    "Antebrazo": self.m((results.pose_landmarks.landmark[14].x,results.pose_landmarks.landmark[14].y,),(results.pose_landmarks.landmark[16].x,results.pose_landmarks.landmark[16].y,),),
                }
                else:
                    frame_data["datos_brazos"]["Brazo Derecho"] = {
                    "Brazo": 0,
                    "Antebrazo": 0,
                }
                    frame_data["datos_brazos"]["Brazo Izquierdo"] = {
                    "Brazo": 0,
                    "Antebrazo": 0,
                }

            # Mano derecha
                if results.right_hand_landmarks:
                    #===========================DESPLAZAMIENTO===========================================================
                    """puntosDer.append([results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[4].x, results.right_hand_landmarks.landmark[4].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[12].x, results.right_hand_landmarks.landmark[12].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[16].x, results.right_hand_landmarks.landmark[16].y])
                    puntosDer.append([results.right_hand_landmarks.landmark[20].x, results.right_hand_landmarks.landmark[20].y])"""
                    frame_data["datos_brazos"]["Mano Derecha"] = {
                    "0_1": self.m((results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y,),(results.right_hand_landmarks.landmark[1].x, results.right_hand_landmarks.landmark[1].y,),),
                    "1_2": self.m((results.right_hand_landmarks.landmark[1].x,results.right_hand_landmarks.landmark[1].y,),(results.right_hand_landmarks.landmark[2].x,results.right_hand_landmarks.landmark[2].y,),),
                    "2_3": self.m((results.right_hand_landmarks.landmark[2].x, results.right_hand_landmarks.landmark[2].y,), ( results.right_hand_landmarks.landmark[3].x, results.right_hand_landmarks.landmark[3].y, ),),
                    "3_4": self.m((results.right_hand_landmarks.landmark[3].x, results.right_hand_landmarks.landmark[3].y,),(results.right_hand_landmarks.landmark[4].x,results.right_hand_landmarks.landmark[4].y,),),
                    "0_5": self.m( (results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y,),(results.right_hand_landmarks.landmark[5].x, results.right_hand_landmarks.landmark[5].y, ),),
                    "5_6": self.m((results.right_hand_landmarks.landmark[5].x,results.right_hand_landmarks.landmark[5].y,),(results.right_hand_landmarks.landmark[6].x,results.right_hand_landmarks.landmark[6].y,),),
                    "6_7": self.m((results.right_hand_landmarks.landmark[6].x,results.right_hand_landmarks.landmark[6].y, ),(results.right_hand_landmarks.landmark[7].x, results.right_hand_landmarks.landmark[7].y,),),
                    "7_8": self.m((results.right_hand_landmarks.landmark[7].x,results.right_hand_landmarks.landmark[7].y,), (results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y,),),
                    "5_9": self.m((results.right_hand_landmarks.landmark[5].x,results.right_hand_landmarks.landmark[5].y,),(results.right_hand_landmarks.landmark[9].x,results.right_hand_landmarks.landmark[9].y,),),
                    "9_10": self.m((results.right_hand_landmarks.landmark[9].x,results.right_hand_landmarks.landmark[9].y,),(results.right_hand_landmarks.landmark[10].x,results.right_hand_landmarks.landmark[10].y,),),
                    "10_11": self.m((results.right_hand_landmarks.landmark[10].x,results.right_hand_landmarks.landmark[10].y,), (results.right_hand_landmarks.landmark[11].x,results.right_hand_landmarks.landmark[11].y,), ),
                    "11_12": self.m((results.right_hand_landmarks.landmark[11].x,results.right_hand_landmarks.landmark[11].y,),(results.right_hand_landmarks.landmark[12].x,results.right_hand_landmarks.landmark[12].y,),),
                    "9_13": self.m((results.right_hand_landmarks.landmark[9].x,results.right_hand_landmarks.landmark[9].y,),(results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y,),),
                    "13_14": self.m((results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y,),(results.right_hand_landmarks.landmark[14].x,results.right_hand_landmarks.landmark[14].y,),),
                    "14_15": self.m((results.right_hand_landmarks.landmark[14].x,results.right_hand_landmarks.landmark[14].y,),(results.right_hand_landmarks.landmark[15].x,results.right_hand_landmarks.landmark[15].y,),),
                    "15_16": self.m((results.right_hand_landmarks.landmark[15].x,results.right_hand_landmarks.landmark[15].y,),(results.right_hand_landmarks.landmark[16].x,results.right_hand_landmarks.landmark[16].y,),),
                    "13_17": self.m((results.right_hand_landmarks.landmark[13].x,results.right_hand_landmarks.landmark[13].y,),(results.right_hand_landmarks.landmark[17].x,results.right_hand_landmarks.landmark[17].y,),),
                    "0_17": self.m((results.right_hand_landmarks.landmark[0].x,results.right_hand_landmarks.landmark[0].y,),(results.right_hand_landmarks.landmark[17].x,results.right_hand_landmarks.landmark[17].y,),),
                    "17_18": self.m((results.right_hand_landmarks.landmark[17].x,results.right_hand_landmarks.landmark[17].y,),(results.right_hand_landmarks.landmark[18].x,results.right_hand_landmarks.landmark[18].y,),),
                    "18_19": self.m((results.right_hand_landmarks.landmark[18].x,results.right_hand_landmarks.landmark[18].y,),(results.right_hand_landmarks.landmark[19].x, results.right_hand_landmarks.landmark[19].y,),),
                    "19_20": self.m((results.right_hand_landmarks.landmark[19].x,results.right_hand_landmarks.landmark[19].y,),(results.right_hand_landmarks.landmark[20].x,results.right_hand_landmarks.landmark[20].y,),),
                }

                else:
                    puntosDer = [
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0]
                        
                    ]
                    frame_data["datos_brazos"]["Mano Derecha"] = {
                    "0_1": 0,
                    "1_2": 0,
                    "2_3": 0,
                    "3_4": 0,
                    "0_5": 0,
                    "5_6": 0,
                    "6_7": 0,
                    "7_8": 0,
                    "5_9": 0,
                    "9_10": 0,
                    "10_11": 0,
                    "11_12": 0,
                    "9_13": 0,
                    "13_14": 0,
                    "14_15": 0,
                    "15_16": 0,
                    "13_17": 0,
                    "0_17": 0,
                    "17_18": 0,
                    "18_19": 0,
                    "19_20": 0,
                }

            # Mano izquierda
                if results.left_hand_landmarks:
                    #==============================DESPLAZAMIENTO=================================
                    """puntosIzq.append([results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[4].x,results.left_hand_landmarks.landmark[4].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[8].x,results.left_hand_landmarks.landmark[8].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[12].x,results.left_hand_landmarks.landmark[12].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[16].x,results.left_hand_landmarks.landmark[16].y])
                    puntosIzq.append([results.left_hand_landmarks.landmark[20].x,results.left_hand_landmarks.landmark[20].y])"""

                    frame_data["datos_brazos"]["Mano Izquierda"] = {
                    "0_1": self.m((results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y,),(results.left_hand_landmarks.landmark[1].x,results.left_hand_landmarks.landmark[1].y,),),
                    "1_2": self.m((results.left_hand_landmarks.landmark[1].x,results.left_hand_landmarks.landmark[1].y,),(results.left_hand_landmarks.landmark[2].x,results.left_hand_landmarks.landmark[2].y,),),
                    "2_3": self.m((results.left_hand_landmarks.landmark[2].x,results.left_hand_landmarks.landmark[2].y,),(results.left_hand_landmarks.landmark[3].x,results.left_hand_landmarks.landmark[3].y,),),
                    "3_4": self.m((results.left_hand_landmarks.landmark[3].x,results.left_hand_landmarks.landmark[3].y,),(results.left_hand_landmarks.landmark[4].x,results.left_hand_landmarks.landmark[4].y,),),
                    "0_5": self.m((results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y,),(results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y,),),
                    "5_6": self.m((results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y,),(results.left_hand_landmarks.landmark[6].x,results.left_hand_landmarks.landmark[6].y,),),
                    "6_7": self.m((results.left_hand_landmarks.landmark[6].x,results.left_hand_landmarks.landmark[6].y,),(results.left_hand_landmarks.landmark[7].x,results.left_hand_landmarks.landmark[7].y,),),
                    "7_8": self.m((results.left_hand_landmarks.landmark[7].x,results.left_hand_landmarks.landmark[7].y,),(results.left_hand_landmarks.landmark[8].x,results.left_hand_landmarks.landmark[8].y,),),
                    "5_9": self.m((results.left_hand_landmarks.landmark[5].x,results.left_hand_landmarks.landmark[5].y,),(results.left_hand_landmarks.landmark[9].x,results.left_hand_landmarks.landmark[9].y,),),
                    "9_10": self.m((results.left_hand_landmarks.landmark[9].x,results.left_hand_landmarks.landmark[9].y,),(results.left_hand_landmarks.landmark[10].x,results.left_hand_landmarks.landmark[10].y,),),
                    "10_11": self.m((results.left_hand_landmarks.landmark[10].x,results.left_hand_landmarks.landmark[10].y,),(results.left_hand_landmarks.landmark[11].x,results.left_hand_landmarks.landmark[11].y,),),
                    "11_12": self.m((results.left_hand_landmarks.landmark[11].x,results.left_hand_landmarks.landmark[11].y,),(results.left_hand_landmarks.landmark[12].x,results.left_hand_landmarks.landmark[12].y,),),
                    "9_13": self.m((results.left_hand_landmarks.landmark[9].x,results.left_hand_landmarks.landmark[9].y,),(results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y,),),
                    "13_14": self.m((results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y,),(results.left_hand_landmarks.landmark[14].x,results.left_hand_landmarks.landmark[14].y,),),
                    "14_15": self.m((results.left_hand_landmarks.landmark[14].x,results.left_hand_landmarks.landmark[14].y,),(results.left_hand_landmarks.landmark[15].x,results.left_hand_landmarks.landmark[15].y,),),
                    "15_16": self.m((results.left_hand_landmarks.landmark[15].x,results.left_hand_landmarks.landmark[15].y,),(results.left_hand_landmarks.landmark[16].x,results.left_hand_landmarks.landmark[16].y,),),
                    "13_17": self.m((results.left_hand_landmarks.landmark[13].x,results.left_hand_landmarks.landmark[13].y,),(results.left_hand_landmarks.landmark[17].x,results.left_hand_landmarks.landmark[17].y,),),
                    "0_17": self.m((results.left_hand_landmarks.landmark[0].x,results.left_hand_landmarks.landmark[0].y,),(results.left_hand_landmarks.landmark[17].x,results.left_hand_landmarks.landmark[17].y,),),
                    "17_18": self.m((results.left_hand_landmarks.landmark[17].x,results.left_hand_landmarks.landmark[17].y,),(results.left_hand_landmarks.landmark[18].x,results.left_hand_landmarks.landmark[18].y,),),
                    "18_19": self.m((results.left_hand_landmarks.landmark[18].x,results.left_hand_landmarks.landmark[18].y,),(results.left_hand_landmarks.landmark[19].x,results.left_hand_landmarks.landmark[19].y,),),
                    "19_20": self.m((results.left_hand_landmarks.landmark[19].x,results.left_hand_landmarks.landmark[19].y,),(results.left_hand_landmarks.landmark[20].x,results.left_hand_landmarks.landmark[20].y,),),
                }
                else:
                    puntosIzq = [
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0],
                        [0,0]
                        
                    ]
                    frame_data["datos_brazos"]["Mano Izquierda"] = {
                    "0_1": 0,
                    "1_2": 0,
                    "2_3": 0,
                    "3_4": 0,
                    "0_5": 0,
                    "5_6": 0,
                    "6_7": 0,
                    "7_8": 0,
                    "5_9": 0,
                    "9_10": 0,
                    "10_11": 0,
                    "11_12": 0,
                    "9_13": 0,
                    "13_14": 0,
                    "14_15": 0,
                    "15_16": 0,
                    "13_17": 0,
                    "0_17": 0,
                    "17_18": 0,
                    "18_19": 0,
                    "19_20": 0,
                }
                return frame_data
    
    def desplazamiento(self, puntos1, puntos2):
        des = []
        for i in enumerate(puntos1):
            x = puntos2[i[0]][0] - puntos1[i[0]][0]
            y = puntos2[i[0]][1] - puntos1[i[0]][1]
            des.append([x, y])
        return des

    def angulos(self, jason):
        jason ["datos_brazos"]["ang_Derecha"] = {
            "0Pulgar_1": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["2_3"], jason ["datos_brazos"]["Mano Derecha"]["3_4"]),
            "0Pulgar_2": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["1_2"], jason ["datos_brazos"]["Mano Derecha"]["2_3"]),
            "0Indice_1": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["5_6"], jason ["datos_brazos"]["Mano Derecha"]["6_7"]),
            "0Indice_2": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["5_9"], jason ["datos_brazos"]["Mano Derecha"]["5_6"]),
            "0Medio_1": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["9_10"], jason ["datos_brazos"]["Mano Derecha"]["10_11"]),
            "0Medio_2": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["9_13"], jason ["datos_brazos"]["Mano Derecha"]["10_11"]),
            "0Anular_1": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["13_14"], jason ["datos_brazos"]["Mano Derecha"]["14_15"]),
            "0Anular_2": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["13_17"], jason ["datos_brazos"]["Mano Derecha"]["13_14"]),
            "0Menique_1": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["17_18"], jason ["datos_brazos"]["Mano Derecha"]["18_19"]),
            "0Menique_2": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["13_17"], jason ["datos_brazos"]["Mano Derecha"]["17_18"]),
            "0Menique_3": self.angulo(jason ["datos_brazos"]["Mano Derecha"]["0_17"], jason ["datos_brazos"]["Mano Derecha"]["17_18"]),
        }

        jason ["datos_brazos"]["ang_Izquierda"] = {
            "0Pulgar_1": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["2_3"], jason ["datos_brazos"]["Mano Izquierda"]["3_4"]),
            "0Pulgar_2": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["1_2"], jason ["datos_brazos"]["Mano Izquierda"]["2_3"]),
            "0Indice_1": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["5_6"], jason ["datos_brazos"]["Mano Izquierda"]["6_7"]),
            "0Indice_2": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["5_9"], jason ["datos_brazos"]["Mano Izquierda"]["5_6"]),
            "0Medio_1": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["9_10"], jason ["datos_brazos"]["Mano Izquierda"]["10_11"]),
            "0Medio_2": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["9_13"], jason ["datos_brazos"]["Mano Izquierda"]["10_11"]),
            "0Anular_1": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["13_14"], jason ["datos_brazos"]["Mano Izquierda"]["14_15"]),
            "0Anular_2": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["13_17"], jason ["datos_brazos"]["Mano Izquierda"]["13_14"]),
            "0Menique_1": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["17_18"], jason ["datos_brazos"]["Mano Izquierda"]["18_19"]),
            "0Menique_2": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["13_17"], jason ["datos_brazos"]["Mano Izquierda"]["17_18"]),
            "0Menique_3": self.angulo(jason ["datos_brazos"]["Mano Izquierda"]["0_17"], jason ["datos_brazos"]["Mano Izquierda"]["17_18"]),
        
        }


        return jason
    
    def velocidad_magnitud(self, json, puntos, puntosTemp):
        if puntosTemp:

            if puntos.right_hand_landmarks and puntosTemp.right_hand_landmarks:
                json["datos_brazos"]["vel_Derecha"] = {
                    "velMunecaX": self.velocidad(puntos.right_hand_landmarks.landmark[0].x, puntosTemp.right_hand_landmarks.landmark[0].x),
                    "velMunecaY": self.velocidad(puntos.right_hand_landmarks.landmark[0].y, puntosTemp.right_hand_landmarks.landmark[0].y),
                    "magMuneca" : self.magnitud(self.velocidad(puntos.right_hand_landmarks.landmark[0].x, puntosTemp.right_hand_landmarks.landmark[0].x), self.velocidad(puntos.right_hand_landmarks.landmark[0].y, puntosTemp.right_hand_landmarks.landmark[0].y)),
                    
                    "velPulgarX": self.velocidad(puntos.right_hand_landmarks.landmark[4].x, puntosTemp.right_hand_landmarks.landmark[4].x),
                    "velPulgarY": self.velocidad(puntos.right_hand_landmarks.landmark[4].y, puntosTemp.right_hand_landmarks.landmark[4].y),
                    "magPulgar" : self.magnitud(self.velocidad(puntos.right_hand_landmarks.landmark[4].x, puntosTemp.right_hand_landmarks.landmark[4].x), self.velocidad(puntos.right_hand_landmarks.landmark[4].y, puntosTemp.right_hand_landmarks.landmark[4].y)),

                    "velIndiceX": self.velocidad(puntos.right_hand_landmarks.landmark[8].x, puntosTemp.right_hand_landmarks.landmark[8].x),
                    "velIndiceY": self.velocidad(puntos.right_hand_landmarks.landmark[8].y, puntosTemp.right_hand_landmarks.landmark[8].y),
                    "magIndice" : self.magnitud(self.velocidad(puntos.right_hand_landmarks.landmark[8].x, puntosTemp.right_hand_landmarks.landmark[8].x), self.velocidad(puntos.right_hand_landmarks.landmark[8].y, puntosTemp.right_hand_landmarks.landmark[8].y)),

                    "velMedioX": self.velocidad(puntos.right_hand_landmarks.landmark[12].x, puntosTemp.right_hand_landmarks.landmark[12].x),
                    "velMedioY": self.velocidad(puntos.right_hand_landmarks.landmark[12].y, puntosTemp.right_hand_landmarks.landmark[12].y),
                    "magMedio" : self.magnitud(self.velocidad(puntos.right_hand_landmarks.landmark[12].x, puntosTemp.right_hand_landmarks.landmark[12].x), self.velocidad(puntos.right_hand_landmarks.landmark[12].y, puntosTemp.right_hand_landmarks.landmark[12].y)),

                    "velAnularX": self.velocidad(puntos.right_hand_landmarks.landmark[16].x, puntosTemp.right_hand_landmarks.landmark[16].x),
                    "velAnularY": self.velocidad(puntos.right_hand_landmarks.landmark[16].y, puntosTemp.right_hand_landmarks.landmark[16].y),
                    "magAnular" : self.magnitud(self.velocidad(puntos.right_hand_landmarks.landmark[16].x, puntosTemp.right_hand_landmarks.landmark[16].x), self.velocidad(puntos.right_hand_landmarks.landmark[16].y, puntosTemp.right_hand_landmarks.landmark[16].y)),

                    "velMeniqueX": self.velocidad(puntos.right_hand_landmarks.landmark[20].x, puntosTemp.right_hand_landmarks.landmark[20].x),
                    "velMeniqueY": self.velocidad(puntos.right_hand_landmarks.landmark[20].y, puntosTemp.right_hand_landmarks.landmark[20].y),
                    "magMenique" : self.magnitud(self.velocidad(puntos.right_hand_landmarks.landmark[20].x, puntosTemp.right_hand_landmarks.landmark[20].x), self.velocidad(puntos.right_hand_landmarks.landmark[20].y, puntosTemp.right_hand_landmarks.landmark[20].y))
                }
            else:
                json["datos_brazos"]["vel_Derecha"] = {
                    "velMunecaX": 0,
                    "velMunecaY": 0,
                    "magMuneca" : 0,
                    
                    "velPulgarX": 0,
                    "velPulgarY": 0,
                    "magPulgar" : 0,

                    "velIndiceX": 0,
                    "velIndiceY": 0,
                    "magIndice" : 0,

                    "velMedioX": 0,
                    "velMedioY": 0,
                    "magMedio" : 0,

                    "velAnularX": 0,
                    "velAnularY": 0,
                    "magAnular" : 0,

                    "velMeniqueX": 0,
                    "velMeniqueY": 0,
                    "magMenique" : 0,
                }
            

            if puntos.left_hand_landmarks and puntosTemp.left_hand_landmarks:
                json["datos_brazos"]["vel_Izquierdo"] = {
                    "velMunecaX": self.velocidad(puntos.left_hand_landmarks.landmark[0].x, puntosTemp.left_hand_landmarks.landmark[0].x),
                    "velMunecaY": self.velocidad(puntos.left_hand_landmarks.landmark[0].y, puntosTemp.left_hand_landmarks.landmark[0].y),
                    "magMuneca" : self.magnitud(self.velocidad(puntos.left_hand_landmarks.landmark[0].x, puntosTemp.left_hand_landmarks.landmark[0].x), self.velocidad(puntos.left_hand_landmarks.landmark[0].y, puntosTemp.left_hand_landmarks.landmark[0].y)),
                    
                    "velPulgarX": self.velocidad(puntos.left_hand_landmarks.landmark[4].x, puntosTemp.left_hand_landmarks.landmark[4].x),
                    "velPulgarY": self.velocidad(puntos.left_hand_landmarks.landmark[4].y, puntosTemp.left_hand_landmarks.landmark[4].y),
                    "magPulgar" : self.magnitud(self.velocidad(puntos.left_hand_landmarks.landmark[4].x, puntosTemp.left_hand_landmarks.landmark[4].x), self.velocidad(puntos.left_hand_landmarks.landmark[4].y, puntosTemp.left_hand_landmarks.landmark[4].y)),

                    "velIndiceX": self.velocidad(puntos.left_hand_landmarks.landmark[8].x, puntosTemp.left_hand_landmarks.landmark[8].x),
                    "velIndiceY": self.velocidad(puntos.left_hand_landmarks.landmark[8].y, puntosTemp.left_hand_landmarks.landmark[8].y),
                    "magIndice" : self.magnitud(self.velocidad(puntos.left_hand_landmarks.landmark[8].x, puntosTemp.left_hand_landmarks.landmark[8].x), self.velocidad(puntos.left_hand_landmarks.landmark[8].y, puntosTemp.left_hand_landmarks.landmark[8].y)),

                    "velMedioX": self.velocidad(puntos.left_hand_landmarks.landmark[12].x, puntosTemp.left_hand_landmarks.landmark[12].x),
                    "velMedioY": self.velocidad(puntos.left_hand_landmarks.landmark[12].y, puntosTemp.left_hand_landmarks.landmark[12].y),
                    "magMedio" : self.magnitud(self.velocidad(puntos.left_hand_landmarks.landmark[12].x, puntosTemp.left_hand_landmarks.landmark[12].x), self.velocidad(puntos.left_hand_landmarks.landmark[12].y, puntosTemp.left_hand_landmarks.landmark[12].y)),

                    "velAnularX": self.velocidad(puntos.left_hand_landmarks.landmark[16].x, puntosTemp.left_hand_landmarks.landmark[16].x),
                    "velAnularY": self.velocidad(puntos.left_hand_landmarks.landmark[16].y, puntosTemp.left_hand_landmarks.landmark[16].y),
                    "magAnular" : self.magnitud(self.velocidad(puntos.left_hand_landmarks.landmark[16].x, puntosTemp.left_hand_landmarks.landmark[16].x), self.velocidad(puntos.left_hand_landmarks.landmark[16].y, puntosTemp.left_hand_landmarks.landmark[16].y)),

                    "velMeniqueX": self.velocidad(puntos.left_hand_landmarks.landmark[20].x, puntosTemp.left_hand_landmarks.landmark[20].x),
                    "velMeniqueY": self.velocidad(puntos.left_hand_landmarks.landmark[20].y, puntosTemp.left_hand_landmarks.landmark[20].y),
                    "magMenique" : self.magnitud(self.velocidad(puntos.left_hand_landmarks.landmark[20].x, puntosTemp.left_hand_landmarks.landmark[20].x), self.velocidad(puntos.left_hand_landmarks.landmark[20].y, puntosTemp.left_hand_landmarks.landmark[20].y))
                }
            else:
                json["datos_brazos"]["vel_Izquierdo"] = {
                    "velMunecaX": 0,
                    "velMunecaY": 0,
                    "magMuneca" : 0,
                    
                    "velPulgarX": 0,
                    "velPulgarY": 0,
                    "magPulgar" : 0,

                    "velIndiceX": 0,
                    "velIndiceY": 0,
                    "magIndice" : 0,

                    "velMedioX": 0,
                    "velMedioY": 0,
                    "magMedio" : 0,

                    "velAnularX": 0,
                    "velAnularY": 0,
                    "magAnular" : 0,

                    "velMeniqueX": 0,
                    "velMeniqueY": 0,
                    "magMenique" : 0,
                }
        else:
            json["datos_brazos"]["vel_Derecha"] = {
                    "velMunecaX": 0,
                    "velMunecaY": 0,
                    "magMuneca" : 0,
                    
                    "velPulgarX": 0,
                    "velPulgarY": 0,
                    "magPulgar" : 0,

                    "velIndiceX": 0,
                    "velIndiceY": 0,
                    "magIndice" : 0,

                    "velMedioX": 0,
                    "velMedioY": 0,
                    "magMedio" : 0,

                    "velAnularX": 0,
                    "velAnularY": 0,
                    "magAnular" : 0,

                    "velMeniqueX": 0,
                    "velMeniqueY": 0,
                    "magMenique" : 0,
                }
            
            json["datos_brazos"]["vel_Izquierdo"] = {
                    "velMunecaX": 0,
                    "velMunecaY": 0,
                    "magMuneca" : 0,
                    
                    "velPulgarX": 0,
                    "velPulgarY": 0,
                    "magPulgar" : 0,

                    "velIndiceX": 0,
                    "velIndiceY": 0,
                    "magIndice" : 0,

                    "velMedioX": 0,
                    "velMedioY": 0,
                    "magMedio" : 0,

                    "velAnularX": 0,
                    "velAnularY": 0,
                    "magAnular" : 0,

                    "velMeniqueX": 0,
                    "velMeniqueY": 0,
                    "magMenique" : 0,
                }




        return json
    
    def movimiento_angulos(self, json, jsonTemp):
        if jsonTemp:
            json ["datos_brazos"]["mAng_Derecha"] = {
                "0Pulgar_1": math.degrees(json["datos_brazos"]["ang_Derecha"]["0Pulgar_1"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Pulgar_1"]),
                "0Pulgar_2": math.degrees(json["datos_brazos"]["ang_Derecha"]["0Pulgar_2"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Pulgar_2"]),
                "0Indice_1": math.degrees(json["datos_brazos"]["ang_Derecha"]["0Indice_1"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Indice_1"]),
                "0Indice_2": math.degrees(json["datos_brazos"]["ang_Derecha"]["0Indice_2"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Indice_2"]),
                "0Medio_1":  math.degrees(json["datos_brazos"]["ang_Derecha"]["0Medio_1"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Medio_1"]),
                "0Medio_2":  math.degrees(json["datos_brazos"]["ang_Derecha"]["0Medio_2"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Medio_2"]),
                "0Anular_1": math.degrees(json["datos_brazos"]["ang_Derecha"]["0Anular_1"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Anular_1"]),
                "0Anular_2": math.degrees(json["datos_brazos"]["ang_Derecha"]["0Anular_2"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Anular_2"]),
                "0Menique_1":math.degrees(json["datos_brazos"]["ang_Derecha"]["0Menique_1"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Menique_1"]),
                "0Menique_2":math.degrees(json["datos_brazos"]["ang_Derecha"]["0Menique_2"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Menique_2"]),
                "0Menique_3":math.degrees(json["datos_brazos"]["ang_Derecha"]["0Menique_3"] - jsonTemp["datos_brazos"]["ang_Derecha"]["0Menique_3"])
            }

            json ["datos_brazos"]["mAng_Izquierda"] = {
                "0Pulgar_1": math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Pulgar_1"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Pulgar_1"]),
                "0Pulgar_2": math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Pulgar_2"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Pulgar_2"]),
                "0Indice_1": math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Indice_1"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Indice_1"]),
                "0Indice_2": math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Indice_2"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Indice_2"]),
                "0Medio_1":  math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Medio_1"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Medio_1"]),
                "0Medio_2":  math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Medio_2"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Medio_2"]),
                "0Anular_1": math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Anular_1"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Anular_1"]),
                "0Anular_2": math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Anular_2"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Anular_2"]),
                "0Menique_1":math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Menique_1"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Menique_1"]),
                "0Menique_2":math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Menique_2"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Menique_2"]),
                "0Menique_3":math.degrees(json["datos_brazos"]["ang_Izquierda"]["0Menique_3"] - jsonTemp["datos_brazos"]["ang_Izquierda"]["0Menique_3"])
    
            
                }
        else:
            json ["datos_brazos"]["mAng_Derecha"] = {
            "0Pulgar_1": 0,
            "0Pulgar_2":0,
            "0Indice_1":0,
            "0Indice_2":0,
            "0Medio_1":0,
            "0Medio_2": 0,
            "0Anular_1":0,
            "0Anular_2":0,
            "0Menique_1": 0,
            "0Menique_2": 0,
            "0Menique_3":0
            }

            json ["datos_brazos"]["mAng_Izquierda"] = {
                "0Pulgar_1": 0,
                "0Pulgar_2": 0,
                "0Indice_1": 0,
                "0Indice_2": 0,
                "0Medio_1": 0,
                "0Medio_2": 0,
                "0Anular_1":0,
                "0Anular_2": 0,
                "0Menique_1":0,
                "0Menique_2": 0,
                "0Menique_3": 0,
            
                }
        
        return json
    def getDatosImg(self, frames):
        datos = []
        print("se aplicara la serie de fourier ==========================")
        keys_ManoDerecha = frames[0]["datos_brazos"]["Mano Derecha"].keys()
        keys_ManoIzquierda = frames[0]["datos_brazos"]["Mano Izquierda"].keys()
        keys_ang_Derecha = frames[0]["datos_brazos"]["ang_Derecha"].keys()
        keys_ang_Izquierda = frames[0]["datos_brazos"]["ang_Izquierda"].keys()
        """keys_vel_Derecha = frames[0]["datos_brazos"]["vel_Derecha"].keys()
        keys_vel_Izquierdo = frames[0]["datos_brazos"]["vel_Izquierdo"].keys()
        keys_mAng_Derecha = frames[0]["datos_brazos"]["mAng_Derecha"].keys()
        keys_mAng_Izquierda = frames[0]["datos_brazos"]["mAng_Izquierda"].keys()"""
        
        f_ManoDerecha = self.trans_fourire(keys_ManoDerecha, frames, "Mano Derecha")
        f_ManoIzquierda = self.trans_fourire(keys_ManoIzquierda, frames, "Mano Izquierda")
        f_ang_Derecha = self.trans_fourire(keys_ang_Derecha, frames, "ang_Derecha")
        f_ang_Izquierda = self.trans_fourire(keys_ang_Izquierda, frames, "ang_Izquierda")
        """f_vel_Derecha = self.trans_fourire(keys_vel_Derecha, frames, "vel_Derecha")
        f_vel_Izquierdo = self.trans_fourire(keys_vel_Izquierdo, frames, "vel_Izquierdo")
        f_mAng_Derecha = self.trans_fourire(keys_mAng_Derecha, frames, "mAng_Derecha")
        f_mAng_Izquierda = self.trans_fourire(keys_mAng_Izquierda, frames, "mAng_Izquierda")"""
        
        for i in f_ManoDerecha:
            datos.extend(f_ManoDerecha[i])
        for i in f_ManoIzquierda:
            datos.extend(f_ManoIzquierda[i])
        for i in f_ang_Derecha:
            datos.extend(f_ang_Derecha[i])
        for i in f_ang_Izquierda:
            datos.extend(f_ang_Izquierda[i])
        """ for i in f_vel_Derecha:
            datos.extend(f_vel_Derecha[i])
        for i in f_vel_Izquierdo:
            datos.extend(f_vel_Izquierdo[i])
        for i in f_mAng_Derecha:
            datos.extend(f_mAng_Derecha[i])
        for i in f_mAng_Izquierda:
            datos.extend(f_mAng_Izquierda[i])"""

        return datos
    
    def trans_fourire(self, keys, frames, laKey):
        fourier_features = {}
        for key in keys:
            signal = [frame["datos_brazos"][laKey][key] for frame in frames]
            fourier_result = fft(signal)
            magnitudes = np.abs(fourier_result)[:5]
            
            fourier_features[key] = magnitudes.tolist()
        return fourier_features
    
    def setJson_P(self, jason):
        self.json_P = jason
    
    def ReiniciarJson(self):
        self.json_P = {
        "id": 1,
        "datos_brazos": {
            "Brazo Derecho": {
                "Brazo": 0,
                "Antebrazo": 0
            },
            "Brazo Izquierdo": {
                "Brazo": 0,
                "Antebrazo": 0
            },
            "Mano Derecha": {
                "0_1": 0,
                "1_2": 0,
                "2_3": 0,
                "3_4": 0,
                "0_5": 0,
                "5_6": 0,
                "6_7": 0,
                "7_8": 0,
                "5_9": 0,
                "9_10": 0,
                "10_11": 0,
                "11_12": 0,
                "9_13": 0,
                "13_14": 0,
                "14_15": 0,
                "15_16": 0,
                "13_17": 0,
                "0_17": 0,
                "17_18": 0,
                "18_19": 0,
                "19_20": 0
            },
            "Mano Izquierda": {
                "0_1": 0,
                "1_2": 0,
                "2_3": 0,
                "3_4": 0,
                "0_5": 0,
                "5_6": 0,
                "6_7": 0,
                "7_8": 0,
                "5_9": 0,
                "9_10": 0,
                "10_11": 0,
                "11_12": 0,
                "9_13": 0,
                "13_14": 0,
                "14_15": 0,
                "15_16": 0,
                "13_17": 0,
                "0_17": 0,
                "17_18": 0,
                "18_19": 0,
                "19_20": 0
            },
            "variable_Derecha": {
                "pulgar": 0,
                "indice": 0,
                "medio": 0,
                "anular": 0,
                "menique": 0,
                "0_X": 0,
                "0_Y": 0,
                "4_X": 0,
                "4_Y": 0,
                "8_X": 0,
                "8_Y": 0,
                "12_X": 0,
                "12_Y": 0,
                "16_X": 0,
                "16_Y": 0,
                "20_X": 0,
                "20_Y": 0
            },
            "variable_Izquierda": {
                "pulgar": 0,
                "indice": 0,
                "medio": 0,
                "anular": 0,
                "menique": 0,
                "0_X": 0,
                "0_Y": 0,
                "4_X": 0,
                "4_Y": 0,
                "8_X": 0,
                "8_Y": 0,
                "12_X": 0,
                "12_Y": 0,
                "16_X": 0,
                "16_Y": 0,
                "20_X": 0,
                "20_Y": 0
            }
        }
    }
    
    