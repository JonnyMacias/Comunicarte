import base64
import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from services import procesamiento 
from services import carpetas
from models import palabrasImp
from fastapi import FastAPI, HTTPException
import os
from pathlib import Path
from typing import Optional
#from services import llamar

bp = Blueprint('ejemplo', __name__, url_prefix='/api')
BASE_DIR = 'resources/static/IMG/Palabras/'  # Cambia esta ruta
ALLOWED_EXTENSIONS = {
    'video': {'.mp4'},
    'image': {'.jpg', '.jpeg', '.png', '.gif'}
}

@bp.route('/proc_img', methods=['POST'])
def proc_img():
    data = request.json
    if len(data) == 0:
        return jsonify({'error': 'no se envio la imagen'})
    img = []
    for imagen in data['imagenes']:

        imagenDta = imagen.split(',')[1]
        if len(imagenDta) > 1:
            decode = base64.b64decode(imagenDta)

            nparr = np.frombuffer(decode, np.uint8)
            img.append(cv2.imdecode(nparr, cv2.IMREAD_COLOR))
    prediccion = procesamiento.llamar(img)
    
    return jsonify({"mensaje": prediccion})

@bp.route('/getCarpetas', methods=['GET'])
def carpeta():
    nombre_carpeta = request.args.get('carpeta')
    if not nombre_carpeta:
        return jsonify({"error": "Parámetro 'carpeta' requerido"}), 400
    return jsonify({"carpeta": carpetas.GetListaCarpetas(nombre_carpeta)})
    

@bp.route('/getPalabra', methods = ['GET'])
def getPalabra():
    return jsonify({"palabra":palabrasImp.consultaAleatoria()})


@bp.route('/getContenido', methods=['GET'])
def getContenido():
    nombre = request.args.get('nombre')
    print(nombre)
    if not nombre:
        return jsonify({"error": "El parámetro 'nombre' es requerido"}), 400

    try:
        for root, _, files in os.walk(BASE_DIR):
            for file in files:
                file_lower = file.lower()
                nombre_lower = nombre.lower()

                if nombre_lower in file_lower:
                    file_path = Path(root) / file
                    extension = file_path.suffix.lower()

                    tipo = None
                    if extension in ALLOWED_EXTENSIONS['video']:
                        tipo = 'video'
                    elif extension in ALLOWED_EXTENSIONS['image']:
                        tipo = 'imagen'

                    if tipo:
                        relative_path = str(file_path.relative_to(BASE_DIR)).replace('\\', '/')
                        return jsonify({
                            "tipo": tipo,
                            "url": f"/../IMG/Palabras/{relative_path}",
                            "nombre": file_path.stem,
                            "ruta_completa": str(file_path)
                        })

        return jsonify({
            "error": f"No se encontró contenido multimedia para '{nombre}'"
        }), 404

    except Exception as e:
        return jsonify({
            "error": f"Error en la búsqueda: {str(e)}"
        }), 500