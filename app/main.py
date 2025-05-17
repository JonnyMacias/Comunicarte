from flask import Flask, send_from_directory
from flask_cors import CORS
import sys
import os
import threading
import webbrowser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'resources', 'static')

def crearApp():
    app = Flask(__name__)
    CORS(app)

    # Ruta para servir el index.html
    @app.route('/')
    def serve_index():
        return send_from_directory(STATIC_DIR, 'index.html')

    # Servir archivos estáticos (JS, CSS, imágenes, etc.)
    @app.route('/<path:path>')
    def serve_static_files(path):
        return send_from_directory(STATIC_DIR, path)

    from app.routes import controllerSenales
    app.register_blueprint(controllerSenales.bp)

    return app

if __name__ == "__main__":
    crearApp().run(debug = True)       