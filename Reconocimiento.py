import face_recognition
import cv2
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
from flask_cors import CORS
from datetime import datetime
import requests

import json

app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# Definimos el servidor donde se encuentra la bd de todas las imagenes registradas
Servidor = 'E:/laragon/www/SGA_2020/SGA_BACKEND/public/imagenes/usuarios/'

# Definimos el numero de camara a utilizar para el reconocimiento default=0
numero_camara = 0
# Clase Rostro que es la que llama la API
class Rostro(Resource):
    # Se define el metodo para poder obtener una respuesta
    def get(self):
        # En la siguiente variable comenzamos un timer para calcular el rendimiento
        start_time = datetime.now()
        # Obtenemos todos los usuarios registrados en la BD
        r = requests.get('http://127.0.0.1:8000/api/usuarios')
        r.status_code
        # Lo convertimos a formato json
        json_body = r.json()
        # Capturamos una imagen coa ayuda de la libreria openCV
        cap = cv2.VideoCapture(numero_camara, cv2.CAP_DSHOW)
        leido, img = cap.read()
        # Creamos una imagen temporal
        cv2.imwrite("./rostro.jpg", img)
        # Creamos el path para face_recongnition
        img_path = "rostro.jpg"
        # Cargamos la imagen que vamos a capturamos
        cara_desconocida = face_recognition.load_image_file(img_path)
        # Obtenemos los locations para saber si detecto un rostro
        face_location_desconocido = face_recognition.face_locations(cara_desconocida)
        print(len(face_location_desconocido))
        # Verificacion para saber si reconocio algun rostro en la imagen
        if len(face_location_desconocido) > 0:
            # Encoding de la imagen donde reconocio un rostro
            face_encoding_desconocido = face_recognition.face_encodings(cara_desconocida, face_location_desconocido)[0]
            ################################################
            # Con ayuda del ciclo for se recorre todos los resultados obtenidos por la API del SGA
            for usuario in json_body:
                # Obtenemos solo la foto de usuario
                foto_usuario = usuario['foto_usuario']
                # Cargamos la imagen que se obtiene desde el servidor
                image = face_recognition.load_image_file(Servidor+foto_usuario)
                # Encoding de la imagen que se cargo desde el servidor
                face_enconding = face_recognition.face_encodings(image)[0]
                # Comparacion de rostros
                Resultado = face_recognition.compare_faces([face_enconding], face_encoding_desconocido)
                # Verificacion si hay coincidencia con las alguna de las imagenes del servidor
                if Resultado[0] == True:
                    # Devolvemos los siguientes campos: codigo_usuario y id_usuario
                    resultado_deb = {'codigo_usuario': str(usuario['codigo_usuario'])}
                    break  # terminamos el programa para que no siga buscando coincidencias
                else:
                    resultado_deb = {'error': 'Persona no encontrada en db'}
            print('Tiempo de Procesamiento: {}'.format(datetime.now() - start_time))
            return jsonify(resultado_deb)
        # Este es el caso cuando no se reconoce ningun rostro en la imagen
        else:
            return {'error': 'No hay un rostro para reconocer'}


api.add_resource(Rostro, '/identificar')  # Ruta Principal

if __name__ == '__main__':
    app.run(port='5004')
