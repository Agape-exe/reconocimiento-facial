from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import base64
import os
from datetime import datetime
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from cv2 import imdecode, IMREAD_COLOR

app = Flask(__name__)
CORS(app)

# Conexión MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="31415",
    database="reconocimiento"
)
cursor = db.cursor()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embedder = FaceNet()

def extraer_vector(imagen_np):
    if imagen_np is None or imagen_np.size == 0:
        return None
    imagen_rgb = np.array(Image.fromarray(imagen_np).convert("RGB"))
    embeddings = embedder.embeddings([imagen_rgb])
    if embeddings is None or len(embeddings) == 0:
        return None
    return embeddings[0]

@app.route("/salud", methods=["GET"])
def salud():
    return jsonify({"mensaje": "Servidor activo"}), 200

@app.route("/registrar_usuario", methods=["POST"])
def registrar_usuario():
    data = request.get_json()
    nombre = data.get('nombre')
    apellidos = data.get('apellido')
    codigo = data.get('codigo')
    correo = data.get('email')
    requisitoriado = data.get('requisitoriado', False)
    imagen_base64 = data.get('imagen')

    if not all([nombre, apellidos, codigo, correo, imagen_base64]):
        return jsonify({"error": "Faltan campos"}), 400

    try:
        imagen_bytes = base64.b64decode(imagen_base64)
        nombre_archivo = f"{codigo}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        ruta_imagen = os.path.join(UPLOAD_FOLDER, nombre_archivo)
        with open(ruta_imagen, 'wb') as f:
            f.write(imagen_bytes)

        imagen_np = imdecode(np.frombuffer(imagen_bytes, np.uint8), IMREAD_COLOR)
        vector = extraer_vector(imagen_np)
        if vector is None:
            return jsonify({"error": "No se detectó rostro válido"}), 400
        vector_str = ','.join(map(str, vector.tolist()))

        sql = """
        INSERT INTO usuarios (nombre, apellidos, codigo, correo, requisitoriado, imagen, vector)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        valores = (nombre, apellidos, codigo, correo, requisitoriado, ruta_imagen, vector_str)
        cursor.execute(sql, valores)
        db.commit()

        return jsonify({"mensaje": "Usuario registrado"}), 200

    except Exception as e:
        print("Error al registrar:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/reconocer_rostro", methods=["POST"])
def reconocer_rostro():
    data = request.get_json()
    imagen_base64 = data.get('imagen')
    if not imagen_base64:
        return jsonify({"error": "No se envió imagen"}), 400

    try:
        imagen_bytes = base64.b64decode(imagen_base64)
        imagen_np = imdecode(np.frombuffer(imagen_bytes, np.uint8), IMREAD_COLOR)
        vector_entrada = extraer_vector(imagen_np)
        if vector_entrada is None:
            return jsonify({"error": "No se detectó rostro válido"}), 400

        cursor.execute("SELECT nombre, apellidos, codigo, requisitoriado, vector FROM usuarios")
        usuarios = cursor.fetchall()

        mejor_similitud = 0
        mejor_usuario = None

        for nombre, apellidos, codigo, requisitoriado, vector_str in usuarios:
            if not vector_str:
                continue
            try:
                vector_db = np.array(list(map(float, vector_str.split(','))))
                similitud = cosine_similarity([vector_entrada], [vector_db])[0][0]
                if similitud > mejor_similitud:
                    mejor_similitud = similitud
                    mejor_usuario = {
                        "nombre": nombre,
                        "apellido": apellidos,
                        "codigo": codigo,
                        "requisitoriado": requisitoriado
                    }
            except Exception as e:
                print(f"Error comparando vectores: {e}")

        if mejor_usuario and mejor_similitud > 0.6:
            return jsonify({
                "nombre": mejor_usuario["nombre"],
                "apellido": mejor_usuario["apellido"],
                "codigo": mejor_usuario["codigo"],
                "requisitoriado": mejor_usuario["requisitoriado"],
                "alerta": mejor_usuario["requisitoriado"] == True
            }), 200
        else:
            return jsonify({"mensaje": "Rostro no reconocido"}), 200

    except Exception as e:
        print("Error en reconocimiento:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/listar_usuarios", methods=["GET"])
def listar_usuarios():
    try:
        cursor.execute("SELECT id, nombre, apellidos, codigo, correo, requisitoriado, imagen FROM usuarios")
        resultados = cursor.fetchall()

        usuarios = []
        for fila in resultados:
            ruta_imagen = fila[6]
            imagen_base64 = ""

            if os.path.exists(ruta_imagen):
                with open(ruta_imagen, "rb") as img_file:
                    imagen_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            usuarios.append({
                "id": fila[0],
                "nombre": fila[1],
                "apellido": fila[2],
                "codigo": fila[3],
                "email": fila[4],
                "requisitoriado": fila[5],
                "imagen": imagen_base64
            })

        return jsonify(usuarios), 200

    except Exception as e:
        print("Error al listar:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/eliminar_usuario/<int:usuario_id>", methods=["DELETE"])
def eliminar_usuario(usuario_id):
    try:
        cursor.execute("SELECT imagen FROM usuarios WHERE id = %s", (usuario_id,))
        resultado = cursor.fetchone()

        if not resultado:
            return jsonify({"error": "Usuario no encontrado"}), 404

        ruta_imagen = resultado[0]

        cursor.execute("DELETE FROM usuarios WHERE id = %s", (usuario_id,))
        db.commit()

        if ruta_imagen and os.path.exists(ruta_imagen):
            os.remove(ruta_imagen)

        return jsonify({"mensaje": "Usuario eliminado correctamente"}), 200

    except Exception as e:
        print("Error al eliminar usuario:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/editar_usuario/<int:usuario_id>", methods=["PUT"])
def editar_usuario(usuario_id):
    data = request.get_json()
    nombre = data.get('nombre')
    apellidos = data.get('apellido')
    codigo = data.get('codigo')
    correo = data.get('email')
    requisitoriado = data.get('requisitoriado', False)
    imagen_base64 = data.get('imagen')

    try:
        cursor.execute("SELECT imagen FROM usuarios WHERE id = %s", (usuario_id,))
        resultado = cursor.fetchone()
        if not resultado:
            return jsonify({"error": "Usuario no encontrado"}), 404

        ruta_anterior = resultado[0]

        if imagen_base64:
            imagen_bytes = base64.b64decode(imagen_base64)
            nombre_archivo = f"{codigo}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            ruta_imagen = os.path.join(UPLOAD_FOLDER, nombre_archivo)
            with open(ruta_imagen, 'wb') as f:
                f.write(imagen_bytes)

            imagen_np = imdecode(np.frombuffer(imagen_bytes, np.uint8), IMREAD_COLOR)
            vector = extraer_vector(imagen_np)
            if vector is None:
                return jsonify({"error": "No se detectó rostro válido"}), 400
            vector_str = ','.join(map(str, vector.tolist()))

            if ruta_anterior and os.path.exists(ruta_anterior):
                os.remove(ruta_anterior)
        else:
            ruta_imagen = ruta_anterior
            cursor.execute("SELECT vector FROM usuarios WHERE id = %s", (usuario_id,))
            vector_str = cursor.fetchone()[0]

        sql = """
        UPDATE usuarios SET nombre = %s, apellidos = %s, codigo = %s,
        correo = %s, requisitoriado = %s, imagen = %s, vector = %s
        WHERE id = %s
        """
        valores = (nombre, apellidos, codigo, correo, requisitoriado, ruta_imagen, vector_str, usuario_id)
        cursor.execute(sql, valores)
        db.commit()

        return jsonify({"mensaje": "Usuario actualizado correctamente"}), 200

    except Exception as e:
        print("Error al editar usuario:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
