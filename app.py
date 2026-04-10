import os
import json
import re
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL_VISION = "openai/gpt-4o-mini"
MODEL_ROUTINE = "openai/gpt-4o-mini"

DATA_FILE = Path("gym_data/machines.json")
PROFILE_FILE = Path("gym_data/profile.json")
Path("gym_data").mkdir(exist_ok=True)


def load_machines():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_profile():
    if PROFILE_FILE.exists():
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_profile(data):
    with open(PROFILE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.route("/")
def index():
    machines = load_machines()
    profile = load_profile()
    return render_template("index.html", machines=machines, profile=profile)


@app.route("/upload-machines", methods=["POST"])
def upload_machines():
    files = request.files.getlist("photos")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No se subieron fotos"}), 400

    content = []
    for f in files:
        if f.filename == "":
            continue
        img_bytes = f.read()
        img_data = base64.standard_b64encode(img_bytes).decode("utf-8")
        media_type = f.content_type if f.content_type else "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{img_data}"},
        })

    if not content:
        return jsonify({"error": "Fotos no válidas"}), 400

    content.append({
        "type": "text",
        "text": (
            "Analiza todas estas fotos de máquinas de gimnasio. "
            "Identifica cada máquina visible y responde ÚNICAMENTE con un JSON válido "
            "con este formato exacto, sin texto adicional antes ni después:\n"
            '{"machines": [{"name": "nombre de la máquina", "muscles": ["músculo1", "músculo2"], '
            '"type": "tipo (cardio/fuerza/funcional)", "description": "descripción breve de uso"}]}'
        ),
    })

    try:
        response = client.chat.completions.create(
            model=MODEL_VISION,
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        text = response.choices[0].message.content or ""

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            machines_data = json.loads(json_match.group())
        else:
            return jsonify({"error": "No se pudo analizar las imágenes"}), 500

        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(machines_data, f, ensure_ascii=False, indent=2)

        return jsonify({"success": True, "machines": machines_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-routine", methods=["POST"])
def get_routine():
    data = request.get_json()
    machines_data = load_machines()

    if not machines_data:
        return jsonify({"error": "No hay máquinas guardadas. Sube las fotos primero."}), 400

    profile = {
        "goal": data.get("goal", "ganar músculo"),
        "level": data.get("level", "principiante"),
        "days": data.get("days", "3"),
        "muscle_group": data.get("muscle_group", "todo el cuerpo"),
    }
    save_profile(profile)

    machines_list = json.dumps(machines_data.get("machines", []), ensure_ascii=False, indent=2)

    prompt = f"""Eres un entrenador personal experto y motivador. Crea una rutina de entrenamiento para HOY.

MAQUINAS DISPONIBLES EN EL GIMNASIO:
{machines_list}

PERFIL DEL USUARIO:
- Objetivo: {profile['goal']}
- Nivel: {profile['level']}
- Dias de entrenamiento por semana: {profile['days']}
- Grupo muscular para HOY: {profile['muscle_group']}

Crea una rutina detallada y motivadora que incluya:
1. Calentamiento (5-10 min) - ejercicios especificos
2. Rutina principal - usando SOLO las maquinas disponibles, con:
   - Nombre del ejercicio y maquina a usar
   - Series x repeticiones (o tiempo)
   - Descanso entre series
   - Consejo de forma/tecnica
3. Vuelta a la calma (5 min)
4. Consejo motivacional final

Formato: usa titulos claros, bullets y separadores visuales para que sea facil de leer en el movil.
Si el grupo muscular elegido no tiene maquinas especificas disponibles, combina con ejercicios corporales complementarios."""

    try:
        response = client.chat.completions.create(
            model=MODEL_ROUTINE,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        routine_text = response.choices[0].message.content or ""
        return jsonify({"routine": routine_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/machines", methods=["GET"])
def get_machines():
    machines = load_machines()
    if machines:
        return jsonify(machines)
    return jsonify({"machines": []})


@app.route("/reset-machines", methods=["POST"])
def reset_machines():
    if DATA_FILE.exists():
        DATA_FILE.unlink()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
