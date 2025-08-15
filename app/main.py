import os
from pathlib import Path
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash

from PIL import Image
from .processing import (
    image_to_matrix_bytes,
    matrix_to_image_bytes,
    save_histogram_png,
    ALLOWED_IMAGE_EXTS,
    ALLOWED_MATRIX_EXTS,
)

# Config
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (BASE_DIR / ".." / "output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = (BASE_DIR / ".." / "uploads").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")  # para flash messages


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    action = request.form.get("action")
    if action == "encode":
        file = request.files.get("image_file")
        if not file or file.filename == "":
            flash("Selecione uma imagem.")
            return redirect(url_for("index"))

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTS:
            flash(f"Extensão não suportada: {ext}")
            return redirect(url_for("index"))

        # Carrega imagem para memória
        img = Image.open(file.stream)

        # Gera matriz e métricas
        name, content, metrics, arr = image_to_matrix_bytes(img)

        # Salva matriz
        matrix_path = OUTPUT_DIR / name
        with open(matrix_path, "wb") as f:
            f.write(content)

        # Salva uma versão PNG (grayscale) para preview
        preview_name = Path(name).with_suffix(".png").name
        preview_path = OUTPUT_DIR / preview_name
        img.convert("L").save(preview_path, format="PNG")

        # Histograma
        hist_name = Path(name).with_suffix(".hist.png").name
        hist_path = OUTPUT_DIR / hist_name
        save_histogram_png(arr, str(hist_path))

        return render_template(
            "result.html",
            mode="encode",
            metrics=metrics,
            preview_file=preview_name,
            download_matrix=name,
            histogram_file=hist_name,
        )

    elif action == "decode":
        file = request.files.get("matrix_file")
        if not file or file.filename == "":
            flash("Selecione um arquivo de matriz.")
            return redirect(url_for("index"))

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_MATRIX_EXTS:
            flash(f"Extensão não suportada: {ext}")
            return redirect(url_for("index"))

        # Lê o texto
        matrix_text = file.stream.read().decode("utf-8", errors="replace")

        # Reconstrói imagem
        name, png_bytes, metrics, arr = matrix_to_image_bytes(matrix_text)

        # Salva imagem
        img_path = OUTPUT_DIR / name
        with open(img_path, "wb") as f:
            f.write(png_bytes)

        # Histograma
        hist_name = Path(name).with_suffix(".hist.png").name
        hist_path = OUTPUT_DIR / hist_name
        save_histogram_png(arr, str(hist_path))

        return render_template(
            "result.html",
            mode="decode",
            metrics=metrics,
            preview_file=name,
            download_image=name,
            histogram_file=hist_name,
        )

    else:
        flash("Ação inválida.")
        return redirect(url_for("index"))


@app.route("/output/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


def create_app():
    return app


if __name__ == "__main__":
    # Execução local (sem gunicorn)
    app.run(host="0.0.0.0", port=8000, debug=True)
