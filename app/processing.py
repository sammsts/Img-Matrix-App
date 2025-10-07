import numpy as np
from scipy import stats

# --- Detecção de Bordas ---
def sobel_manual(arr: np.ndarray) -> np.ndarray:
    """
    Detecção de bordas usando operador Sobel (implementação manual).

    # Explicação do código manual:
    # 1. Define os kernels Kx e Ky para detectar bordas horizontais e verticais.
    # 2. Converte o array para int32 para evitar overflow durante os cálculos.
    # 3. Para cada pixel (exceto bordas), extrai uma região 3x3 ao redor.
    # 4. Calcula gx (gradiente horizontal) e gy (vertical) multiplicando a região pelos kernels e somando.
    # 5. Usa np.hypot(gx, gy) para obter a magnitude do gradiente (intensidade da borda).
    # 6. Limita o valor entre 0 e 255 e armazena no pixel correspondente da saída.
    # 7. Retorna a imagem resultante como uint8.
    """
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=int)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)
    arr = arr.astype(np.int32)
    h, w = arr.shape
    out = np.zeros_like(arr)
    for i in range(1, h-1):
        for j in range(1, w-1):
            region = arr[i-1:i+2, j-1:j+2]
            gx = np.sum(Kx * region)
            gy = np.sum(Ky * region)
            val = min(255, int(np.hypot(gx, gy)))
            out[i, j] = val
    return out.astype(np.uint8)

def prewitt(arr: np.ndarray) -> np.ndarray:
    """
    Detecção de bordas usando operador Prewitt (scipy.ndimage).
    """
    from scipy import ndimage
    arr = arr.astype(np.float32)
    gx = ndimage.prewitt(arr, axis=1)
    gy = ndimage.prewitt(arr, axis=0)
    mag = np.hypot(gx, gy)
    mag = np.clip(mag, 0, 255)
    return mag.astype(np.uint8)

def laplaciano(arr: np.ndarray) -> np.ndarray:
    """
    Detecção de bordas usando operador Laplaciano (scipy.ndimage).
    """
    from scipy import ndimage
    arr = arr.astype(np.float32)
    lap = ndimage.laplace(arr)
    lap = np.abs(lap)
    lap = np.clip(lap, 0, 255)
    return lap.astype(np.uint8)

def kirsch(arr: np.ndarray) -> np.ndarray:
    """
    Detecção de bordas usando operador Kirsch (scipy.ndimage, via máximo de convoluções).
    """
    from scipy import ndimage
    arr = arr.astype(np.float32)
    kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
    ]
    responses = [ndimage.convolve(arr, k, mode='nearest') for k in kernels]
    kirsch = np.max(responses, axis=0)
    kirsch = np.clip(kirsch, 0, 255)
    return kirsch.astype(np.uint8)

def reduce_noise(arr, mask_size=3, method="median", scan="row"):
    """
    Aplica filtro de redução de ruído salt-and-pepper.
    arr: matriz numpy 2D (imagem em tons de cinza)
    mask_size: tamanho da janela (ímpar, 3-21)
    method: 'median' ou 'mode'
    scan: 'row' (linha) ou 'col' (coluna)
    """
    pad = mask_size // 2
    padded = np.pad(arr, pad, mode='edge')
    result = arr.copy()
    h, w = arr.shape

    if scan == "row":
        for i in range(h):
            for j in range(w):
                window = padded[i:i+mask_size, j:j+mask_size]
                vals = window.flatten()
                if method == "median":
                    result[i, j] = np.median(vals)
                elif method == "mode":
                    result[i, j] = stats.mode(vals, keepdims=True)[0][0]
    elif scan == "col":
        for j in range(w):
            for i in range(h):
                window = padded[i:i+mask_size, j:j+mask_size]
                vals = window.flatten()
                if method == "median":
                    result[i, j] = np.median(vals)
                elif method == "mode":
                    result[i, j] = stats.mode(vals, keepdims=True)[0][0]
    return result.astype(np.uint8)
import io
import time
from typing import Tuple, Dict, Any, List

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # backend para gerar figuras sem display
import matplotlib.pyplot as plt


ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ALLOWED_MATRIX_EXTS = {".txt", ".mtx"}


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _ensure_grayscale(img: Image.Image) -> Image.Image:
    # Converte para L (8-bit grayscale) garantidamente
    if img.mode != "L":
        img = img.convert("L")
    return img


def compute_entropy(arr: np.ndarray) -> float:
    """
    Calcula entropia de Shannon em bits/pixel para imagem em tons de cinza.
    """
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 255))
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist.astype(np.float64) / total
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def image_to_matrix_bytes(img: Image.Image) -> Tuple[str, bytes, Dict[str, Any], np.ndarray]:
    """
    Converte uma imagem para matriz textual.
    Retorna: (nome_arquivo, bytes_conteudo, metricas, array_pixels)
    """
    start_time = time.perf_counter()

    gray = _ensure_grayscale(img)
    width, height = gray.size
    arr = np.array(gray, dtype=np.uint8)

    # Monta linhas de texto
    lines: List[str] = [f"{width} {height}"]
    for y in range(height):
        lines.append(" ".join(str(int(v)) for v in arr[y, :]))
    content = ("\n".join(lines) + "\n").encode("utf-8")

    elapsed = time.perf_counter() - start_time

    metrics = compute_metrics(arr)
    metrics["processing_time_sec"] = round(elapsed, 4)
    metrics["entropy_bits_per_pixel"] = compute_entropy(arr)
    metrics["file_size_bytes"] = len(content)

    name = f"matriz_{width}x{height}_{_timestamp()}.txt"
    return name, content, metrics, arr


def matrix_to_image_bytes(matrix_text: str) -> Tuple[str, bytes, Dict[str, Any], np.ndarray]:
    """
    Converte uma matriz textual para imagem PNG.
    Retorna: (nome_arquivo, png_bytes, metricas, array_pixels)
    """
    start_time = time.perf_counter()

    lines = [ln.strip() for ln in matrix_text.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Arquivo de matriz vazio.")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError("Cabeçalho inválido. Esperado: <width> <height>.")

    width, height = int(header[0]), int(header[1])
    if len(lines[1:]) < height:
        raise ValueError(f"Dados insuficientes: esperadas {height} linhas de pixels, mas vieram {len(lines[1:])}.")

    data = []
    for y in range(height):
        row_vals = lines[1 + y].split()
        if len(row_vals) < width:
            raise ValueError(f"Linha {y} possui {len(row_vals)} valores; esperado {width}.")
        row = [int(v) for v in row_vals[:width]]
        data.extend(row)

    arr = np.array(data, dtype=np.uint8).reshape((height, width))
    img = Image.fromarray(arr, mode="L")

    # Exporta como PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    elapsed = time.perf_counter() - start_time

    metrics = compute_metrics(arr)
    metrics["processing_time_sec"] = round(elapsed, 4)
    metrics["entropy_bits_per_pixel"] = compute_entropy(arr)
    metrics["file_size_bytes"] = len(png_bytes)

    name = f"reconstruida_{width}x{height}_{_timestamp()}.png"
    return name, png_bytes, metrics, arr


def compute_metrics(arr: np.ndarray) -> Dict[str, Any]:
    arr = arr.astype(np.uint8)
    stats = {
        "width": int(arr.shape[1]) if arr.ndim == 2 else 0,
        "height": int(arr.shape[0]) if arr.ndim == 2 else 0,
        "min": int(arr.min()) if arr.size else 0,
        "max": int(arr.max()) if arr.size else 0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
    }
    return stats


def save_histogram_png(arr: np.ndarray, out_path: str) -> None:
    # Grava histograma de 0..255
    plt.figure()
    plt.hist(arr.flatten(), bins=256, range=(0, 255))
    plt.title("Histograma (tons de cinza)")
    plt.xlabel("Nível de cinza (0–255)")
    plt.ylabel("Quantidade de pixels")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def mirror_vertical(arr: np.ndarray) -> np.ndarray:
    """Espelha verticalmente (flipud)"""
    return np.flipud(arr)

def mirror_horizontal(arr: np.ndarray) -> np.ndarray:
    """Espelha horizontalmente (fliplr)"""
    return np.fliplr(arr)

def rotate_90_left(arr: np.ndarray) -> np.ndarray:
    """Rotaciona 90º para a esquerda (counter-clockwise)"""
    return np.rot90(arr, k=1)

def rotate_90_right(arr: np.ndarray) -> np.ndarray:
    """Rotaciona 90º para a direita (clockwise)"""
    return np.rot90(arr, k=-1)
