# =============================================================================
# app.py — Backend Flask untuk Program Tiket Sampah Adiwiyata
# MI Nurul Mun'im
#
# Fitur aktif:
#   - Confidence Threshold 0.72: hanya deteksi dengan conf >= 0.72 yang dihitung
# =============================================================================

import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Inisialisasi Aplikasi Flask
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
# Load model YOLOv8 hasil training
# -----------------------------------------------------------------------------
model = YOLO("best (1).pt")

# -----------------------------------------------------------------------------
# Buka webcam (index 0 = kamera utama perangkat)
# -----------------------------------------------------------------------------
camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture("http://192.168.1.11:4747/video")

# -----------------------------------------------------------------------------
# Mapping nama kelas YOLO -> kategori sampah
# -----------------------------------------------------------------------------
waste_category = {
    "Botol Plastik":   "Non Organik",
    "Daun Ketapang":   "Organik",
    "Daun Mangga":     "Organik",
    "Gelas Plastik":   "Non Organik",
    "Kantong Plastik": "Non Organik",
    "Kardus":          "Organik",
    "Kemasan Plastik": "Non Organik",
    "Kertas":          "Organik",
    "Kertas Minyak":   "Non Organik",
    "Kotak Susu":      "Non Organik",
    "Mika":            "Non Organik",
}

# =============================================================================
# KONFIGURASI FILTER DETEKSI
# =============================================================================
# Confidence threshold: prediksi di bawah nilai ini diabaikan sepenuhnya.
# 0.72 = model harus yakin minimal 72% sebelum objek dianggap terdeteksi.
CONF_THRESHOLD = 0.5



# =============================================================================
# STATE GLOBAL
# =============================================================================

# Hitungan deteksi terbaru — ditampilkan ke frontend via /api/live-counts
live_counts = {
    "organic_count":    0,
    "nonorganic_count": 0,
    "total_count":      0,
    "fps":              0,
}

# Lock untuk keamanan akses multi-thread pada live_counts
counts_lock = threading.Lock()


# =============================================================================
# GENERATOR: Ambil frame, filter confidence, validasi delay, kirim MJPEG
# =============================================================================
def generate_frames():
    """
    Generator utama yang:
      1. Membaca frame dari webcam secara terus-menerus.
      2. Menjalankan inferensi YOLO dengan confidence threshold 0.72.
      3. Menggambar bounding box + label pada setiap objek yang lolos threshold.
      4. Memperbarui live_counts global untuk endpoint API.
      5. Menghasilkan byte MJPEG untuk di-stream ke browser.
    """
    global live_counts
    prev_time = 0.0

    while True:
        success, frame = camera.read()
        if not success:
            # Kamera gagal dibaca — tunggu sebentar lalu coba lagi
            time.sleep(0.05)
            continue

        annotated_frame = frame.copy()

        # ── Jalankan inferensi YOLO ──────────────────────────────────────────
        # conf=CONF_THRESHOLD: bounding box dengan conf rendah tidak dikirim
        # results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        results = model(frame, conf=CONF_THRESHOLD, iou=0.45, verbose=False)

        boxes = results[0].boxes
        names = model.names

        organic_frame    = 0
        nonorganic_frame = 0

        for box in boxes:
            conf       = float(box.conf[0])
            cls_id     = int(box.cls[0])
            class_name = names[cls_id]

            # ── Confidence Threshold ─────────────────────────────────────────
            # Abaikan prediksi di bawah 72%
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            category = waste_category.get(class_name, "Tidak diketahui")

            # Tentukan warna berdasarkan kategori
            if category == "Organik":
                color         = (34, 197, 94)   # hijau (BGR OpenCV)
                organic_frame += 1
            else:
                color            = (239, 68, 68) # merah (BGR OpenCV)
                nonorganic_frame += 1

            # Gambar bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Latar belakang teks agar mudah dibaca
            label = f"{class_name} | {category} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - th - 6),
                (x1 + tw + 4, y1),
                color, -1,
            )
            cv2.putText(
                annotated_frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        # ── Hitung FPS ───────────────────────────────────────────────────────
        current_time = time.time()
        fps       = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # ── Overlay info di sudut kiri atas frame ────────────────────────────
        overlay_lines = [
            (f"FPS         : {int(fps)}",          (20,  38), (255, 255,   0)),
            (f"Organik     : {organic_frame}",      (20,  72), ( 34, 197,  94)),
            (f"Non Organik : {nonorganic_frame}",   (20, 106), (239,  68,  68)),
            (f"Conf >= {int(CONF_THRESHOLD * 100)}%",(20, 140), (180, 180, 180)),
        ]
        for text, pos, col in overlay_lines:
            cv2.putText(
                annotated_frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, col, 2, cv2.LINE_AA,
            )

        # ── Perbarui state global (thread-safe) ──────────────────────────────
        with counts_lock:
            live_counts["organic_count"]    = organic_frame
            live_counts["nonorganic_count"] = nonorganic_frame
            live_counts["total_count"]      = organic_frame + nonorganic_frame
            live_counts["fps"]              = int(fps)

        # ── Encode frame ke JPEG lalu hasilkan sebagai byte stream ──────────
        ret, buffer = cv2.imencode(
            ".jpg", annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# =============================================================================
# ROUTE: Halaman Utama Dashboard
# =============================================================================
@app.route("/")
def index():
    """Render halaman dashboard utama (index.html)."""
    return render_template("index.html")


# =============================================================================
# ROUTE: Halaman Deteksi Kamera
# =============================================================================
@app.route("/deteksi")
def deteksi():
    """Render halaman deteksi kamera real-time (deteksi.html)."""
    return render_template("deteksi.html")


# =============================================================================
# ROUTE: Stream Video MJPEG
# =============================================================================
@app.route("/video")
def video():
    """
    Mengalirkan video kamera sebagai MJPEG stream.
    Digunakan oleh tag <img src="/video"> di halaman deteksi.
    """
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# =============================================================================
# API ENDPOINT: Data Hitungan Deteksi Real-Time
# =============================================================================
@app.route("/api/live-counts", methods=["GET"])
def api_live_counts():
    """
    Mengembalikan jumlah deteksi terbaru dalam format JSON.
    Dipanggil JavaScript di halaman deteksi setiap 1-2 detik.

    Response JSON:
    {
        "organic_count":    <int>,
        "nonorganic_count": <int>,
        "total_count":      <int>,
        "fps":              <int>
    }
    """
    with counts_lock:
        data = dict(live_counts)
    return jsonify(data)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Nonaktifkan debug=True di lingkungan produksi
    app.run(debug=True, host="0.0.0.0", port=5000)
