"""Yüz tespit sınıfları: MediaPipe Face Detection (Tasks API) ve OpenCV Haar Cascade.

Her iki sınıf da aynı arayüzü sunar: `detect(image)` BGR bir görüntü alır ve
(x, y, genişlik, yükseklik) biçiminde yüz kutularının listesini döner.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from blur import Box

MODELS_DIR = Path(__file__).parent / "models"
MEDIAPIPE_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
HAAR_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def ensure_mediapipe_model(path: Path = MEDIAPIPE_MODEL_PATH) -> Path:
    """Model dosyası yoksa resmi MediaPipe deposundan indirir ve yolunu döner."""
    if path.exists():
        return path

    print(f"MediaPipe model not found, downloading to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Önce geçici dosyaya indiriyoruz; indirme yarıda kesilirse
    # bozuk bir model dosyası bir sonraki çalıştırmada kullanılmaz.
    tmp_path = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, tmp_path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the MediaPipe model ({exc}). Check your internet "
            f"connection or download it manually from:\n  {MEDIAPIPE_MODEL_URL}\n"
            f"and place it at: {path}"
        ) from exc

    tmp_path.replace(path)
    print("Download complete.")
    return path


class MediaPipeDetector:
    """MediaPipe Tasks API'si (BlazeFace kısa mesafe modeli) ile yüz tespiti."""

    def __init__(self, confidence: float = 0.5) -> None:
        """Modeli yükler (gerekirse indirir) ve tespit ediciyi oluşturur."""
        model_path = ensure_mediapipe_model()
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image: np.ndarray) -> list[Box]:
        """BGR görüntüdeki yüzleri tespit eder ve kutularını döner."""
        # OpenCV görüntüleri BGR sırasında okur, MediaPipe ise RGB bekler;
        # dönüştürmezsek kırmızı ve mavi kanallar yer değiştirir ve
        # model renkleri yanlış yorumlayarak doğruluğu düşer.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detector.detect(mp_image)
        boxes: list[Box] = []
        for detection in result.detections:
            bb = detection.bounding_box
            boxes.append((bb.origin_x, bb.origin_y, bb.width, bb.height))
        return boxes

    def close(self) -> None:
        """Model kaynaklarını serbest bırakır."""
        self._detector.close()


class HaarDetector:
    """OpenCV ile birlikte gelen Haar Cascade sınıflandırıcısı ile yüz tespiti."""

    def __init__(self) -> None:
        """OpenCV paketindeki önden yüz (frontal face) cascade dosyasını yükler."""
        self._cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {HAAR_CASCADE_PATH}")

    def detect(self, image: np.ndarray) -> list[Box]:
        """BGR görüntüdeki yüzleri tespit eder ve kutularını döner."""
        # Haar özellikleri yalnızca parlaklık farklarına bakar; gri tonlama
        # hem yeterlidir hem de işlemi hızlandırır.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Histogram eşitleme kontrastı artırarak loş görüntülerde tespiti iyileştirir.
        gray = cv2.equalizeHist(gray)

        # minNeighbors: bir adayın yüz sayılması için gereken komşu tespit sayısı;
        # yükseltmek yanlış pozitifleri azaltır ama bazı yüzleri kaçırabilir.
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [tuple(int(v) for v in face) for face in faces]

    def close(self) -> None:
        """Arayüz uyumu için; Haar Cascade serbest bırakılacak kaynak tutmaz."""


# Her iki sınıf da aynı detect()/close() arayüzünü sunar.
Detector = MediaPipeDetector | HaarDetector


def create_detector(method: str, confidence: float = 0.5) -> Detector:
    """Seçilen yönteme göre uygun yüz tespit ediciyi oluşturur."""
    if method == "haar":
        return HaarDetector()
    return MediaPipeDetector(confidence)
