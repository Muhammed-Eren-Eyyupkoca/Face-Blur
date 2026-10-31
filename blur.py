"""Yüz bölgelerini genişletme, bulanıklaştırma ve kutu çizme fonksiyonları."""

import cv2
import numpy as np

# (x, y, genişlik, yükseklik) — piksel cinsinden yüz kutusu
Box = tuple[int, int, int, int]


def expand_box(box: Box, padding: float, image_shape: tuple[int, ...]) -> Box | None:
    """Kutuyu her yönde `padding` oranında büyütür ve görüntü sınırlarına kırpar.

    Kırpma sonrası alanı kalmayan kutular için None döner.
    """
    x, y, w, h = box
    img_h, img_w = image_shape[:2]

    # Tespit kutuları genelde alnı, saçı ve çeneyi tam kapsamaz;
    # kenar payı bu bölgelerin de gizlenmesini sağlar.
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    # Kutu görüntü dışına taşabilir (ör. yüz kenarda yarım görünüyorsa),
    # bu yüzden koordinatları görüntü sınırlarına sıkıştırıyoruz.
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def gaussian_blur(region: np.ndarray, strength: int) -> np.ndarray:
    """Bölgeye, boyutuyla orantılı çekirdekle Gaussian blur uygular."""
    # Çekirdek boyutunu yüz boyutuna göre ölçekliyoruz; böylece küçük ve
    # büyük yüzler aynı şiddet değerinde benzer oranda bulanıklaşır.
    kernel = int(min(region.shape[:2]) * strength / 10)

    # Gaussian çekirdeğinin bir merkez pikseli olmalı, bu yüzden OpenCV
    # tek sayı ister; çift çıkarsa bir artırıyoruz.
    kernel = max(3, kernel | 1)

    # sigma=0 verildiğinde OpenCV sigmayı çekirdek boyutundan hesaplar.
    return cv2.GaussianBlur(region, (kernel, kernel), 0)


def pixelate(region: np.ndarray, strength: int) -> np.ndarray:
    """Bölgeyi küçültüp tekrar büyüterek mozaik (piksel) efekti uygular."""
    h, w = region.shape[:2]

    # Şiddet arttıkça kısa kenardaki blok sayısı azalır (1 → 20, 10 → 2 blok).
    blocks = max(2, 22 - 2 * strength)
    scale = blocks / min(h, w)
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))

    # INTER_LINEAR ile küçültme her bloğun ortalama rengini verir;
    # INTER_NEAREST ile büyütme ise blokların keskin kenarlı kalmasını sağlar.
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def blur_faces(
    image: np.ndarray,
    boxes: list[Box],
    method: str = "gaussian",
    strength: int = 7,
    padding: float = 0.2,
) -> np.ndarray:
    """Görüntüdeki her yüz kutusunu bulanıklaştırılmış haliyle değiştirir."""
    output = image.copy()
    blur_fn = pixelate if method == "pixelate" else gaussian_blur

    for box in boxes:
        expanded = expand_box(box, padding, image.shape)
        if expanded is None:
            continue
        x, y, w, h = expanded

        # NumPy dilimi orijinal dizinin bir görünümüdür (view); bulanık
        # bölgeyi aynı dilime yazmak onu görüntüye geri yerleştirir.
        output[y:y + h, x:x + w] = blur_fn(output[y:y + h, x:x + w], strength)

    return output


def draw_boxes(image: np.ndarray, boxes: list[Box], padding: float = 0.2) -> np.ndarray:
    """Hata ayıklama için tespit edilen (ve genişletilmiş) yüz kutularını çizer."""
    output = image.copy()

    for box in boxes:
        # Ham tespit kutusu: yeşil
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Bulanıklaştırılacak genişletilmiş alan: sarı (OpenCV renkleri BGR sırasındadır)
        expanded = expand_box(box, padding, image.shape)
        if expanded is not None:
            ex, ey, ew, eh = expanded
            cv2.rectangle(output, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 1)

    return output
