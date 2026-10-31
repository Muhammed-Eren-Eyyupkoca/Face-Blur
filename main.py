"""Face Blur — görsel, video ve webcam görüntülerindeki yüzleri otomatik bulanıklaştırır.

Örnekler:
    python main.py image samples/photo.jpg
    python main.py video samples/clip.mp4 --blur pixelate
    python main.py webcam --method haar
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from blur import Box, blur_faces, draw_boxes
from detector import Detector, create_detector

OUTPUTS_DIR = Path(__file__).parent / "outputs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4"}


# --------------------------------------------------------------------------- #
# Argüman doğrulama
# --------------------------------------------------------------------------- #

def strength_type(value: str) -> int:
    """Bulanıklık şiddetinin 1-10 aralığında bir tam sayı olduğunu doğrular."""
    number = int(value)
    if not 1 <= number <= 10:
        raise argparse.ArgumentTypeError("must be an integer between 1 and 10")
    return number


def padding_type(value: str) -> float:
    """Kenar payının 0-1 aralığında bir oran olduğunu doğrular."""
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return number


def confidence_type(value: str) -> float:
    """Güven eşiğinin 0-1 aralığında olduğunu doğrular."""
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Alt komutları (image / video / webcam) ve ortak seçenekleri tanımlar."""
    # Ortak seçenekler bir "ebeveyn" ayrıştırıcıda tanımlanır ve her alt komuta
    # eklenir; böylece seçenekler mod adından sonra yazılabilir.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--method", choices=["mediapipe", "haar"], default="mediapipe",
                        help="face detection method (default: mediapipe)")
    common.add_argument("--blur", choices=["gaussian", "pixelate"], default="gaussian",
                        help="blur type (default: gaussian)")
    common.add_argument("--strength", type=strength_type, default=7,
                        help="blur strength from 1 (light) to 10 (heavy) (default: 7)")
    common.add_argument("--padding", type=padding_type, default=0.2,
                        help="extra margin around each face as a fraction of its size (default: 0.2)")
    common.add_argument("--confidence", type=confidence_type, default=0.5,
                        help="minimum detection confidence, MediaPipe only (default: 0.5)")
    common.add_argument("--show-boxes", action="store_true",
                        help="draw boxes around detected faces instead of blurring (debugging)")

    parser = argparse.ArgumentParser(
        description="Automatically detect and blur faces in images, videos or a webcam feed.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    image = subparsers.add_parser("image", parents=[common], help="blur faces in a single image")
    image.add_argument("input", type=Path, help="input image (.jpg, .jpeg, .png)")
    image.add_argument("-o", "--output", type=Path,
                       help="output path (default: outputs/<name>_blurred.<ext>)")

    video = subparsers.add_parser("video", parents=[common], help="blur faces in a video file")
    video.add_argument("input", type=Path, help="input video (.mp4)")
    video.add_argument("-o", "--output", type=Path,
                       help="output path (default: outputs/<name>_blurred.mp4)")

    webcam = subparsers.add_parser("webcam", parents=[common],
                                   help="live preview from a webcam (q: quit, s: save frame)")
    webcam.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")

    return parser


# --------------------------------------------------------------------------- #
# Yardımcı fonksiyonlar
# --------------------------------------------------------------------------- #

def default_output_path(input_path: Path) -> Path:
    """Giriş adının sonuna `_blurred` ekleyerek outputs/ altında bir yol üretir."""
    return OUTPUTS_DIR / f"{input_path.stem}_blurred{input_path.suffix.lower()}"


def validate_file(path: Path, allowed: set[str]) -> None:
    """Giriş dosyasının var olduğunu ve uzantısının desteklendiğini kontrol eder."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() not in allowed:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: {', '.join(sorted(allowed))}"
        )


def prepare_paths(args: argparse.Namespace) -> None:
    """Giriş dosyasını doğrular ve çıktı yolunu belirler (webcam modunda bir şey yapmaz).

    Model yüklenmeden önce çağrılır; böylece hatalı bir yol için model boşuna indirilmez.
    """
    if args.mode == "webcam":
        return

    allowed = IMAGE_EXTENSIONS if args.mode == "image" else VIDEO_EXTENSIONS
    validate_file(args.input, allowed)

    args.output = args.output or default_output_path(args.input)
    if args.output.suffix.lower() not in allowed:
        raise ValueError(f"Output file must be one of: {', '.join(sorted(allowed))}")


def read_image(path: Path) -> np.ndarray:
    """Görseli diskten okur; Türkçe karakter içeren yollarda da çalışır."""
    # cv2.imread Windows'ta ASCII olmayan yolları (ör. "Masaüstü") okuyamaz;
    # dosyayı NumPy ile bayt olarak okuyup bellekte çözmek bu sorunu aşar.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image (corrupted or not an image?): {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Görseli diske yazar; gerekirse hedef klasörü oluşturur."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # read_image ile aynı neden: imwrite yerine bellekte kodlayıp NumPy ile yazıyoruz.
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image as '{path.suffix}': {path}")
    buffer.tofile(str(path))


def process_frame(
    frame: np.ndarray, detector: Detector, args: argparse.Namespace
) -> tuple[np.ndarray, list[Box]]:
    """Tek bir karede yüzleri bulur; bulanıklaştırılmış (veya kutulu) kareyi ve kutuları döner."""
    boxes = detector.detect(frame)
    if args.show_boxes:
        return draw_boxes(frame, boxes, args.padding), boxes
    return blur_faces(frame, boxes, args.blur, args.strength, args.padding), boxes


# --------------------------------------------------------------------------- #
# Modlar
# --------------------------------------------------------------------------- #

def run_image(args: argparse.Namespace, detector: Detector) -> None:
    """Tek bir görseli işler ve sonucu kaydeder."""
    image = read_image(args.input)
    result, boxes = process_frame(image, detector, args)

    if boxes:
        print(f"Detected {len(boxes)} face(s).")
    else:
        print("No faces found; the image is saved unchanged.")

    write_image(args.output, result)
    print(f"Saved: {args.output}")


def run_video(args: argparse.Namespace, detector: Detector) -> None:
    """Video dosyasını kare kare işler ve aynı FPS/çözünürlükte kaydeder (ses aktarılmaz)."""
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    # Çıktının orijinalle aynı hız ve boyutta olması için bu değerleri kaynaktan alıyoruz.
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0  # bazı dosyalar FPS bilgisi içermez
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # "mp4v" codec'i OpenCV'nin pip paketinde ek kurulum gerektirmeden çalışır.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {args.output}")

    frame_count = 0
    frames_with_faces = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:  # videonun sonu
                break
            result, boxes = process_frame(frame, detector, args)
            writer.write(result)

            frame_count += 1
            frames_with_faces += bool(boxes)
            if frame_count % 30 == 0 or frame_count == total:
                print(f"\rProcessed {frame_count}/{total or '?'} frames", end="", flush=True)
    finally:
        capture.release()
        writer.release()

    print(f"\nFaces found in {frames_with_faces} of {frame_count} frames.")
    print(f"Saved: {args.output} ({width}x{height} @ {fps:.2f} FPS, no audio)")


def run_webcam(args: argparse.Namespace, detector: Detector) -> None:
    """Webcam'den canlı önizleme gösterir; 'q' ile çıkar, 's' ile kareyi kaydeder."""
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {args.camera}). Is it connected and not used "
            "by another application? Try a different index with --camera."
        )

    window = "Face Blur - q: quit, s: save"
    print("Webcam started. Press 'q' to quit, 's' to save the current frame.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            result, _ = process_frame(frame, detector, args)
            cv2.imshow(window, result)

            # waitKey hem pencereyi günceller hem de tuşu okur;
            # & 0xFF bazı platformlarda dönen ek bitleri temizler.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = OUTPUTS_DIR / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}_blurred.jpg"
                write_image(path, result)
                print(f"Saved: {path}")

            # Kullanıcı pencereyi X ile kapatırsa döngüden çık.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    """Argümanları ayrıştırır, seçilen modu çalıştırır ve çıkış kodunu döner."""
    args = build_parser().parse_args()
    runners = {"image": run_image, "video": run_video, "webcam": run_webcam}

    try:
        prepare_paths(args)
        detector = create_detector(args.method, args.confidence)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        runners[args.mode](args, detector)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        detector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
