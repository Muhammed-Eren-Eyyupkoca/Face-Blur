**English** | [Türkçe](README.tr.md)

# Face Blur

A command-line tool that automatically detects faces in **images**, **video files** and a **live webcam feed** and blurs them, built with Python, OpenCV and MediaPipe.

- Two detectors: **MediaPipe Face Detection** (default, more accurate) and **OpenCV Haar Cascade** (no extra model needed)
- Two effects: **Gaussian blur** and **pixelation** (mosaic)
- Adjustable blur strength, padding around faces and detection confidence
- `--show-boxes` debug mode that draws the detected faces instead of blurring them

## Before / After

<!-- Replace with your own images, e.g. docs/before.jpg and docs/after.jpg -->
| Before | After |
| :----: | :---: |
| _add `docs/before.jpg`_ | _add `docs/after.jpg`_ |

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/<your-username>/Face-Blur.git
cd Face-Blur

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### MediaPipe model download

The project uses MediaPipe's current **Tasks API**, which needs a model file (`blaze_face_short_range.tflite`, ~230 KB). You don't have to download it yourself: on the first run with `--method mediapipe` it is downloaded automatically from Google's official MediaPipe model storage into the `models/` folder and reused afterwards. If you are offline, download it manually from
[this link](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite)
and place it at `models/blaze_face_short_range.tflite`.

The Haar Cascade detector uses an XML file that ships with the OpenCV package, so it works without any download.

## Usage

The tool has three modes: `image`, `video` and `webcam`. Options are written after the mode.

### Image

```bash
python main.py image samples/photo.jpg
# -> outputs/photo_blurred.jpg

python main.py image samples/photo.jpg --blur pixelate --strength 9 -o result.png
python main.py image samples/photo.jpg --show-boxes          # debug: draw boxes
```

Supported formats: `.jpg`, `.jpeg`, `.png`.

### Video

```bash
python main.py video samples/clip.mp4
# -> outputs/clip_blurred.mp4

python main.py video samples/clip.mp4 --method haar --padding 0.3
```

The output keeps the original FPS and resolution.
**Note:** audio is **not** copied, the output video is silent. To bring the audio back you can merge it with [FFmpeg](https://ffmpeg.org/):

```bash
ffmpeg -i outputs/clip_blurred.mp4 -i samples/clip.mp4 -map 0:v -map 1:a? -c copy clip_with_audio.mp4
```

### Webcam

```bash
python main.py webcam
python main.py webcam --camera 1 --blur pixelate
```

| Key | Action |
| --- | ------ |
| `q` | quit |
| `s` | save the current frame to `outputs/webcam_<timestamp>_blurred.jpg` |

When no output path is given, files are saved to `outputs/` with `_blurred` appended to the original name.

## Parameters

| Parameter | Modes | Default | Description |
| --------- | ----- | ------- | ----------- |
| `input` | image, video | – | Path to the input file |
| `-o`, `--output` | image, video | `outputs/<name>_blurred.<ext>` | Output path |
| `--camera` | webcam | `0` | Camera index |
| `--method` | all | `mediapipe` | Detector: `mediapipe` or `haar` |
| `--blur` | all | `gaussian` | Effect: `gaussian` or `pixelate` |
| `--strength` | all | `7` | Blur strength, `1` (light) – `10` (heavy). Scales with the face size |
| `--padding` | all | `0.2` | Extra margin on each side of a face, as a fraction of its size (`0.2` = 20%) |
| `--confidence` | all | `0.5` | Minimum detection score, `0.0` – `1.0`. **MediaPipe only**; Haar Cascade does not produce a comparable score |
| `--show-boxes` | all | off | Draw boxes instead of blurring: green = detection, yellow = area that would be blurred |

## How it works

1. **Detection** – Each image or video frame is passed to the detector. MediaPipe expects RGB, while OpenCV reads images as BGR, so the frame is converted first. Haar Cascade works on a grayscale, histogram-equalized copy. The detector returns a rectangle `(x, y, width, height)` for every face.
2. **Box expansion** – Detection boxes are usually tight and can miss the forehead, hair or chin. Every box is enlarged by `--padding` on each side and then clipped to the image borders, so faces at the edge of the frame never cause an error.
3. **Blurring the region** – The pixels inside the box are cut out and processed:
   - *Gaussian*: a Gaussian filter whose kernel size grows with the face size and `--strength` (the kernel size is always odd so it has a center pixel).
   - *Pixelate*: the region is shrunk to a few pixels and scaled back up with nearest-neighbour interpolation, producing large solid blocks.
4. **Putting it back** – The processed region is written back to the same position in the image. If no face is found, the image is left unchanged.

## Haar Cascade vs. MediaPipe

| | Haar Cascade | MediaPipe Face Detection |
| - | ------------ | ------------------------ |
| Technique | Classic machine learning (2001, Viola–Jones) on hand-crafted features | Lightweight neural network (BlazeFace) |
| Accuracy | Mostly frontal faces; misses profiles and tilted heads, more false positives | Handles different angles, lighting and partial occlusion much better |
| Confidence score | No | Yes (`--confidence`) |
| Speed | Fast on CPU | Also real-time on CPU |
| Setup | Ships with OpenCV | Needs a ~230 KB model (downloaded automatically) |
| Best range | Depends on `minSize` | The *short-range* model is optimised for faces within ~2 m of the camera; small, distant faces may be missed |

## Project structure

```
Face-Blur/
├── main.py            # command-line interface (argparse) and the image/video/webcam modes
├── detector.py        # face detectors (MediaPipe Tasks API + Haar Cascade)
├── blur.py            # box expansion, Gaussian blur, pixelation, debug boxes
├── requirements.txt
├── samples/           # put your input images/videos here
├── outputs/           # results (ignored by git)
└── models/            # downloaded model files (ignored by git)
```

## Notes

- `requirements.txt` pins `opencv-contrib-python` 4.x on purpose: OpenCV 5.x wheels no longer include the Haar cascade XML files. MediaPipe already depends on `opencv-contrib-python`, so don't install `opencv-python` alongside it (both provide the `cv2` module and conflict).
- MediaPipe/LiteRT prints a few `INFO:` / `W0000` lines to the terminal when the model is loaded. They are harmless.

## License

[MIT](LICENSE)
