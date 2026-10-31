[English](README.md) | **Türkçe**

# Face Blur

Python, OpenCV ve MediaPipe ile geliştirilmiş; **fotoğraflardaki**, **video dosyalarındaki** ve **canlı webcam görüntüsündeki** yüzleri otomatik olarak tespit edip bulanıklaştıran bir komut satırı aracı.

- İki tespit yöntemi: **MediaPipe Face Detection** (varsayılan, daha doğru) ve **OpenCV Haar Cascade** (ek model gerektirmez)
- İki efekt: **Gaussian blur** ve **pikselleştirme** (mozaik)
- Ayarlanabilir bulanıklık şiddeti, yüz çevresi kenar payı ve tespit güven eşiği
- Bulanıklaştırmak yerine tespit edilen yüzleri kutuyla gösteren `--show-boxes` hata ayıklama modu

## Öncesi / Sonrası

<!-- Kendi görsellerinizle değiştirin, ör. docs/before.jpg ve docs/after.jpg -->
| Öncesi | Sonrası |
| :----: | :-----: |
| _`docs/before.jpg` ekleyin_ | _`docs/after.jpg` ekleyin_ |

## Kurulum

Python 3.10+ gerekir.

```bash
git clone https://github.com/<kullanici-adiniz>/Face-Blur.git
cd Face-Blur

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### MediaPipe modelinin indirilmesi

Proje MediaPipe'ın güncel **Tasks API**'sini kullanır ve bu API bir model dosyasına (`blaze_face_short_range.tflite`, ~230 KB) ihtiyaç duyar. Dosyayı elle indirmeniz gerekmez: `--method mediapipe` ile ilk çalıştırmada Google'ın resmi MediaPipe model deposundan otomatik olarak `models/` klasörüne indirilir ve sonraki çalıştırmalarda tekrar kullanılır. İnternet bağlantınız yoksa dosyayı
[bu bağlantıdan](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite)
indirip `models/blaze_face_short_range.tflite` konumuna koyabilirsiniz.

Haar Cascade yöntemi OpenCV paketiyle birlikte gelen bir XML dosyası kullanır, bu yüzden herhangi bir indirme gerektirmez.

## Kullanım

Aracın üç modu vardır: `image`, `video` ve `webcam`. Seçenekler mod adından sonra yazılır.

### Görsel

```bash
python main.py image samples/photo.jpg
# -> outputs/photo_blurred.jpg

python main.py image samples/photo.jpg --blur pixelate --strength 9 -o result.png
python main.py image samples/photo.jpg --show-boxes          # hata ayıklama: kutu çiz
```

Desteklenen formatlar: `.jpg`, `.jpeg`, `.png`.

### Video

```bash
python main.py video samples/clip.mp4
# -> outputs/clip_blurred.mp4

python main.py video samples/clip.mp4 --method haar --padding 0.3
```

Çıktı videosu orijinal FPS ve çözünürlüğü korur.
**Not:** ses **aktarılmaz**, çıktı videosu sessizdir. Sesi geri eklemek için [FFmpeg](https://ffmpeg.org/) kullanabilirsiniz:

```bash
ffmpeg -i outputs/clip_blurred.mp4 -i samples/clip.mp4 -map 0:v -map 1:a? -c copy clip_with_audio.mp4
```

### Webcam

```bash
python main.py webcam
python main.py webcam --camera 1 --blur pixelate
```

| Tuş | İşlem |
| --- | ----- |
| `q` | çıkış |
| `s` | o anki kareyi `outputs/webcam_<zaman>_blurred.jpg` olarak kaydet |

Çıktı yolu verilmezse dosyalar `outputs/` klasörüne, orijinal adın sonuna `_blurred` eklenerek kaydedilir.

## Parametreler

| Parametre | Modlar | Varsayılan | Açıklama |
| --------- | ------ | ---------- | -------- |
| `input` | image, video | – | Giriş dosyasının yolu |
| `-o`, `--output` | image, video | `outputs/<ad>_blurred.<uzantı>` | Çıktı yolu |
| `--camera` | webcam | `0` | Kamera numarası |
| `--method` | hepsi | `mediapipe` | Tespit yöntemi: `mediapipe` veya `haar` |
| `--blur` | hepsi | `gaussian` | Efekt: `gaussian` veya `pixelate` |
| `--strength` | hepsi | `7` | Bulanıklık şiddeti, `1` (hafif) – `10` (yoğun). Yüz boyutuyla orantılı ölçeklenir |
| `--padding` | hepsi | `0.2` | Yüzün her kenarına eklenen pay, yüz boyutunun oranı olarak (`0.2` = %20) |
| `--confidence` | hepsi | `0.5` | Minimum tespit skoru, `0.0` – `1.0`. **Yalnızca MediaPipe**; Haar Cascade karşılaştırılabilir bir skor üretmez |
| `--show-boxes` | hepsi | kapalı | Bulanıklaştırmak yerine kutu çizer: yeşil = tespit, sarı = bulanıklaştırılacak alan |

## Nasıl çalışır?

1. **Tespit** – Her görsel veya video karesi tespit ediciye verilir. MediaPipe RGB bekler, OpenCV ise görüntüleri BGR olarak okur; bu yüzden kare önce dönüştürülür. Haar Cascade ise gri tonlamalı ve histogramı eşitlenmiş bir kopya üzerinde çalışır. Tespit edici her yüz için bir dikdörtgen `(x, y, genişlik, yükseklik)` döner.
2. **Kutu genişletme** – Tespit kutuları genelde dardır ve alnı, saçı ya da çeneyi dışarıda bırakabilir. Her kutu her kenardan `--padding` kadar büyütülür ve ardından görüntü sınırlarına kırpılır; böylece kenardaki yüzler hataya yol açmaz.
3. **Bölgeyi bulanıklaştırma** – Kutunun içindeki pikseller kesilip işlenir:
   - *Gaussian*: çekirdek boyutu yüz boyutu ve `--strength` ile büyüyen bir Gaussian filtresi (çekirdeğin bir merkez pikseli olması için boyutu her zaman tek sayıdır).
   - *Pikselleştirme*: bölge birkaç piksele küçültülür ve en yakın komşu (nearest-neighbour) yöntemiyle tekrar büyütülerek büyük düz renkli bloklar oluşturulur.
4. **Geri yerleştirme** – İşlenen bölge görüntüdeki aynı konuma geri yazılır. Hiç yüz bulunamazsa görüntü değiştirilmeden bırakılır.

## Haar Cascade ve MediaPipe karşılaştırması

| | Haar Cascade | MediaPipe Face Detection |
| - | ------------ | ------------------------ |
| Yöntem | El yapımı özellikler üzerinde klasik makine öğrenmesi (2001, Viola–Jones) | Hafif bir sinir ağı (BlazeFace) |
| Doğruluk | Çoğunlukla önden yüzler; profil ve eğik başları kaçırır, daha fazla yanlış pozitif | Farklı açı, ışık ve kısmi örtülmede çok daha başarılı |
| Güven skoru | Yok | Var (`--confidence`) |
| Hız | CPU'da hızlı | CPU'da da gerçek zamanlı |
| Kurulum | OpenCV ile birlikte gelir | ~230 KB model gerekir (otomatik indirilir) |
| En iyi mesafe | `minSize` değerine bağlı | *Kısa mesafe* modeli kameraya ~2 m'ye kadar olan yüzler için optimize edilmiştir; küçük, uzaktaki yüzler kaçabilir |

## Proje yapısı

```
Face-Blur/
├── main.py            # komut satırı arayüzü (argparse) ve image/video/webcam modları
├── detector.py        # yüz tespit sınıfları (MediaPipe Tasks API + Haar Cascade)
├── blur.py            # kutu genişletme, Gaussian blur, pikselleştirme, hata ayıklama kutuları
├── requirements.txt
├── samples/           # giriş görsellerinizi/videolarınızı buraya koyun
├── outputs/           # sonuçlar (git tarafından yok sayılır)
└── models/            # indirilen model dosyaları (git tarafından yok sayılır)
```

## Notlar

- `requirements.txt` bilinçli olarak `opencv-contrib-python` 4.x sürümünü sabitler: OpenCV 5.x paketleri artık Haar cascade XML dosyalarını içermiyor. MediaPipe zaten `opencv-contrib-python`'a bağımlıdır; yanına `opencv-python` kurmayın (ikisi de `cv2` modülünü sağlar ve çakışır).
- MediaPipe/LiteRT model yüklenirken terminale birkaç `INFO:` / `W0000` satırı yazar. Bunlar zararsızdır.

## Lisans

[MIT](LICENSE)
