# Deteksi Wajah, Mata, dan Senyuman dengan Haar Cascade

## Deskripsi Proyek
Proyek ini menggunakan OpenCV dan model Haar Cascade untuk mendeteksi wajah, mata, dan senyuman secara real-time melalui webcam. Aplikasi ini dirancang untuk mengenali wajah manusia, mendeteksi mata, serta senyuman dengan menampilkan kotak pembatas (bounding box) di area yang terdeteksi.

---

## Persyaratan Sistem
- Python 3.x
- OpenCV library (cv2)
- Webcam

---

## Cara Kerja
1. **Memuat Model Haar Cascade:**
   - Proyek menggunakan model pralatih Haar Cascade dari OpenCV untuk mendeteksi wajah, mata, dan senyuman.
   - File model:
     - `haarcascade_frontalface_default.xml` untuk deteksi wajah
     - `haarcascade_eye.xml` untuk deteksi mata
     - `haarcascade_smile.xml` untuk deteksi senyuman

2. **Pengambilan Gambar Real-Time:**
   - Webcam digunakan untuk menangkap video secara langsung (bisa menggunakan kamera bawaan laptop).
   - Gambar yang diambil diubah menjadi skala abu-abu karena Haar Cascade bekerja lebih baik pada gambar monokrom.

3. **Proses Deteksi:**
   - Wajah yang terdeteksi akan dibingkai dengan kotak biru.
   - Mata yang terdeteksi akan dibingkai dengan kotak hijau.
   - Senyuman yang terdeteksi dalam wajah akan dibingkai dengan kotak kuning.

4. **Tampilan Hasil:**
   - Hasil deteksi ditampilkan di jendela bernama `Masih Pemula`.
   - Program akan terus berjalan hingga tombol `q` ditekan atau gunakan `ctrl + c` pada terminal ketika program masih berjalan.

---

## Instalasi dan Penggunaan
1. **Instalasi Library yang Dibutuhkan:**
   ```bash
   pip install opencv-python
   ```

2. **Jalankan Kode Python:**
   Simpan kode berikut dalam file Python, misalnya `deteksi.py`, lalu jalankan file tersebut:
   ```bash
   python deteksi.py
   ```

---

## Kode Program
```python
import cv2

# Load the pre-trained Haar Cascade Classifier for face, hand, and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize webcam
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip the frame horizontally to prevent mirroring
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale (Haar Cascade requires grayscale image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces and detect smiles within faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_region_gray = gray[y:y + h, x:x + w]
        face_region_color = frame[y:y + h, x:x + w]

        # Detect smiles in the face region
        smiles = smile_cascade.detectMultiScale(face_region_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_region_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

    # Draw rectangles around detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Masih Pemula', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
```

---

## Catatan Penting
1. Pastikan webcam terhubung dan berfungsi dengan baik.
2. Jika OpenCV tidak dapat menemukan file Haar Cascade, pastikan path file sudah benar.
3. Untuk menghentikan program, tekan tombol `q` atau gunakan `ctrl + c` pada terminal ketika program masih berjalan.

---

## Output
Program akan membuka jendela video real-time yang:
- Menampilkan wajah dengan kotak biru.
- Menampilkan mata dengan kotak hijau.
- Menampilkan senyuman dengan kotak kuning.

---

## Troubleshooting
1. **Webcam tidak terdeteksi:**
   - Periksa koneksi webcam.
   - Pastikan webcam tidak digunakan oleh aplikasi lain.

2. **Model tidak terdeteksi:**
   - Periksa lokasi file Haar Cascade.
   - Gunakan versi terbaru dari OpenCV.

---

## Lisensi
Proyek ini menggunakan OpenCV yang berada di bawah lisensi BSD.

---

## Author
- [@irsyamokta](https://github.com/irsyamokta)
