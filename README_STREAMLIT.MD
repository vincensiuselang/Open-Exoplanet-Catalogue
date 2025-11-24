1️⃣ Hasil Prediksi

Bagian ini adalah yang sebenarnya menunjukkan hasil prediksi dari planet yang Anda inputkan:

- Kelas yang diprediksi: menunjukkan kelas planet yang diprediksi.
  - 0: Major/Common (planet umum)
  - 1: Minor/Rare (planet langka, minor class)

- Kepercayaan (Minor Class): nilai antara 0-1 yang menunjukkan kemungkinan planet menjadi planet minor class.
  - Contoh: kepercayaan 0.40 berarti 40% kepercayaan bahwa planet tersebut adalah planet minor class.

Jadi, Hasil Prediksi adalah hal yang sebenarnya "kelas apa yang diprediksi" serta "keberatan model bagaimana".

2️⃣ Gauge Kepercayaan

Ini adalah visualisasi garis horizontal yang menunjukkan kepercayaan dari minor class:

- Panjang bar: jumlah kepercayaan dari minor class (0-1).
- Garis vertikal merah: garis batasan yang digunakan oleh model untuk menentukan kelas.
  - Contoh: garis batasan = 0.5
  - Jika kepercayaan ≥ 0.5, prediksi adalah minor class (1).
  - Jika kepercayaan < 0.5, prediksi adalah major class (0).

Jadi tujuannya adalah membuat mudah untuk melihat keberatan model tanpa perlu melihat angka.

Maka Hasil Prediksi adalah jawaban dari model, dan Gauge Kepercayaan adalah visualisasi dari kepercayaan model terkait kemungkinan.