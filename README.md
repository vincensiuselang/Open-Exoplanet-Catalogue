# Proyek Klasifikasi Planet

## Ikhtisar
Proyek ini bertujuan untuk **mengklasifikasikan planet** berdasarkan karakteristiknya menjadi dua kategori:

- **Kelas 0:** Jenis planet yang paling umum
- **Kelas 1:** Jenis planet yang langka atau khusus

Kumpulan data ini berisi fitur-fitur seperti massa planet, radius, parameter orbit, dan karakteristik bintang induk. Karena **sifat data yang sangat tidak seimbang** (kelas minor ~5%), teknik khusus diterapkan untuk meningkatkan deteksi planet langka.

---

## Dataset
- Kolom meliputi:
- `PlanetaryMassJpt` – Massa planet dalam massa Jupiter
- `RadiusJpt` – Jari-jari planet dalam jari-jari Jupiter
- `PeriodDays` – Periode orbit dalam hari
- `SemiMajorAxisAU` – Sumbu semi-mayor dalam AU
- `Eksentrisitas` – Eksentrisitas orbit
- `HostStarMassSlrMass` – Massa bintang induk dalam massa Matahari
- `HostStarRadiusSlrRad` – Jari-jari bintang induk dalam jari-jari Matahari
- `TypeFlag` – Kelas kategorikal asli

- **Variabel target:** `TargetBinary`
- 0 → asli `TypeFlag = 0`
- 1 → asli `TypeFlag != 0`

- **Ketidakseimbangan:** ~2717 kelas mayor vs 150 kelas minor (sebelum SMOTE)

---

## Rekayasa Fitur
Beberapa fitur turunan dibuat untuk meningkatkan kinerja model:

- `EccentricitySquared` – kuadrat eksentrisitas
- `OrbitalEnergy` – perkiraan energi orbital
- `SemiMajorAxisLog` – transformasi logaritma dari sumbu semi-mayor
- `ScaledPeriod` – periode ternormalisasi
- `RadiusSqrt` / `MassSqrt` – transformasi akar kuadrat
- `MassRadiusRatio`, `DensityApprox` – diturunkan dari massa dan radius
- `PeriodLog` – periode tertransformasi logaritma

Fitur-fitur ini penting untuk **menyoroti pola** planet kelas minor.

---

## Metodologi
1. **Pra-pemrosesan Data**
- Menangani nilai yang hilang
- Membagi dataset menjadi pelatihan (80%) dan pengujian (20%)

2. **Penanganan Data Tidak Seimbang**
- Menerapkan **SMOTE** untuk melakukan oversampling kelas minor dalam set pelatihan

3. **Model**
- **CatBoostClassifier**
- Parameter:
- `iterasi=500`
- `kedalaman=5`
- `laju_pembelajaran=0,05`
- `subsampel=0,8`
- `colsample_bylevel=0,8`

4. **Penyetelan Ambang Batas**
- Ambang batas probabilitas untuk memprediksi kelas minor yang dioptimalkan untuk **skor F1 kelas 1**

5. **Metrik Evaluasi**
- Presisi, Perolehan, Skor F1
- Makro F1
- Akurasi Seimbang
- Matriks Kebingungan

---

## Hasil
- **Ambang batas terbaik (kelas minor):** 0,33
- **Skor F1 kelas minor:** 0,34
- **Ingatan kelas minor:** 0,66
- **Presisi/Ingatan kelas mayor:** 0,98 / 0,87
- **Akurasi Seimbang:** 0,77

- **Fitur utama (kepentingan):**
1. `EccentricitySquared`
2. `Eccentricity`
3. `OrbitalEnergy`
4. `SemiMajorAxisAU`
5. `HostStarMassSlrMass`

> Model ini menunjukkan **deteksi planet langka yang efektif** meskipun terdapat ketidakseimbangan yang tinggi, dengan tingkat kepentingan fitur yang dapat diinterpretasikan.

---

## Kesimpulan
Proyek ini menunjukkan:

- Menangani **himpunan data tidak seimbang** menggunakan SMOTE
- Pentingnya **rekayasa fitur** untuk meningkatkan deteksi kelas minor
- **Penyetelan ambang batas** untuk mengoptimalkan perolehan kembali kelas minor
- Penggunaan **CatBoostClassifier** untuk data tabular dengan interpretabilitas yang baik

Alur kerja ini dapat diadaptasi untuk **masalah klasifikasi tidak seimbang** lainnya dalam astronomi atau himpunan data tabular umum.

---

## Langkah Selanjutnya / Peningkatan Opsional
- Visualisasikan **matriks kebingungan** dan **pentingnya fitur**
- Coba **kehilangan fokus** atau metode peningkatan lainnya untuk lebih meningkatkan presisi kelas minor
- Validasi silang untuk **evaluasi yang lebih kuat**
