# Prediksi Daya Energi Surya Berbasis OLS 7-Parameter NASA POWER
### Studi Kasus Wilayah Bontang, Kalimantan Timur (2015–2025)

**Lokasi Studi:** Bontang, Kalimantan Timur (0,1333°N; 117,50°E)  
**Periode Data:** Januari 2015 – Desember 2025 (n = 132 bulan)  
**Metode Utama:** Ordinary Least Squares (OLS) via Normal Equations  
**GitHub:** https://github.com/syauqinzul/nasa-power-solar-prediction-7param-ols

---

## Deskripsi

Proyek ini mengimplementasikan model regresi linier berganda menggunakan pendekatan **Aljabar Linear (OLS — Normal Equations)** untuk memprediksi estimasi daya panel surya harian (kWh/m²/hari) berdasarkan tujuh parameter meteorologi yang bersumber dari **NASA POWER**. Seluruh komputasi dilakukan secara **from-scratch** menggunakan NumPy — tanpa bergantung pada library machine learning tingkat tinggi seperti scikit-learn.

Model ini dikembangkan sebagai bagian dari penelitian akademis yang diterbitkan di jurnal SINTA 2, mencakup:

- Formulasi sistem persamaan linier dalam bentuk matriks **Xᵀ · X · β = Xᵀ · Y**
- Estimasi koefisien OLS melalui **Normal Equations**: `β̂ = (XᵀX)⁻¹ XᵀY`
- Delapan uji diagnostik statistik terintegrasi: t-test, AIC/BIC, VIF, Durbin-Watson, Breusch-Godfrey, model lag, Shapiro-Wilk, dan Box-Cox
- Perbandingan komparatif OLS vs model Machine Learning (Linear Regression, Random Forest, SVR)
- Diskusi epistemologis eksplisit tentang batas interpretasi model deterministik

---

## Struktur Repositori

```
project-root/
├── main.py                                          # Skrip analisis utama
├── POWER_Point_Monthly_20150101_20251231_...csv     # Dataset NASA POWER
├── README.md                                        # Dokumentasi repositori
├── requirements.txt                                 # Dependensi Python
├── .gitignore
└── outputs/                                         # Hasil analisis (di-generate otomatis)
    ├── Gambar1_Panel_Utama_4in1.png
    ├── Gambar2_Autokorelasi_Diagnostik.png
    ├── Gambar3_AIC_BIC_Comparison.png
    ├── Gambar4_BoxCox_Normalitas.png
    ├── Gambar5_Residual_Diagnostic.png
    ├── Gambar6_Heatmap_Korelasi.png
    ├── Gambar7_Pipeline_Flowchart.png
    ├── Gambar8_ML_Comparison.png
    └── hasil_analisis_lengkap.txt
```

---

## Fondasi Matematis

### Definisi Variabel

| Simbol | Parameter NASA POWER | Satuan | Keterangan |
|--------|----------------------|--------|------------|
| x₁ | `ALLSKY_SFC_SW_DWN` | kWh/m²/hari | Irradiansi global permukaan (GHI) — CERES SYN1deg |
| x₂ | `T2M` | °C | Suhu udara 2 meter — MERRA-2 |
| x₃ | `WS10M` | m/s | Kecepatan angin 10 meter — MERRA-2 |
| x₄ | `PS` | kPa | Tekanan permukaan atmosfer — MERRA-2 |
| x₅ ★ | `CLOUD_AMT` | % | Tutupan awan rata-rata — CERES SYN1deg |
| x₆ ★ | `IMERG_PRECTOT` | mm/hari | Curah hujan terkoreksi IMERG |
| x₇ ★ | `ALLSKY_SFC_SW_DNI` | kWh/m²/hari | Irradiansi langsung normal (DNI) — CERES SYN1deg |

★ = Parameter baru yang tidak terdapat pada model-model sebelumnya

### Variabel Target (Y)

Estimasi daya panel surya menggunakan model fisika PV standar dengan koreksi suhu (Skoplaki & Palyvos, 2009):

$$Y = x_1 \times \eta_{STC} \times [1 - \beta_T (x_2 - T_{STC})]$$

**Parameter fisika yang digunakan:**
- η_STC = 0,18 (efisiensi panel pada kondisi standar)
- β_T = 0,004 /°C (koefisien temperatur)
- T_STC = 25,0 °C (suhu referensi STC)

> **Catatan epistemologis:** Y bersifat deterministik — secara matematis merupakan fungsi eksak dari x₁ dan x₂. R² ≈ 1,0 yang diperoleh adalah konsekuensi aljabar dari struktur data ini, bukan indikator prediksi empiris. Keterbatasan ini dibahas secara eksplisit dalam jurnal.

### Model Regresi Linear Berganda (M7)

$$\hat{Y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4 + \beta_5 x_5 + \beta_6 x_6 + \beta_7 x_7 + \varepsilon$$

Estimasi koefisien menggunakan **Normal Equations**:

$$\hat{\beta} = (X^\top X)^{-1} X^\top Y$$

Diselesaikan dengan `numpy.linalg.solve()` untuk stabilitas numeris (tanpa inversi eksplisit).

---

## Metodologi Analisis

### 1. Pengolahan Data

- Membaca data CSV NASA POWER (format wide per bulan) dengan `skiprows=19`
- Pivot dari format wide ke format long dan merge ketujuh parameter
- Menggantikan nilai sentinel `-999` dengan `NaN` dan menghapus baris kosong
- Tidak ditemukan missing value pada ketujuh parameter setelah preprocessing
- **Split data kronologis:** Train 80% (n=106, Jan 2015–Okt 2023) | Test 20% (n=26, Nov 2023–Des 2025)

### 2. Konstruksi Matriks & Estimasi OLS

```python
# Membangun matriks desain X (n × 8) dengan kolom intercept
X = np.column_stack([np.ones(n), x1, x2, x3, x4, x5, x6, x7])

# Normal Equations menggunakan np.linalg.solve (numerik stabil)
XtX = X_train.T @ X_train
XtY = X_train.T @ Y_train
beta_hat = np.linalg.solve(XtX, XtY)
```

### 3. Metrik Evaluasi Model

| Metrik | Formula |
|--------|---------|
| MAE | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ |
| MAPE | $\frac{1}{n}\sum \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$ |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ |
| RMSE | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ |

### 4. Uji Statistik yang Dilakukan (8 Prosedur)

| No. | Uji | Metode | Keterangan |
|-----|-----|--------|------------|
| 1 | Signifikansi koefisien | **t-test** dua arah | df = 98, α = 0,05, nilai kritis |t| ≈ 1,984 |
| 2 | Pemilihan model | **AIC/BIC** | Perbandingan M7 (k=8) vs M2 tereduksi (k=3) |
| 3 | Multikolinearitas | **VIF** | VIF > 10 bermasalah; VIF < 10 aman |
| 4 | Autokorelasi | **Durbin-Watson** | dL=1,6013; dU=1,8216 (n=132, k=7, α=0,05) |
| 5 | Autokorelasi (formal) | **Breusch-Godfrey LM** | AR(1) dan AR(2); statistik LM ~ χ²(p) |
| 6 | Autokorelasi (kovariat) | **Model Lag Y(t-1)** | Menambahkan variabel lag sebagai kovariat |
| 7 | Normalitas residual | **Shapiro-Wilk** | W, p-value, skewness, excess kurtosis |
| 8 | Transformasi | **Box-Cox** | Optimasi λ via log-likelihood; λ=3,0 |

### 5. Perbandingan Model Machine Learning (Validasi Komparatif)

Sebagai validasi tambahan — bukan pengganti model utama OLS — tiga model ML diimplementasikan:

| Model | Konfigurasi |
|-------|-------------|
| Linear Regression | sklearn, baseline |
| Random Forest | n_estimators=200, max_depth=10, random_state=42 |
| SVR | kernel='rbf', C=10, ε=0,001 |

Seluruh model ML menggunakan fitur identik (x₁–x₇) dengan StandardScaler dan train/test split 80:20 yang sama.

---

## Hasil Utama

### Koefisien OLS M7

```
ŷ = 0,1175 + 0,1786·x₁ − 0,003615·x₂ − 2,79×10⁻⁵·x₃
        − 1,94×10⁻⁴·x₄ − 2,64×10⁻⁶·x₅ − 6,21×10⁻⁶·x₆ − 5,42×10⁻⁵·x₇
```

Koefisien x₁ = 0,1786 mendekati η_STC = 0,18; koefisien x₂ = −0,003615 setara −η_STC·β. OLS merekonstruksi secara akurat struktur matematika model fisika PV.

### Performa Model

| Dataset | n | MAE (kWh/m²/hari) | MAPE (%) | R² | RMSE |
|---------|---|-------------------|----------|----|------|
| Training (Jan 2015–Okt 2023) | 106 | 0,00007637 | 0,008726 | 0,99999770 | 0,00010880 |
| Test (Nov 2023–Des 2025) | 26 | 0,00011426 | 0,013653 | 0,99999622 | 0,00015496 |
| Keseluruhan (2015–2025) | 132 | 0,00008383 | 0,009696 | 0,99999744 | 0,00011931 |

### Uji Signifikansi Koefisien

Hanya **x₁ (ALLSKY_SFC_SW_DWN)** dan **x₂ (T2M)** yang signifikan secara statistik (p < 0,001), konsisten dengan model fisika PV: `Ŷ ≈ f(Irradiansi, Suhu)`. Variabel x₃–x₇ tidak signifikan — merupakan konsekuensi deterministik yang dapat diprediksi.

### Perbandingan AIC/BIC

| Model | k | SSE | AIC | BIC |
|-------|---|-----|-----|-----|
| M2: Intercept + x₁ + x₂ | 3 | 2,39×10⁻³ | −2374,20 | −2365,56 |
| M7: Intercept + x₁–x₇ | 8 | 1,88×10⁻³ | −2368,92 | −2345,85 |
| ΔAIC / ΔBIC (M7−M2) | — | — | +5,29 | +19,70 |

ΔBIC = +19,70 (|ΔBIC| > 10) → bukti sangat kuat M2 lebih parsimoni. **Model M2 direkomendasikan untuk aplikasi praktis berbasis data empiris lapangan.**

### Multikolinearitas (VIF)

Seluruh VIF < 10. VIF tertinggi x₇ = 7,65. Penggantian parameter lama dengan CLOUD_AMT, IMERG_PRECTOT, dan ALLSKY_DNI berhasil mengeliminasi multikolinearitas ekstrem model sebelumnya.

### Autokorelasi

| Uji | Hasil | Keputusan |
|-----|-------|-----------|
| Durbin-Watson | DW = 1,6635 | Inconclusive (dL < DW < dU) |
| Breusch-Godfrey AR(1) | LM = 11,267; p = 0,0008 | **TOLAK H₀** — autokorelasi terkonfirmasi |
| Breusch-Godfrey AR(2) | LM = 14,305; p = 0,0008 | **TOLAK H₀** — autokorelasi terkonfirmasi |

Autokorelasi mencerminkan pola musiman iklim tropis (siklus monsun). Model SARIMA/ARIMAX direkomendasikan untuk penelitian lanjutan.

### Normalitas Residual

| Kondisi | W (Shapiro-Wilk) | p-value | Skewness | Keputusan |
|---------|------------------|---------|----------|-----------|
| Sebelum Box-Cox | 0,8989 | 5,6×10⁻⁸ | −1,1478 | TOLAK H₀ |
| Sesudah Box-Cox (λ=3,0) | 0,9428 | 2,9×10⁻⁵ | −0,0664 | TOLAK H₀ |

Ketidaknormalan bersifat struktural. **Robust Regression (Huber/MM-Estimator) direkomendasikan** untuk penelitian lanjutan.

### Perbandingan OLS vs Machine Learning (Test Set, n=26)

| Model | MAE | RMSE | R² (Test) | MAPE (%) |
|-------|-----|------|-----------|----------|
| **OLS M7 ★** | 0,000114 | 0,000155 | **0,99999622** | **0,0137** |
| Linear Regression | 0,000114 | 0,000155 | 0,99999622 | 0,0137 |
| Random Forest | 0,006041 | 0,009361 | 0,98622214 | 0,7769 |
| SVR (RBF) | 0,013914 | 0,022193 | 0,92255715 | 1,7127 |

OLS superior pada data deterministik. Random Forest dan SVR mengalami overfitting parsial pada noise residual deterministik.

---

## Output Visualisasi

| File | Isi |
|------|-----|
| `Gambar1_Panel_Utama_4in1.png` | Deret waktu Y aktual vs prediksi, scatter, residual over time, distribusi residual |
| `Gambar2_Autokorelasi_Diagnostik.png` | ACF residual, DW zone plot, scatter Y(t-1) vs residual, ACF model lag |
| `Gambar3_AIC_BIC_Comparison.png` | Perbandingan AIC/BIC antara M2 dan M7 |
| `Gambar4_BoxCox_Normalitas.png` | Log-likelihood vs λ, histogram sebelum/sesudah, Q-Q plot sebelum/sesudah, perbandingan SW |
| `Gambar5_Residual_Diagnostic.png` | Q-Q plot, residual vs fitted, ACF (train/test dibedakan warna) |
| `Gambar6_Heatmap_Korelasi.png` | Heatmap korelasi Pearson 7 parameter + Y |
| `Gambar7_Pipeline_Flowchart.png` | Diagram alur pipeline penelitian dari akuisisi NASA POWER hingga output |
| `Gambar8_ML_Comparison.png` | Perbandingan performa OLS vs RF vs SVR vs Linear Regression pada test set |
| `hasil_analisis_lengkap.txt` | Ringkasan numerik lengkap seluruh 8 uji statistik |

---

## Cara Menjalankan

### Prasyarat

```bash
pip install -r requirements.txt
```

**Isi `requirements.txt`:**

```
numpy
pandas
matplotlib
scipy
statsmodels
scikit-learn
```

### Eksekusi

```bash
python main.py
```

Seluruh output akan tersimpan otomatis di folder `outputs/`.

### Sumber Data NASA POWER

Data dapat diunduh dari portal NASA POWER:  
👉 https://power.larc.nasa.gov/data-access-viewer/

**Pengaturan unduh:**
- Temporal Resolution: Monthly
- Koordinat: 0,1333°N; 117,50°E (Bontang, Kalimantan Timur)
- Periode: 2015-01-01 s.d. 2025-12-31
- Parameters: `ALLSKY_SFC_SW_DWN`, `T2M`, `WS10M`, `PS`, `CLOUD_AMT`, `IMERG_PRECTOT`, `ALLSKY_SFC_SW_DNI`

---

## Konteks Wilayah

**Bontang, Kalimantan Timur** dipilih sebagai lokasi studi karena:

- Terletak di kawasan ekuatorial (0,13°N) dengan intensitas radiasi surya 4,08–5,80 kWh/m²/hari
- Berdekatan dengan lokasi Ibu Kota Nusantara (IKN), strategis untuk perencanaan energi terbarukan nasional
- Iklim tropis lembab dengan tutupan awan rata-rata 81,74% dan variabilitas monsun signifikan
- Ketersediaan data NASA POWER yang lengkap dan konsisten untuk periode 2015–2025

---

## Referensi Metode

- **NASA POWER:** Prediction Of Worldwide Energy Resources — [power.larc.nasa.gov](https://power.larc.nasa.gov)
- **OLS Normal Equations:** Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.)
- **Model Fisika PV:** Skoplaki, E., & Palyvos, J. A. (2009). Solar Energy, 83(5), 614–624
- **Durbin-Watson:** Savin & White (1977) — Tabel nilai kritis untuk n=132, k=7
- **Breusch-Godfrey:** Breusch (1978), Godfrey (1978)
- **AIC/BIC:** Akaike (1974), Schwarz (1978)
- **VIF:** O'Brien (2007). *A Caution Regarding Rules of Thumb for Variance Inflation Factors*
- **NASA POWER Akurasi:** Lim et al. (2023). Applied Energy, 329, 120329
- **ML vs OLS Review:** Raza et al. (2022). Renewable and Sustainable Energy Reviews, 92
- **Epistemologi Model PV:** Voyant et al. (2022). Solar Energy, 105, 569–582; Ahmed et al. (2024). Energy Conversion and Management, 286

---

## Sitasi

Jika menggunakan kode atau data dalam publikasi, harap sitasi jurnal yang berkaitan:

> [Syauqi Nuzul Abdi]. (2025). *Prediksi Daya Energi Surya Berbasis OLS 7-Parameter NASA POWER: Studi Kasus Wilayah Bontang, Kalimantan Timur (2015–2025)*. []. 
