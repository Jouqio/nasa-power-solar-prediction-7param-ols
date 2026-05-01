Analisis dan Evaluasi Model Prediksi Daya Energi Surya Menggunakan Pendekatan Aljabar Linear Berbasis Data NASA dan Implementasi Python

Lokasi Studi: Bontang, Kalimantan Timur (0.1333°N, 117.50°E)
Periode Data: Januari 2015 – Desember 2025 (n = 132 bulan)
Metode Utama: Ordinary Least Squares (OLS) via Normal Equations


Deskripsi
Proyek ini mengimplementasikan model regresi linier berganda menggunakan pendekatan Aljabar Linear (OLS — Normal Equations) untuk memprediksi estimasi daya panel surya harian (kWh/m²/hari) berdasarkan tujuh parameter meteorologi yang bersumber dari NASA POWER. Seluruh komputasi dilakukan secara from-scratch menggunakan NumPy — tanpa bergantung pada library machine learning tingkat tinggi seperti scikit-learn.
Model ini dikembangkan sebagai bagian dari analisis akademis yang mencakup:

Formulasi sistem persamaan linier dalam bentuk matriks Xᵀ · X · β = Xᵀ · Y
Estimasi koefisien OLS melalui Normal Equations: β̂ = (XᵀX)⁻¹ XᵀY
Evaluasi statistik komprehensif: uji signifikansi, AIC/BIC, VIF, autokorelasi, dan normalitas residual


Struktur Repositori
project-root/
├── main.py                                          # Skrip analisis utama
├── POWER_Point_Monthly_20150101_20251231_...csv     # Dataset NASA POWER
├── requirements.txt                                 # Dependensi Python
├── .gitignore
└── outputs/                                         # Hasil analisis (di-generate otomatis)
    ├── heatmap_korelasi.png
    ├── distribusi_residual.png
    ├── acf_residual.png
    ├── scatter_pred_interval.png
    ├── timeseries_7param.png
    ├── residual_diagnostic.png
    ├── train_test_scatter.png
    ├── AIC_BIC_Comparison.png
    ├── Autokorelasi_Diagnostik.png
    ├── Panel_Utama_4in1.png
    └── hasil_analisis_lengkap.txt

Fondasi Matematis
Definisi Variabel
SimbolParameter NASA POWERSatuanKeteranganx₁ALLSKY_SFC_SW_DWNkWh/m²/hariIrradiansi global (GHI)x₂T2M°CSuhu udara 2 meterx₃WS10Mm/sKecepatan angin 10 meterx₄PSkPaTekanan permukaanx₅CLOUD_AMT%Tutupan awanx₆IMERG_PRECTOTmm/hariCurah hujanx₇ALLSKY_SFC_SW_DNIkWh/m²/hariIrradiansi langsung (DNI)
Variabel Target (Y)
Estimasi daya panel surya menggunakan model fisika PV:
Y=x1×ηSTC×[1−βT(x2−TSTC)]Y = x_1 \times \eta_{STC} \times [1 - \beta_T (x_2 - T_{STC})]Y=x1​×ηSTC​×[1−βT​(x2​−TSTC​)]
Parameter fisika yang digunakan:

η_STC = 0,18 (efisiensi panel pada kondisi standar)
β_T = 0,004 /°C (koefisien temperatur)
T_STC = 25,0 °C (suhu referensi STC)

Model Regresi Linear Berganda (M7)
Y^=β0+β1x1+β2x2+β3x3+β4x4+β5x5+β6x6+β7x7+ε\hat{Y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4 + \beta_5 x_5 + \beta_6 x_6 + \beta_7 x_7 + \varepsilonY^=β0​+β1​x1​+β2​x2​+β3​x3​+β4​x4​+β5​x5​+β6​x6​+β7​x7​+ε
Estimasi koefisien menggunakan Normal Equations:
β^=(X⊤X)−1X⊤Y\hat{\beta} = (X^\top X)^{-1} X^\top Yβ^​=(X⊤X)−1X⊤Y

Metodologi Analisis
1. Pengolahan Data

Membaca data CSV NASA POWER (format wide per bulan) dengan skiprows=19
Melakukan pivot dan merge ketujuh parameter
Menggantikan nilai sentinel -999 dengan NaN dan menghapus baris kosong
Split data: Train 80% (n=106, Jan 2015–Okt 2023) | Test 20% (n=26)

2. Konstruksi Matriks & Estimasi OLS
python# Membangun matriks desain X (n × 8) dengan kolom intercept
X = np.column_stack([np.ones(n), x1, x2, x3, x4, x5, x6, x7])

# Normal Equations menggunakan np.linalg.solve (numerik stabil)
XtX = X_train.T @ X_train
XtY = X_train.T @ Y_train
beta_hat = np.linalg.solve(XtX, XtY)
3. Metrik Evaluasi Model
MetrikFormulaMAE$\frac{1}{n}\sumMAPE$\frac{1}{n}\sum \leftR²1−SSresSStot1 - \frac{SS_{res}}{SS_{tot}}
1−SStot​SSres​​RMSE1n∑(yi−y^i)2\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}
n1​∑(yi​−y^​i​)2​
4. Uji Statistik yang Dilakukan
§ 3.5 — Uji Signifikansi Koefisien & AIC/BIC

t-test per koefisien (df = 98, α = 0,05, nilai kritis |t| ≈ 1,984)
Standar error dari matriks kovarians: Cov(β̂) = MSE · (XᵀX)⁻¹
Perbandingan AIC dan BIC antara Model M7 (7 variabel) dan Model Reduksi M2 (x₁ + x₂):

AIC=n⋅ln⁡(SSEn)+2kBIC=n⋅ln⁡(SSEn)+k⋅ln⁡(n)AIC = n \cdot \ln\left(\frac{SSE}{n}\right) + 2k \qquad BIC = n \cdot \ln\left(\frac{SSE}{n}\right) + k \cdot \ln(n)AIC=n⋅ln(nSSE​)+2kBIC=n⋅ln(nSSE​)+k⋅ln(n)
§ 3.6 — Variance Inflation Factor (VIF)
Deteksi multikolinearitas antar variabel bebas:
VIFi=11−Ri2VIF_i = \frac{1}{1 - R^2_i}VIFi​=1−Ri2​1​
VIF > 10 → multikolinearitas tinggi | VIF 5–10 → perlu perhatian | VIF < 5 → aman
§ 3.7 — Uji Autokorelasi (Tiga Pendekatan)
PendekatanMetodeKeteranganADurbin-WatsonUji autokorelasi AR(1); dL=1,6013, dU=1,8216 (n=132, k=7, α=0,05)BBreusch-Godfrey LMUji formal hingga orde AR(2); statistik LM ~ χ²(p)CModel Lag Y_(t-1)Menambahkan variabel lag sebagai kovariat; evaluasi koefisien γDARIMA/ARIMAXPemodelan struktur autokorelasi sisa (jika statsmodels tersedia)
§ 3.8 — Normalitas Residual

Shapiro-Wilk test (W, p-value)
Analisis skewness dan excess kurtosis
Visualisasi: Q-Q Plot dan histogram distribusi residual


Output Visualisasi
FileIsiGambar1_Panel_Utama_4in1.pngPanel 4-in-1: prediksi vs aktual (train/test), residual over time, distribusi residual, scatter plotGambar_AIC_BIC_Comparison.pngPerbandingan AIC/BIC antara M2 dan M7Gambar_Autokorelasi_Diagnostik.png4 panel: ACF residual, DW zone plot, scatter Y_(t-1) vs residual, ACF model lagGambar_Heatmap_Korelasi.pngHeatmap korelasi Pearson 7 parameter + YGambar_Residual_Diagnostic.png3-in-1: Q-Q plot, residual vs fitted, ACFhasil_analisis_lengkap.txtRingkasan numerik lengkap seluruh uji statistik

Cara Menjalankan
Prasyarat
bashpip install -r requirements.txt
Isi requirements.txt:
numpy
pandas
matplotlib
scipy
statsmodels
Eksekusi
bashpython main.py
Seluruh output akan tersimpan otomatis di folder outputs/.
Sumber Data NASA POWER
Data dapat diunduh dari portal NASA POWER:
https://power.larc.nasa.gov/data-access-viewer/
Pengaturan unduh:

Temporal Resolution: Monthly
Koordinat: 0.1333°N, 117.50°E (Bontang, Kaltim)
Periode: 2015-01-01 s.d. 2025-12-31
Parameters: ALLSKY_SFC_SW_DWN, T2M, WS10M, PS, CLOUD_AMT, IMERG_PRECTOT, ALLSKY_SFC_SW_DNI


Ringkasan Hasil
Performa Model M7
MetrikTraining (n=106)Test (n=26)Keseluruhan (n=132)MAE——sangat rendahMAPE——< 1%R²≈ 1,000≈ 1,000≈ 1,000RMSE——sangat rendah

Nilai eksak tersedia di outputs/hasil_analisis_lengkap.txt setelah menjalankan main.py.

Perbandingan Model
ModelVariabelkKeteranganM7 ★x₁ s.d. x₇ + intercept8Model utama penelitianM2x₁ + x₂ + intercept3Model reduksi/parsimoni

Model M7 digunakan sebagai model utama. Berdasarkan analisis AIC/BIC, M2 direkomendasikan untuk aplikasi lapangan karena lebih parsimoni dengan hanya mengandalkan x₁ (irradiansi) dan x₂ (suhu) yang keduanya signifikan secara statistik.

Temuan Uji Signifikansi
Hanya x₁ (ALLSKY_SFC_SW_DWN) dan x₂ (T2M) yang signifikan secara statistik (p < 0,05), konsisten dengan model fisika PV:
Y^≈f(Irradiansi, Suhu)\hat{Y} \approx f(\text{Irradiansi},\ \text{Suhu})Y^≈f(Irradiansi, Suhu)

Konteks Wilayah
Bontang, Kalimantan Timur dipilih sebagai lokasi studi karena:

Terletak di kawasan ekuatorial dengan intensitas radiasi surya tinggi dan relatif stabil sepanjang tahun
Merupakan wilayah dengan potensi pengembangan energi surya yang signifikan di Pulau Kalimantan
Ketersediaan data NASA POWER yang lengkap dan konsisten untuk periode 2015–2025


Referensi Metode

NASA POWER: Prediction Of Worldwide Energy Resources — power.larc.nasa.gov
OLS Normal Equations: Strang, G. (2016). Introduction to Linear Algebra (5th ed.)
Durbin-Watson: Savin & White (1977) — Tabel nilai kritis untuk n=132, k=7
Breusch-Godfrey: Breusch (1978), Godfrey (1978)
AIC/BIC: Akaike (1974), Schwarz (1978)
VIF: O'Brien (2007) — A Caution Regarding Rules of Thumb for Variance Inflation Factors

