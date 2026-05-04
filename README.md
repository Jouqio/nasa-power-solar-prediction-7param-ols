# Prediksi Daya Energi Surya — OLS 7-Parameter NASA POWER
### Bontang, Kalimantan Timur (2015–2025)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat&logo=creativecommons&logoColor=white)
![SINTA](https://img.shields.io/badge/Jurnal-SINTA%202-blue?style=flat)
![NASA POWER](https://img.shields.io/badge/Data-NASA%20POWER-E03C31?style=flat&logo=nasa&logoColor=white)

Implementasi model **OLS (Normal Equations)** untuk memprediksi daya panel surya harian berbasis 7 parameter meteorologi NASA POWER. Dibangun *from-scratch* dengan NumPy, dievaluasi dengan 8 uji statistik, dan dibandingkan dengan model Machine Learning.

---

## Hasil

| Dataset | R² | MAPE |
|---------|----|------|
| Training (n=106) | 0,99999770 | 0,0087% |
| Test (n=26) | 0,99999622 | 0,0137% |

> R² ≈ 1,0 adalah konsekuensi matematis dari variabel target yang deterministik, bukan indikator prediksi empiris murni.

---

## Cara Pakai

```bash
pip install -r requirements.txt
python main.py
```

Data: [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/) — koordinat `0,1333°N; 117,50°E`, resolusi Monthly, periode 2015–2025.

---

## Sitasi

```bibtex
@article{[namaauthor]2025solar,
  title   = {Prediksi Daya Energi Surya Berbasis OLS 7-Parameter NASA POWER},
  author  = {[Nama Penulis] and [Nama Co-author]},
  journal = {[Nama Jurnal SINTA 2]},
  year    = {2025},
  doi     = {[doi]}
}
```

[CC BY 4.0](LICENSE) — bebas digunakan dengan wajib sitasi.
