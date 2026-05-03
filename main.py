"""
=======================================================================
ANALISIS MODEL PREDIKSI DAYA ENERGI SURYA - 7 PARAMETER
Bontang, Kalimantan Timur (0.1333°N, 117.50°E)
Data: NASA POWER Monthly 2015-2025 (n=132)
Metode: OLS Aljabar Linear (Normal Equations)

VERSI FINAL — mencakup seluruh uji statistik:
  § 3.5  Uji Signifikansi Koefisien + Analisis AIC/BIC + Reduksi Model M2
  § 3.7  Uji Autokorelasi: Durbin-Watson + Breusch-Godfrey +
          Variabel Lag + ARIMA/ARIMAX
  § 3.8  Normalitas Residual + Transformasi Box-Cox

7 Variabel Bebas:
  x1 = ALLSKY_SFC_SW_DWN  (Irradiansi global, kWh/m²/hari)
  x2 = T2M                (Suhu 2 meter, °C)
  x3 = WS10M              (Kecepatan angin 10m, m/s)
  x4 = PS                 (Tekanan permukaan, kPa)
  x5 = CLOUD_AMT          (Tutupan awan, %)
  x6 = IMERG_PRECTOT      (Curah hujan, mm/hari)
  x7 = ALLSKY_SFC_SW_DNI  (Irradiansi langsung, kWh/m²/hari)

Target Y: Estimasi daya panel surya (kWh/m²/hari)
  Y = x1 × η_STC × [1 − β(T2M − T_STC)]
=======================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import shapiro, norm, chi2, boxcox
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# ============================================================
# 1. BACA DATA CSV NASA POWER
# ============================================================
CSV_FILE = "POWER_Point_Monthly_20150101_20251231_000d13N_117d50E_UTC.csv"

df_raw = pd.read_csv(CSV_FILE, skiprows=19)
df_raw.columns = ['PARAMETER', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANN']

MONTHS        = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
PARAMS_NEEDED = ['ALLSKY_SFC_SW_DWN', 'T2M', 'WS10M', 'PS', 'CLOUD_AMT',
                 'IMERG_PRECTOT', 'ALLSKY_SFC_SW_DNI']


def pivot_param(df, param_name):
    sub  = df[df['PARAMETER'] == param_name][['YEAR'] + MONTHS].copy()
    sub  = sub.sort_values('YEAR')
    vals = []
    for _, row in sub.iterrows():
        for m in MONTHS:
            vals.append({'YEAR':       int(row['YEAR']),
                         'MONTH':      MONTHS.index(m) + 1,
                         param_name:   float(row[m])})
    return pd.DataFrame(vals)


df_list   = [pivot_param(df_raw, p) for p in PARAMS_NEEDED]
df_merged = reduce(lambda a, b: pd.merge(a, b, on=['YEAR', 'MONTH']), df_list)
df_merged = df_merged.sort_values(['YEAR', 'MONTH']).reset_index(drop=True)
df_merged.replace(-999, np.nan, inplace=True)
df_merged.dropna(inplace=True)

# Gunakan seluruh data 2015-2025 (n=132)
df_merged = df_merged[df_merged['YEAR'] <= 2025].reset_index(drop=True)

print(f"Total data  : {len(df_merged)} baris")
print(f"Periode     : {int(df_merged['YEAR'].min())}-"
      f"{int(df_merged['MONTH'].iloc[0]):02d} s/d "
      f"{int(df_merged['YEAR'].max())}-"
      f"{int(df_merged['MONTH'].iloc[-1]):02d}")

# ============================================================
# 2. DEFINISI VARIABEL & HITUNG Y
# ============================================================
eta_STC = 0.18
beta_T  = 0.004
T_STC   = 25.0

x1 = df_merged['ALLSKY_SFC_SW_DWN'].values
x2 = df_merged['T2M'].values
x3 = df_merged['WS10M'].values
x4 = df_merged['PS'].values
x5 = df_merged['CLOUD_AMT'].values
x6 = df_merged['IMERG_PRECTOT'].values
x7 = df_merged['ALLSKY_SFC_SW_DNI'].values

Y = x1 * eta_STC * (1 - beta_T * (x2 - T_STC))

# ============================================================
# 3. MATRIKS X — MODEL 7 VARIABEL (M7) dengan intercept
# ============================================================
n = len(Y)
X = np.column_stack([np.ones(n), x1, x2, x3, x4, x5, x6, x7])

feature_names = ['Intercept', 'x1(ALLSKY_DWN)', 'x2(T2M)', 'x3(WS10M)',
                 'x4(PS)', 'x5(CLOUD_AMT)', 'x6(IMERG_PRECTOT)', 'x7(ALLSKY_DNI)']
var_names     = ['x1(ALLSKY_DWN)', 'x2(T2M)', 'x3(WS10M)',
                 'x4(PS)', 'x5(CLOUD_AMT)', 'x6(IMERG_PRECTOT)', 'x7(ALLSKY_DNI)']

# ============================================================
# 4. TRAIN/TEST SPLIT 80:20  → 106 train | 26 test
# ============================================================
n_train = 106   # Jan 2015 – Okt 2023
n_test  = n - n_train
print(f"Train: {n_train} bulan  |  Test: {n_test} bulan  |  Total: {n} bulan")

X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# ============================================================
# 5. OLS M7: β̂ = (XᵀX)⁻¹ XᵀY
# ============================================================
XtX_train = X_train.T @ X_train
XtY_train = X_train.T @ Y_train
beta_hat  = np.linalg.solve(XtX_train, XtY_train)

Y_pred_train = X_train @ beta_hat
Y_pred_test  = X_test  @ beta_hat
Y_pred_all   = X       @ beta_hat

residuals_train = Y_train - Y_pred_train
residuals_test  = Y_test  - Y_pred_test
residuals_all   = Y       - Y_pred_all

# ============================================================
# 6. METRIK EVALUASI
# ============================================================
def compute_metrics(y_true, y_pred, label=''):
    mae    = np.mean(np.abs(y_true - y_pred))
    mape   = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = 1 - ss_res / ss_tot
    rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if label:
        print(f"\n--- Metrik {label} ---")
        print(f"  MAE  : {mae:.8f}")
        print(f"  MAPE : {mape:.7f}%")
        print(f"  R²   : {r2:.8f}")
        print(f"  RMSE : {rmse:.8f}")
    return mae, mape, r2, rmse


metrics_train = compute_metrics(Y_train, Y_pred_train, "TRAINING (n=106)")
metrics_test  = compute_metrics(Y_test,  Y_pred_test,  "TEST (n=26)")
metrics_all   = compute_metrics(Y,       Y_pred_all,   "KESELURUHAN (n=132)")

# ============================================================
# 7. UJI SIGNIFIKANSI KOEFISIEN (t-test)  ←  § 3.5
# ============================================================
p_ols    = X_train.shape[1]           # 8 (intercept + 7 variabel)
dof      = n_train - p_ols            # 98
mse_train = np.sum(residuals_train ** 2) / dof
cov_beta  = mse_train * np.linalg.inv(XtX_train)
se_beta   = np.sqrt(np.diag(cov_beta))
t_stat    = beta_hat / se_beta
p_values  = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=dof))
t_crit    = stats.t.ppf(0.975, df=dof)   # ≈ 1.984 untuk df=98

print(f"\n{'='*75}")
print("§ 3.5  UJI SIGNIFIKANSI KOEFISIEN (t-test, df=98, α=0.05)")
print(f"       Nilai kritis |t| = {t_crit:.4f}")
print(f"{'='*75}")
print(f"{'Parameter':<22} {'Koefisien':>12} {'SE':>12} {'t-stat':>10} {'p-value':>10} {'Sig':>5}")
print("-"*75)
for i, name in enumerate(feature_names):
    sig = ("***" if p_values[i] < 0.001 else
           "**"  if p_values[i] < 0.01  else
           "*"   if p_values[i] < 0.05  else "n.s.")
    print(f"{name:<22} {beta_hat[i]:>12.6f} {se_beta[i]:>12.6f} "
          f"{t_stat[i]:>10.4f} {p_values[i]:>10.4f} {sig:>5}")
print("Signifikansi: *** p<0.001  ** p<0.01  * p<0.05  n.s. = tidak signifikan")
sig_vars = [feature_names[i] for i in range(len(feature_names)) if p_values[i] < 0.05]
ns_vars  = [feature_names[i] for i in range(len(feature_names)) if p_values[i] >= 0.05]
print(f"\n  Signifikan (p<0.05)  : {', '.join(sig_vars)}")
print(f"  Tidak signifikan     : {', '.join(ns_vars)}")
print("  → Hanya x1 dan x2 yang memiliki pengaruh statistik nyata terhadap Y.")
print("    Konsisten dengan model fisika PV: Y = f(radiasi, suhu) deterministik.")

# ============================================================
# 7b. ANALISIS AIC/BIC + REDUKSI MODEL M2  ←  § 3.5 (lanjutan)
# ============================================================
def compute_aic_bic(residuals, n_obs, k_params):
    """
    AIC = n·ln(SSE/n) + 2k
    BIC = n·ln(SSE/n) + k·ln(n)
    k   = jumlah parameter (termasuk intercept)
    """
    sse = np.sum(residuals ** 2)
    aic = n_obs * np.log(sse / n_obs) + 2 * k_params
    bic = n_obs * np.log(sse / n_obs) + k_params * np.log(n_obs)
    return sse, aic, bic


# Model M7 (7 variabel, k=8)
k_M7 = 8
sse_M7, aic_M7, bic_M7 = compute_aic_bic(residuals_all, n, k_M7)

# Model M2 (x1 + x2, k=3) — fit OLS baru
X_M2       = np.column_stack([np.ones(n), x1, x2])
X_M2_train = X_M2[:n_train]
beta_M2    = np.linalg.solve(X_M2_train.T @ X_M2_train,
                              X_M2_train.T @ Y_train)
Y_pred_M2    = X_M2 @ beta_M2
residuals_M2 = Y - Y_pred_M2
k_M2         = 3

sse_M2, aic_M2, bic_M2 = compute_aic_bic(residuals_M2, n, k_M2)

mae_M2  = np.mean(np.abs(residuals_M2))
mape_M2 = np.mean(np.abs(residuals_M2 / Y)) * 100
r2_M2   = 1 - sse_M2 / np.sum((Y - np.mean(Y)) ** 2)
rmse_M2 = np.sqrt(np.mean(residuals_M2 ** 2))

delta_aic = aic_M7 - aic_M2
delta_bic = bic_M7 - bic_M2

print(f"\n{'='*75}")
print("§ 3.5  ANALISIS AIC/BIC — M7 vs M2 (2 variabel inti: x1 + x2)")
print(f"{'='*75}")
print(f"{'Model':<28} {'k':>4} {'SSE':>14} {'AIC':>10} {'BIC':>10}")
print("-"*70)
print(f"{'M2: Intercept+x1+x2':<28} {k_M2:>4} {sse_M2:>14.8f} {aic_M2:>10.2f} {bic_M2:>10.2f}")
print(f"{'M7: Intercept+x1..x7 ★':<28} {k_M7:>4} {sse_M7:>14.8f} {aic_M7:>10.2f} {bic_M7:>10.2f}")
print("-"*70)
print(f"{'ΔAIC (M7 − M2)':<28} {delta_aic:>38.2f}")
print(f"{'ΔBIC (M7 − M2)':<28} {delta_bic:>49.2f}")
print(f"\n★ = Model yang digunakan dalam penelitian ini")
print(f"\nMetrik M2 (n=132): MAE={mae_M2:.8f}  MAPE={mape_M2:.6f}%  "
      f"R²={r2_M2:.8f}  RMSE={rmse_M2:.8f}")
print(f"Metrik M7 (n=132): MAE={metrics_all[0]:.8f}  MAPE={metrics_all[1]:.6f}%  "
      f"R²={metrics_all[2]:.8f}  RMSE={metrics_all[3]:.8f}")
print(f"\nPenalti BIC per parameter: ln({n}) = {np.log(n):.4f} >> penalti AIC = 2")
print(f"Koefisien M2: α={beta_M2[0]:.6f}, a(x1)={beta_M2[1]:.6f}, b(x2)={beta_M2[2]:.6f}")

# ============================================================
# 8. VIF (Variance Inflation Factor)
# ============================================================
X_vars    = X_train[:, 1:]   # tanpa intercept
k_vif     = X_vars.shape[1]
vif_values = []
for i in range(k_vif):
    y_i    = X_vars[:, i]
    X_i    = np.column_stack([np.ones(n_train), np.delete(X_vars, i, axis=1)])
    b_i    = np.linalg.solve(X_i.T @ X_i, X_i.T @ y_i)
    y_hat  = X_i @ b_i
    ss_res = np.sum((y_i - y_hat) ** 2)
    ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
    r2_i   = 1 - ss_res / ss_tot
    vif_values.append(1 / (1 - r2_i) if r2_i < 1 else np.inf)

print(f"\n{'='*55}")
print("§ 3.6  VIF (Variance Inflation Factor)")
print(f"{'='*55}")
for name, vif in zip(var_names, vif_values):
    flag = (" ⚠ TINGGI >10"  if vif > 10 else
            " ⚠ Perhatikan" if vif > 5  else " ✓ Aman")
    print(f"  {name:<22} VIF = {vif:.4f}{flag}")

# ============================================================
# 9. DURBIN-WATSON  ←  § 3.7
# ============================================================
res = residuals_all   # digunakan untuk semua uji autokorelasi

dw_num = np.sum(np.diff(res) ** 2)
dw_den = np.sum(res ** 2)
DW     = dw_num / dw_den

# Nilai kritis DW untuk n=132, k=7, α=0.05 (tabel Savin-White)
dL, dU = 1.6013, 1.8216

if DW < dL:
    dw_verdict = "Autokorelasi positif terdeteksi (DW < dL)"
elif DW > (4 - dL):
    dw_verdict = "Autokorelasi negatif terdeteksi (DW > 4-dL)"
elif DW < dU:
    dw_verdict = f"INCONCLUSIVE — zona abu-abu (dL={dL} < DW={DW:.4f} < dU={dU})"
elif DW > (4 - dU):
    dw_verdict = "Inconclusive autokorelasi negatif"
else:
    dw_verdict = "Tidak ada autokorelasi (dU < DW < 4-dU)"

print(f"\n{'='*65}")
print("§ 3.7  UJI AUTOKORELASI")
print(f"{'='*65}")
print(f"\n[A] DURBIN-WATSON")
print(f"  DW          = {DW:.4f}")
print(f"  dL (n=132, k=7, α=0.05) = {dL}  |  dU = {dU}")
print(f"  Interpretasi: {dw_verdict}")

# ============================================================
# 9b. UJI BREUSCH-GODFREY (LM Test)  ←  § 3.7 Pendekatan 1
# ============================================================
def breusch_godfrey_test(residuals, X_full, nlags=2):
    """
    Uji Breusch-Godfrey hingga orde nlags.
    H0: tidak ada autokorelasi hingga orde p.
    LM = (n - p) * R²_aux  ~  χ²(p) di bawah H0.
    """
    n_obs   = len(residuals)
    results = {}
    for p in range(1, nlags + 1):
        lag_matrix = np.zeros((n_obs, p))
        for j in range(1, p + 1):
            lag_matrix[j:, j - 1] = residuals[:n_obs - j]
        X_aux  = np.column_stack([X_full[p:], lag_matrix[p:]])
        e_aux  = residuals[p:]
        b_aux     = np.linalg.lstsq(X_aux, e_aux, rcond=None)[0]
        e_hat_aux = e_aux - X_aux @ b_aux
        ss_res_aux = np.sum(e_hat_aux ** 2)
        ss_tot_aux = np.sum((e_aux - e_aux.mean()) ** 2)
        R2_aux    = 1 - ss_res_aux / ss_tot_aux if ss_tot_aux > 0 else 0
        n_eff     = n_obs - p
        LM_stat   = n_eff * R2_aux
        p_val     = 1 - chi2.cdf(LM_stat, df=p)
        results[p] = {'LM': LM_stat, 'p_value': p_val, 'R2_aux': R2_aux, 'n_eff': n_eff}
    return results


bg_results = breusch_godfrey_test(residuals_all, X, nlags=2)

print(f"\n[B] UJI BREUSCH-GODFREY (LM Test, H0: tidak ada autokorelasi)")
print(f"  {'Orde (p)':<12} {'LM Statistic':>14} {'df':>5} {'p-value':>12} {'R²_aux':>10} {'Keputusan':>20}")
print(f"  {'-'*75}")
for p, res_bg in bg_results.items():
    keputusan = "Tolak H0 ✗" if res_bg['p_value'] < 0.05 else "Gagal Tolak H0 ✓"
    print(f"  AR({p}){'':<8} {res_bg['LM']:>14.4f} {p:>5} {res_bg['p_value']:>12.4f} "
          f"{res_bg['R2_aux']:>10.6f} {keputusan:>20}")

bg_p1 = bg_results[1]['p_value']
bg_p2 = bg_results[2]['p_value']
bg_verdict = ("Autokorelasi TERKONFIRMASI — remediasi diperlukan"
              if bg_p1 < 0.05 or bg_p2 < 0.05
              else "Autokorelasi TIDAK terkonfirmasi — model OLS dapat dipertahankan")
print(f"\n  Kesimpulan BG: {bg_verdict}")

# ============================================================
# 9c. MODEL LAG — Y_(t-1) sebagai variabel tambahan  ←  § 3.7 Pendekatan 2
# ============================================================
print(f"\n[C] MODEL LAG — menambahkan Y_{{t-1}} sebagai kovariat (n_eff = {n-1})")

Y_lag      = Y[:-1]            # Y_(t-1)
X_lag_full = X[1:]             # X pada bulan t
Y_lag_full = Y[1:]             # Y pada bulan t

n_train_lag  = n_train - 1    # 105
X_lag_train  = np.column_stack([X_lag_full[:n_train_lag], Y_lag[:n_train_lag]])
Y_lag_train  = Y_lag_full[:n_train_lag]

beta_lag        = np.linalg.lstsq(X_lag_train, Y_lag_train, rcond=None)[0]
Y_pred_lag_all  = np.column_stack([X_lag_full, Y_lag]) @ beta_lag
residuals_lag   = Y_lag_full - Y_pred_lag_all

DW_lag = (np.sum(np.diff(residuals_lag) ** 2) /
          np.sum(residuals_lag ** 2))

bg_lag = breusch_godfrey_test(residuals_lag,
                               np.column_stack([X_lag_full, Y_lag]),
                               nlags=1)

mae_lag  = np.mean(np.abs(residuals_lag))
mape_lag = np.mean(np.abs(residuals_lag / Y_lag_full)) * 100
r2_lag   = 1 - np.sum(residuals_lag**2) / np.sum((Y_lag_full - Y_lag_full.mean())**2)
rmse_lag = np.sqrt(np.mean(residuals_lag**2))
gamma_lag = beta_lag[-1]

print(f"  Koefisien Y_{{t-1}} (γ)  = {gamma_lag:.6f}")
print(f"  DW model lag           = {DW_lag:.4f}")
print(f"  BG AR(1) p-value       = {bg_lag[1]['p_value']:.4f}")
print(f"  MAE  = {mae_lag:.8f}  |  R² = {r2_lag:.8f}")
print(f"  MAPE = {mape_lag:.6f}%   |  RMSE = {rmse_lag:.8f}")

if DW_lag >= dU and bg_lag[1]['p_value'] >= 0.05:
    lag_verdict = "Autokorelasi teratasi setelah penambahan Y_(t-1) ✓"
elif DW_lag >= dL:
    lag_verdict = "Autokorelasi berkurang namun masih perlu evaluasi lebih lanjut"
else:
    lag_verdict = "Autokorelasi masih terdeteksi — pertimbangkan ARIMA/SARIMA"
print(f"  Interpretasi: {lag_verdict}")

# ============================================================
# 9d. ARIMA / ARIMAX  ←  § 3.7 Pendekatan 3
# ============================================================
print(f"\n[D] ARIMA / ARIMAX — pemodelan struktur autokorelasi sisa")
arimax_available = False
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey as sm_bg
    from statsmodels.tsa.stattools import adfuller

    adf_stat, adf_p, adf_lags, adf_nobs, adf_cv, _ = adfuller(Y, autolag='AIC')
    print(f"\n  ADF Test pada Y (uji stasionaritas):")
    print(f"    ADF Statistic = {adf_stat:.4f}  |  p-value = {adf_p:.4f}")
    for key, cv_val in adf_cv.items():
        print(f"    Nilai Kritis {key} = {cv_val:.4f}")
    d_order = 0 if adf_p < 0.05 else 1
    print(f"    → d = {d_order} ({'stasioner' if d_order == 0 else 'tidak stasioner, perlu differencing'})")

    exog_cols   = np.column_stack([x1, x2])
    arimax_order = (1, d_order, 1)

    model_arimax = SARIMAX(Y,
                           exog=exog_cols,
                           order=arimax_order,
                           trend='c',
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    fit_arimax = model_arimax.fit(disp=False)

    res_arimax = fit_arimax.resid
    DW_arimax  = (np.sum(np.diff(res_arimax)**2) /
                  np.sum(res_arimax**2))
    bg_arimax_lm, bg_arimax_p, _, _ = sm_bg(fit_arimax, nlags=2, store=False)

    print(f"\n  ARIMAX{arimax_order} dengan exog=[x1, x2]:")
    print(f"  {fit_arimax.summary().tables[1].as_text()}")
    print(f"\n  AIC  (ARIMAX) = {fit_arimax.aic:.4f}")
    print(f"  BIC  (ARIMAX) = {fit_arimax.bic:.4f}")
    print(f"  DW   (ARIMAX) = {DW_arimax:.4f}")
    print(f"  BG AR(2) p    = {bg_arimax_p:.4f}")

    arimax_verdict = ("Autokorelasi teratasi oleh ARIMAX ✓"
                      if bg_arimax_p >= 0.05
                      else "Autokorelasi masih tersisa — pertimbangkan orde lebih tinggi")
    print(f"  Interpretasi: {arimax_verdict}")

    print(f"\n  Perbandingan AIC lintas model:")
    print(f"  {'Model':<30} {'AIC':>10} {'BIC':>10}")
    print(f"  {'-'*52}")
    print(f"  {'OLS M2 (x1+x2, k=3)':<30} {aic_M2:>10.2f} {bic_M2:>10.2f}")
    print(f"  {'OLS M7 (x1-x7, k=8)':<30} {aic_M7:>10.2f} {bic_M7:>10.2f}")
    print(f"  {'ARIMAX{} (x1+x2+AR+MA)':<30} {fit_arimax.aic:>10.2f} {fit_arimax.bic:>10.2f}".format(arimax_order))

    arimax_available = True
    aic_arimax = fit_arimax.aic
    bic_arimax = fit_arimax.bic

except ImportError:
    print("  ⚠ statsmodels tidak tersedia.")
    print("  Install: pip install statsmodels")
    print("  Panduan implementasi ARIMAX(1,0,1):")
    print("    from statsmodels.tsa.statespace.sarimax import SARIMAX")
    print("    exog = np.column_stack([x1, x2])")
    print("    model = SARIMAX(Y, exog=exog, order=(1, 0, 1), trend='c')")
    print("    fit   = model.fit(disp=False)")
    print("    print(fit.summary())")

# ============================================================
# 10. SHAPIRO-WILK & DISTRIBUSI RESIDUAL  ←  § 3.8
# ============================================================
sw_stat, sw_pval = shapiro(residuals_all)
skew_val = stats.skew(residuals_all)
kurt_val = stats.kurtosis(residuals_all, fisher=False)

sw_interp = ("Residual TIDAK berdistribusi normal (H₀ DITOLAK, p < 0.05)"
             if sw_pval < 0.05 else
             "Residual berdistribusi normal (gagal tolak H₀, p ≥ 0.05)")

print(f"\n{'='*65}")
print("§ 3.8  UJI NORMALITAS RESIDUAL (Shapiro-Wilk)")
print(f"{'='*65}")
print(f"  Shapiro-Wilk W  = {sw_stat:.4f}")
print(f"  p-value         = {sw_pval:.4e}")
print(f"  Skewness        = {skew_val:.4f}  (ideal: ≈ 0)")
print(f"  Excess Kurtosis = {kurt_val - 3:.4f}  (Fisher convention)")
print(f"  Interpretasi    : {sw_interp}")

# ============================================================
# 10b. TRANSFORMASI BOX-COX  ←  § 3.8 lanjutan
# ============================================================
print(f"\n{'='*65}")
print("§ 3.8b  TRANSFORMASI BOX-COX (Remediasi Ketidaknormalan)")
print(f"{'='*65}")

Y_positive = Y.copy()
print(f"  Min Y = {Y_positive.min():.6f} (harus > 0 untuk Box-Cox)")

Y_bc, lam_opt = boxcox(Y_positive)
print(f"  λ optimal (Box-Cox) = {lam_opt:.4f}")

beta_bc   = np.linalg.solve(X_train.T @ X_train,
                             X_train.T @ Y_bc[:n_train])
Y_pred_bc = X @ beta_bc
resid_bc  = Y_bc - Y_pred_bc

sw_bc, p_bc = shapiro(resid_bc)
skew_bc     = stats.skew(resid_bc)
kurt_bc     = stats.kurtosis(resid_bc, fisher=True)

print(f"\n  Sebelum Box-Cox : SW W={sw_stat:.4f},  p={sw_pval:.2e},  Skew={skew_val:.4f}")
print(f"  Sesudah Box-Cox : SW W={sw_bc:.4f},  p={p_bc:.2e},  Skew={skew_bc:.4f}")
bc_interp = ("H₀ masih DITOLAK ✗ — ketidaknormalan struktural terkonfirmasi"
             if p_bc < 0.05 else
             "H₀ gagal ditolak ✓ — normalitas tercapai setelah Box-Cox")
print(f"  Interpretasi: {bc_interp}")

# ============================================================
# 11. ACF RESIDUAL (manual, tanpa library tambahan)
# ============================================================
def acf_manual(x, nlags=20):
    n_pts = len(x)
    x_dm  = x - x.mean()
    c0    = np.sum(x_dm ** 2) / n_pts
    return np.array([1.0] + [
        np.sum(x_dm[:n_pts - lag] * x_dm[lag:]) / (n_pts * c0)
        for lag in range(1, nlags + 1)
    ])


lags_range = 20
acf_vals   = acf_manual(residuals_all, nlags=lags_range)
conf_bound = 1.96 / np.sqrt(n)
lags_x     = np.arange(0, lags_range + 1)

# ============================================================
# 12. PREDICTION INTERVAL 95%
# ============================================================
t_crit_pi = stats.t.ppf(0.975, dof)
leverage  = np.diag(X @ np.linalg.inv(XtX_train) @ X.T)
pred_se   = np.sqrt(mse_train * (1 + leverage))
pi_lower  = Y_pred_all - t_crit_pi * pred_se
pi_upper  = Y_pred_all + t_crit_pi * pred_se

# ============================================================
# 13. PLOT — GAMBAR 1: Panel Utama 4-in-1
# ============================================================
min_val = min(Y.min(), Y_pred_all.min()) - 0.02
max_val = max(Y.max(), Y_pred_all.max()) + 0.02

idx_train_arr = np.arange(n_train)
idx_test_arr  = np.arange(n_train, n)

fig = plt.figure(figsize=(17, 12))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# (a) Time-series
ax1.plot(idx_train_arr, Y_train,      color='purple',  lw=1.3, ls='-',  label='Y Aktual (Train)')
ax1.plot(idx_train_arr, Y_pred_train, color='magenta', lw=1.0, ls='--', label='Y Prediksi (Train)')
ax1.plot(idx_test_arr,  Y_test,       color='black',   lw=1.3, ls='-',  label='Y Aktual (Test)')
ax1.plot(idx_test_arr,  Y_pred_test,  color='gray',    lw=1.0, ls='--', label='Y Prediksi (Test)')
ax1.axvline(n_train - 0.5, color='orange', lw=1.5, ls='--', label='Batas Train/Test')
ax1.set_xlabel('Indeks Sampel Bulanan', fontsize=9)
ax1.set_ylabel('Daya (kWh/m²/hari)', fontsize=9)
ax1.set_title('(a) Daya Aktual vs Prediksi OLS\n'
              f'n=132 | Train=106 | Test=26', fontsize=10, fontweight='bold')
ax1.legend(fontsize=7, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, n + 1)
box_txt = (f"Koef. OLS (training):\n"
           f"  α  = {beta_hat[0]:.6f}\n"
           f"  x1 = {beta_hat[1]:.6f}\n"
           f"  x2 = {beta_hat[2]:.6f}\n"
           f"  x3 = {beta_hat[3]:.2e}\n"
           f"  x4 = {beta_hat[4]:.2e}\n"
           f"  x5 = {beta_hat[5]:.2e}\n"
           f"  x6 = {beta_hat[6]:.2e}\n"
           f"  x7 = {beta_hat[7]:.2e}\n"
           f"\nTest: MAE={metrics_test[0]:.6f}\n"
           f"      R²={metrics_test[2]:.8f}\n"
           f"\nDW={DW:.4f} | SW p={sw_pval:.2e}")
ax1.text(0.01, 0.01, box_txt, transform=ax1.transAxes, fontsize=5.0,
         va='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.88))

# (b) Scatter aktual vs prediksi
ax2.scatter(Y_train, Y_pred_train, c='blue',  s=22, alpha=0.8,
            label=f'Train (n={n_train})', zorder=4)
ax2.scatter(Y_test,  Y_pred_test,  c='green', s=25, marker='D', alpha=0.9,
            label=f'Test (n={n_test})',  zorder=4)
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='y = x')
ax2.set_xlabel('Y Aktual (kWh/m²/hari)', fontsize=9)
ax2.set_ylabel('Y Prediksi (kWh/m²/hari)', fontsize=9)
ax2.set_title(f'(b) Scatter: Aktual vs Prediksi\nR² (Test) = {metrics_test[2]:.8f}',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(min_val, max_val)
ax2.set_ylim(min_val, max_val)

# (c) Plot residual
ax3.plot(idx_train_arr, residuals_train, color='blue',  lw=1.0, alpha=0.85,
         label='Residual Train')
ax3.plot(idx_test_arr,  residuals_test,  color='green', lw=1.2, marker='o', ms=3,
         alpha=0.9, label='Residual Test')
ax3.axhline(0, color='red',    lw=1.0, ls='--')
ax3.axvline(n_train - 0.5, color='orange', lw=1.5, ls='--', label='Batas Train/Test')
ax3.set_xlabel('Indeks Sampel Bulanan', fontsize=9)
ax3.set_ylabel('Residual (Y – Ŷ)', fontsize=9)
ax3.set_title('(c) Plot Residual\n'
              f'DW={DW:.4f} | BG AR(1) p={bg_results[1]["p_value"]:.4f}',
              fontsize=10, fontweight='bold')
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1, n + 1)

# (d) Distribusi residual
mu_r, sd_r = residuals_all.mean(), residuals_all.std()
x_norm = np.linspace(residuals_all.min(), residuals_all.max(), 300)
ax4.hist(residuals_all, bins=50, density=True, color='steelblue', alpha=0.70,
         edgecolor='white', label='Histogram')
ax4.plot(x_norm, norm.pdf(x_norm, mu_r, sd_r), 'r-', lw=2.2,
         label='Kurva Normal')
ax4.set_xlabel('Residual', fontsize=9)
ax4.set_ylabel('Densitas', fontsize=9)
ax4.set_title(f'(d) Distribusi Residual\n'
              f'SW: W={sw_stat:.4f}, p={sw_pval:.2e} | Skew={skew_val:.4f}',
              fontsize=10, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

fig.suptitle('Analisis OLS 7 Parameter – Daya Energi Surya\n'
             'Bontang, Kalimantan Timur | NASA POWER 2015–2025 (n=132)',
             fontsize=12, fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig('outputs/Gambar1_Panel_Utama_4in1.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("\n✅ Gambar 1 tersimpan: outputs/Gambar1_Panel_Utama_4in1.png")

# ============================================================
# 14. PLOT — GAMBAR 2: AIC/BIC Comparison  ←  § 3.5
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('white')

model_labels = ['M2\n(x1+x2,\nk=3)', 'M7\n(x1–x7,\nk=8)']
aic_vals_bar = [aic_M2, aic_M7]
bic_vals_bar = [bic_M2, bic_M7]
colors_bar   = ['#4a90d9', '#e05c5c']

ax_a, ax_b = axes[0], axes[1]
bars_a = ax_a.bar(model_labels, aic_vals_bar, color=colors_bar, alpha=0.85,
                  edgecolor='white', width=0.5)
ax_a.set_title('Perbandingan AIC\n(semakin kecil = lebih baik)',
               fontsize=11, fontweight='bold')
ax_a.set_ylabel('AIC', fontsize=10)
for bar, val in zip(bars_a, aic_vals_bar):
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
              f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_a.set_ylim(min(aic_vals_bar) - 50, max(aic_vals_bar) + 60)
ax_a.grid(True, axis='y', alpha=0.3)
ax_a.text(0.5, 0.02, f'ΔAIC = {delta_aic:.2f}', transform=ax_a.transAxes,
          ha='center', fontsize=9, color='darkgreen',
          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

bars_b = ax_b.bar(model_labels, bic_vals_bar, color=colors_bar, alpha=0.85,
                  edgecolor='white', width=0.5)
ax_b.set_title('Perbandingan BIC\n(semakin kecil = lebih baik)',
               fontsize=11, fontweight='bold')
ax_b.set_ylabel('BIC', fontsize=10)
for bar, val in zip(bars_b, bic_vals_bar):
    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
              f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_b.set_ylim(min(bic_vals_bar) - 50, max(bic_vals_bar) + 60)
ax_b.grid(True, axis='y', alpha=0.3)
ax_b.text(0.5, 0.02, f'ΔBIC = {delta_bic:.2f}', transform=ax_b.transAxes,
          ha='center', fontsize=9, color='darkorange',
          bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.6))

fig.suptitle('§ 3.5 Analisis AIC/BIC — Model M7 vs M2\n'
             f'(n=132, penalti BIC per parameter = ln(132) = {np.log(n):.2f})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Gambar_AIC_BIC_Comparison.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Gambar AIC/BIC tersimpan: outputs/Gambar_AIC_BIC_Comparison.png")

# ============================================================
# 15. PLOT — GAMBAR 3: Autokorelasi Diagnostik  ←  § 3.7
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

# (a) ACF Residual M7
ax = axes[0, 0]
ax.bar(lags_x, acf_vals, color='steelblue', alpha=0.8, width=0.5)
ax.axhline( conf_bound, color='red', ls='--', lw=1.5,
            label=f'Batas 95% (±{conf_bound:.3f})')
ax.axhline(-conf_bound, color='red', ls='--', lw=1.5)
ax.fill_between(lags_x, -conf_bound, conf_bound, alpha=0.12, color='red')
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Lag (bulan)', fontsize=10)
ax.set_ylabel('Autokorelasi', fontsize=10)
ax.set_title(f'(a) ACF Residual M7\nDW={DW:.4f} | zona {"inconclusive" if dL < DW < dU else "definitif"}',
             fontsize=11, fontweight='bold')
ax.set_xticks(lags_x)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Breusch-Godfrey LM Statistics
ax = axes[0, 1]
bg_lm_vals = [bg_results[p]['LM'] for p in [1, 2]]
bg_p_vals  = [bg_results[p]['p_value'] for p in [1, 2]]
bar_colors = ['#2ecc71' if p >= 0.05 else '#e74c3c' for p in bg_p_vals]
bars = ax.bar(['AR(1)', 'AR(2)'], bg_lm_vals, color=bar_colors, alpha=0.85,
              edgecolor='white', width=0.4)
chi2_05_1 = chi2.ppf(0.95, df=1)
chi2_05_2 = chi2.ppf(0.95, df=2)
ax.axhline(chi2_05_1, color='red', ls='--', lw=1.5,
           label=f'χ²(1, 0.05)={chi2_05_1:.2f}')
ax.axhline(chi2_05_2, color='orange', ls=':', lw=1.5,
           label=f'χ²(2, 0.05)={chi2_05_2:.2f}')
for bar, val, p in zip(bars, bg_lm_vals, bg_p_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f'LM={val:.3f}\np={p:.4f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('LM Statistic', fontsize=10)
ax.set_title('(b) Breusch-Godfrey LM Test\n'
             'Hijau = gagal tolak H₀ | Merah = tolak H₀',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.3)

# (c) Perbandingan residual M7 vs Model Lag
ax = axes[1, 0]
ax.plot(np.arange(n),    residuals_all,  color='steelblue', lw=1.0,
        alpha=0.8, label=f'Residual M7 (DW={DW:.4f})')
ax.plot(np.arange(1, n), residuals_lag,  color='tomato',    lw=1.0,
        alpha=0.8, label=f'Residual Model Lag (DW={DW_lag:.4f})', ls='--')
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.axvline(n_train - 0.5, color='orange', lw=1.5, ls='--', alpha=0.7,
           label='Batas Train/Test')
ax.set_xlabel('Indeks Waktu (bulan)', fontsize=10)
ax.set_ylabel('Residual', fontsize=10)
ax.set_title('(c) Perbandingan Residual: M7 vs Model Lag\n'
             f'γ(Y_{{t-1}}) = {gamma_lag:.6f}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) ACF Residual Model Lag
acf_lag = acf_manual(residuals_lag, nlags=lags_range)
conf_bound_lag = 1.96 / np.sqrt(n - 1)
ax = axes[1, 1]
ax.bar(lags_x, acf_lag, color='tomato', alpha=0.8, width=0.5)
ax.axhline( conf_bound_lag, color='red', ls='--', lw=1.5,
            label=f'Batas 95% (±{conf_bound_lag:.3f})')
ax.axhline(-conf_bound_lag, color='red', ls='--', lw=1.5)
ax.fill_between(lags_x, -conf_bound_lag, conf_bound_lag, alpha=0.12, color='red')
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Lag (bulan)', fontsize=10)
ax.set_ylabel('Autokorelasi', fontsize=10)
ax.set_title(f'(d) ACF Residual Model Lag\n'
             f'DW={DW_lag:.4f} | BG AR(1) p={bg_lag[1]["p_value"]:.4f}',
             fontsize=11, fontweight='bold')
ax.set_xticks(lags_x)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle('§ 3.7 Diagnostik Autokorelasi — DW + Breusch-Godfrey + Model Lag\n'
             f'Bontang, Kaltim | NASA POWER 2015–2025 (n=132)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Gambar_Autokorelasi_Diagnostik.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Gambar Autokorelasi tersimpan: outputs/Gambar_Autokorelasi_Diagnostik.png")

# ============================================================
# 16. PLOT — GAMBAR 4: Heatmap Korelasi
# ============================================================
data_corr = pd.DataFrame({
    'x1(ALLSKY_DWN)': x1, 'x2(T2M)': x2, 'x3(WS10M)': x3,
    'x4(PS)': x4,          'x5(CLOUD_AMT)': x5, 'x6(PRECTOT)': x6,
    'x7(ALLSKY_DNI)': x7,  'Y(Daya)': Y
})
corr_matrix = data_corr.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_matrix.columns, fontsize=9)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val   = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)
ax.set_title('Heatmap Korelasi – 7 Parameter + Y\nBontang, Kaltim 2015–2025',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Gambar_Heatmap_Korelasi.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gambar Heatmap tersimpan: outputs/Gambar_Heatmap_Korelasi.png")

# ============================================================
# 17. PLOT — GAMBAR 5: Residual Diagnostic 3-in-1
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('white')

# Q-Q Plot
(osm, osr), (slope, intercept_qq, _) = stats.probplot(residuals_all, dist='norm')
axes[0].scatter(osm, osr, s=20, color='steelblue', alpha=0.7)
x_line = np.array([osm.min(), osm.max()])
axes[0].plot(x_line, slope * x_line + intercept_qq, 'r-', lw=2)
axes[0].set_xlabel('Kuantil Teoritis', fontsize=10)
axes[0].set_ylabel('Kuantil Sampel', fontsize=10)
axes[0].set_title('Q-Q Plot Residual', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Residual vs Fitted
colors_rf = ['royalblue'] * n_train + ['tomato'] * n_test
axes[1].scatter(Y_pred_all, residuals_all, s=20, alpha=0.7, c=colors_rf)
axes[1].axhline(0, color='red', ls='--', lw=1.5)
axes[1].set_xlabel('Nilai Fitted (Prediksi)', fontsize=10)
axes[1].set_ylabel('Residual', fontsize=10)
axes[1].set_title('Residual vs Fitted\n(Cek Heteroskedastisitas)', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# ACF
axes[2].bar(lags_x, acf_vals, color='steelblue', alpha=0.8, width=0.5)
axes[2].axhline( conf_bound, color='red', ls='--', lw=1.5)
axes[2].axhline(-conf_bound, color='red', ls='--', lw=1.5)
axes[2].fill_between(lags_x, -conf_bound, conf_bound, alpha=0.12, color='red')
axes[2].axhline(0, color='black', lw=0.8)
axes[2].set_xlabel('Lag', fontsize=10)
axes[2].set_ylabel('ACF', fontsize=10)
axes[2].set_title(f'ACF Residual\nDW={DW:.4f} | BG p={bg_results[1]["p_value"]:.4f}',
                  fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Residual Diagnostic – OLS M7, 7 Parameter', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Gambar_Residual_Diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gambar Residual Diagnostik tersimpan: outputs/Gambar_Residual_Diagnostic.png")

# ============================================================
# 18. PLOT — GAMBAR 6: Box-Cox & Normalitas  ←  § 3.8
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('white')

# (a) Log-likelihood vs λ
lam_range = np.linspace(-2, 6, 400)
ll_vals   = []
for lam in lam_range:
    if lam == 0:
        y_t = np.log(Y_positive)
    else:
        y_t = (Y_positive ** lam - 1) / lam
    s2  = np.var(y_t)
    ll  = -0.5 * n * np.log(s2) + (lam - 1) * np.sum(np.log(Y_positive))
    ll_vals.append(ll)
axes[0, 0].plot(lam_range, ll_vals, 'b-', lw=2)
axes[0, 0].axvline(lam_opt, color='red', ls='--', lw=2,
                   label=f'λ optimal = {lam_opt:.2f}')
axes[0, 0].set_xlabel('λ', fontsize=10)
axes[0, 0].set_ylabel('Log-Likelihood', fontsize=10)
axes[0, 0].set_title('(a) Log-Likelihood vs λ', fontsize=11, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# (b) Distribusi residual SEBELUM
x_n = np.linspace(residuals_all.min(), residuals_all.max(), 300)
axes[0, 1].hist(residuals_all, bins=40, density=True,
                color='steelblue', alpha=0.7, edgecolor='white')
axes[0, 1].plot(x_n, norm.pdf(x_n, residuals_all.mean(), residuals_all.std()),
                'r-', lw=2, label='Kurva Normal')
axes[0, 1].set_title(f'(b) Distribusi Residual SEBELUM\n'
                     f'SW W={sw_stat:.4f}, p={sw_pval:.2e}\nSkew={skew_val:.4f}',
                     fontsize=10, fontweight='bold')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# (c) Distribusi residual SESUDAH
x_bc_n = np.linspace(resid_bc.min(), resid_bc.max(), 300)
axes[0, 2].hist(resid_bc, bins=40, density=True,
                color='tomato', alpha=0.7, edgecolor='white')
axes[0, 2].plot(x_bc_n, norm.pdf(x_bc_n, resid_bc.mean(), resid_bc.std()),
                'b-', lw=2, label='Kurva Normal')
axes[0, 2].set_title(f'(c) Distribusi Residual SESUDAH Box-Cox (λ={lam_opt:.2f})\n'
                     f'SW W={sw_bc:.4f}, p={p_bc:.2e}\nSkew={skew_bc:.4f}',
                     fontsize=10, fontweight='bold')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

# (d) Q-Q SEBELUM
(osm_b, osr_b), (sl_b, ic_b, _) = stats.probplot(residuals_all, dist='norm')
axes[1, 0].scatter(osm_b, osr_b, s=18, color='steelblue', alpha=0.7)
xl = np.array([osm_b.min(), osm_b.max()])
axes[1, 0].plot(xl, sl_b * xl + ic_b, 'r-', lw=2)
axes[1, 0].set_xlabel('Kuantil Teoritis', fontsize=9)
axes[1, 0].set_ylabel('Kuantil Sampel', fontsize=9)
axes[1, 0].set_title('(d) Q-Q Plot SEBELUM Box-Cox', fontsize=10, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# (e) Q-Q SESUDAH
(osm_a, osr_a), (sl_a, ic_a, _) = stats.probplot(resid_bc, dist='norm')
axes[1, 1].scatter(osm_a, osr_a, s=18, color='tomato', alpha=0.7)
xl2 = np.array([osm_a.min(), osm_a.max()])
axes[1, 1].plot(xl2, sl_a * xl2 + ic_a, 'b-', lw=2)
axes[1, 1].set_xlabel('Kuantil Teoritis', fontsize=9)
axes[1, 1].set_ylabel('Kuantil Sampel', fontsize=9)
axes[1, 1].set_title(f'(e) Q-Q Plot SESUDAH Box-Cox (λ={lam_opt:.2f})',
                     fontsize=10, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# (f) Perbandingan Shapiro-Wilk W
axes[1, 2].bar(['Sebelum\nBox-Cox', 'Sesudah\nBox-Cox'],
               [sw_stat, sw_bc],
               color=['steelblue', 'tomato'], alpha=0.85,
               edgecolor='white', width=0.4)
axes[1, 2].axhline(0.95, color='green', ls='--', lw=2,
                   label='Threshold W=0.95')
axes[1, 2].set_ylim(0.85, 1.0)
axes[1, 2].set_ylabel('Shapiro-Wilk W', fontsize=10)
axes[1, 2].set_title('(f) Perbandingan Shapiro-Wilk W\n(semakin mendekati 1 = lebih normal)',
                     fontsize=10, fontweight='bold')
for i_b, val in enumerate([sw_stat, sw_bc]):
    axes[1, 2].text(i_b, val + 0.001, f'W={val:.4f}',
                    ha='center', fontsize=10, fontweight='bold')
axes[1, 2].legend(fontsize=9)
axes[1, 2].grid(True, axis='y', alpha=0.3)

fig.suptitle(f'§ 3.8 Analisis Box-Cox & Uji Normalitas Residual\n'
             f'Bontang, Kaltim | NASA POWER 2015–2025 | λ optimal = {lam_opt:.2f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Gambar_BoxCox_Normalitas.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Gambar Box-Cox tersimpan: outputs/Gambar_BoxCox_Normalitas.png")

# ============================================================
# 19. SIMPAN HASIL LENGKAP KE TXT
# ============================================================
with open('outputs/hasil_analisis_lengkap.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 75 + "\n")
    f.write("ANALISIS OLS – ESTIMASI DAYA PANEL SURYA\n")
    f.write("7 Parameter, Bontang, Kaltim | NASA POWER 2015–2025 (n=132)\n")
    f.write("=" * 75 + "\n\n")
    f.write(f"Data: n={n} | Train={n_train} | Test={n_test}\n\n")

    f.write("─" * 75 + "\n")
    f.write("§ 3.5  KOEFISIEN OLS + UJI SIGNIFIKANSI (t-test, df=98)\n")
    f.write("─" * 75 + "\n")
    f.write(f"{'Parameter':<22} {'Koef.':>12} {'SE':>12} {'t-stat':>10} "
            f"{'p-value':>10} {'Sig':>5}\n")
    f.write("-" * 75 + "\n")
    for i, name in enumerate(feature_names):
        sig = ("***" if p_values[i] < 0.001 else
               "**"  if p_values[i] < 0.01  else
               "*"   if p_values[i] < 0.05  else "n.s.")
        f.write(f"{name:<22} {beta_hat[i]:>12.6f} {se_beta[i]:>12.6f} "
                f"{t_stat[i]:>10.4f} {p_values[i]:>10.4f} {sig:>5}\n")
    f.write("*** p<0.001  ** p<0.01  * p<0.05  n.s.=tidak signifikan\n\n")

    f.write("─" * 75 + "\n")
    f.write("§ 3.5  ANALISIS AIC/BIC\n")
    f.write("─" * 75 + "\n")
    f.write(f"{'Model':<28} {'k':>4} {'SSE':>16} {'AIC':>10} {'BIC':>10}\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'M2 (x1+x2, k=3)':<28} {k_M2:>4} {sse_M2:>16.10f} {aic_M2:>10.4f} {bic_M2:>10.4f}\n")
    f.write(f"{'M7 (x1-x7, k=8) ★':<28} {k_M7:>4} {sse_M7:>16.10f} {aic_M7:>10.4f} {bic_M7:>10.4f}\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'ΔAIC (M7 − M2)':<28} {delta_aic:>49.4f}\n")
    f.write(f"{'ΔBIC (M7 − M2)':<28} {delta_bic:>60.4f}\n")
    f.write(f"ln(n) = ln({n}) = {np.log(n):.4f}\n\n")

    f.write("─" * 75 + "\n")
    f.write("§ 3.6  VIF\n")
    f.write("─" * 75 + "\n")
    for name, vif in zip(var_names, vif_values):
        flag = " (TINGGI >10)" if vif > 10 else (" (Perhatikan 5-10)" if vif > 5 else " (Aman)")
        f.write(f"  {name:<22} VIF = {vif:.4f}{flag}\n")

    f.write("\n" + "─" * 75 + "\n")
    f.write("§ 3.7  UJI AUTOKORELASI\n")
    f.write("─" * 75 + "\n")
    f.write(f"[A] Durbin-Watson  : DW = {DW:.4f}  |  {dw_verdict}\n")
    f.write(f"    dL={dL}, dU={dU} (n=132, k=7, α=0.05)\n\n")
    f.write(f"[B] Breusch-Godfrey (LM Test):\n")
    for p, res_bg in bg_results.items():
        keputusan = "Tolak H0" if res_bg['p_value'] < 0.05 else "Gagal Tolak H0"
        f.write(f"    AR({p}): LM={res_bg['LM']:.4f}, p={res_bg['p_value']:.4f} → {keputusan}\n")
    f.write(f"    Kesimpulan: {bg_verdict}\n\n")
    f.write(f"[C] Model Lag [Y_{{t-1}} sebagai kovariat]:\n")
    f.write(f"    Koef. γ(Y_{{t-1}}) = {gamma_lag:.6f}\n")
    f.write(f"    DW_lag = {DW_lag:.4f}  |  BG AR(1) p = {bg_lag[1]['p_value']:.4f}\n")
    f.write(f"    MAE={mae_lag:.8f}  MAPE={mape_lag:.6f}%  R²={r2_lag:.8f}\n")
    f.write(f"    Interpretasi: {lag_verdict}\n\n")

    f.write("─" * 75 + "\n")
    f.write("§ 3.8  NORMALITAS RESIDUAL & BOX-COX\n")
    f.write("─" * 75 + "\n")
    f.write(f"  Shapiro-Wilk : W={sw_stat:.4f}, p={sw_pval:.4e}\n")
    f.write(f"  Skewness     : {skew_val:.4f}  |  Excess Kurtosis: {kurt_val-3:.4f}\n")
    f.write(f"  Interpretasi : {sw_interp}\n\n")
    f.write(f"  Box-Cox λ optimal = {lam_opt:.4f}\n")
    f.write(f"  SW sesudah Box-Cox: W={sw_bc:.4f}, p={p_bc:.4e}\n")
    f.write(f"  Interpretasi: {bc_interp}\n\n")

    f.write("─" * 75 + "\n")
    f.write("METRIK EVALUASI\n")
    f.write("─" * 75 + "\n")
    f.write(f"{'Metrik':<10} {'Training':>12} {'Test':>12} {'Keseluruhan':>14}\n")
    f.write("-" * 52 + "\n")
    for i, mn in enumerate(['MAE', 'MAPE(%)', 'R²', 'RMSE']):
        f.write(f"{mn:<10} {metrics_train[i]:>12.8f} {metrics_test[i]:>12.8f} "
                f"{metrics_all[i]:>14.8f}\n")
    f.write(f"\nM2 (x1+x2): MAE={mae_M2:.8f}  R²={r2_M2:.8f}\n")

print("\n✅ Hasil analisis: outputs/hasil_analisis_lengkap.txt")

# ============================================================
# 20. RINGKASAN KONSOL
# ============================================================
print(f"\n{'='*70}")
print("RINGKASAN AKHIR — MODEL OLS 7 PARAMETER")
print(f"{'='*70}")
print(f"Ŷ = α + a·x1 + b·x2 + c·x3 + d·x4 + e·x5 + f·x6 + g·x7")
print(f"  α  = {beta_hat[0]:>12.6f}  (Intercept)")
for i, name in enumerate(var_names):
    print(f"  {chr(97+i+1)}  = {beta_hat[i+1]:>12.6f}  ({name})")

print(f"\n{'Metrik':<10} {'Training':>12} {'Test':>12} {'Keseluruhan':>14}")
print("-" * 52)
for i, mn in enumerate(['MAE', 'MAPE(%)', 'R²', 'RMSE']):
    print(f"{mn:<10} {metrics_train[i]:>12.8f} {metrics_test[i]:>12.8f} "
          f"{metrics_all[i]:>14.8f}")

print(f"\n§ 3.5  AIC/BIC:")
print(f"  M2 (k=3): AIC={aic_M2:.2f}, BIC={bic_M2:.2f}")
print(f"  M7 (k=8): AIC={aic_M7:.2f}, BIC={bic_M7:.2f}")
print(f"  ΔAIC={delta_aic:.2f}  ΔBIC={delta_bic:.2f}")

print(f"\n§ 3.7  Autokorelasi:")
print(f"  DW = {DW:.4f}  ({dw_verdict})")
print(f"  BG AR(1): LM={bg_results[1]['LM']:.4f}, p={bg_results[1]['p_value']:.4f}")
print(f"  BG AR(2): LM={bg_results[2]['LM']:.4f}, p={bg_results[2]['p_value']:.4f}")
print(f"  → {bg_verdict}")
print(f"  Model Lag: DW={DW_lag:.4f}, BG p={bg_lag[1]['p_value']:.4f}")
print(f"  → {lag_verdict}")

print(f"\n§ 3.8  Normalitas: SW W={sw_stat:.4f}, p={sw_pval:.2e}")
print(f"  → {sw_interp}")
print(f"  Box-Cox λ={lam_opt:.2f}: SW W={sw_bc:.4f}, p={p_bc:.2e}")
print(f"  → {bc_interp}")

print(f"\n{'='*70}")
print("OUTPUT TERSIMPAN DI FOLDER: outputs/")
print(f"{'='*70}")
print("  Gambar1_Panel_Utama_4in1.png         ← Panel utama 4-in-1")
print("  Gambar_AIC_BIC_Comparison.png        ← § 3.5 AIC/BIC")
print("  Gambar_Autokorelasi_Diagnostik.png   ← § 3.7 DW + BG + Lag")
print("  Gambar_Heatmap_Korelasi.png          ← Heatmap korelasi Pearson")
print("  Gambar_Residual_Diagnostic.png       ← Q-Q, Residual vs Fitted, ACF")
print("  Gambar_BoxCox_Normalitas.png         ← § 3.8 Box-Cox & normalitas")
print("  hasil_analisis_lengkap.txt           ← Ringkasan seluruh uji statistik")
print(f"{'='*70}")