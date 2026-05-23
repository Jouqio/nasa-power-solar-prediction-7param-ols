"""
Refactored PV Power Estimation Pipeline
=======================================
Transition from deterministic tautology to stochastic modeling

Author: [Your Name]
Date: January 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA GENERATION (STOCHASTIC SYNTHETIC)
# ============================================================================

class StochasticPVSimulator:
    """
    Generate realistic PV power data with stochastic loss components
    
    References:
    -----------
    - Soiling: Micheli et al. (2017), Kimber et al. (2006)
    - Degradation: Jordan et al. (2016)
    - Inverter: Notton et al. (2010)
    """
    
    def __init__(self, system_capacity_kw=1.0, eta_stc=0.18, beta=-0.0045):
        self.capacity = system_capacity_kw
        self.eta_stc = eta_stc
        self.beta = beta
        
        # Draw stochastic parameters (constant for entire simulation)
        self.degradation_rate = np.random.normal(0.007, 0.002)
        self.beta_effective = self.beta + np.random.normal(0, 0.0003)
        
        # Initialize soiling state
        self.soiling_loss = 0.0
        
    def simulate_power(self, ghi, t2m, cloud_amt, rain, ws10m, n_days):
        """
        Simulate PV power with full stochastic model
        
        Parameters:
        -----------
        ghi : array
            NASA POWER GHI (kWh/m²/day)
        t2m : array
            NASA POWER T2M (°C)
        cloud_amt : array
            NASA POWER CLOUD_AMT (%)
        rain : array
            NASA POWER IMERG_PRECTOT (mm/day)
        ws10m : array
            NASA POWER WS10M (m/s)
        n_days : int
            Number of days to simulate
            
        Returns:
        --------
        results : dict
            Dictionary containing power time series and loss components
        """
        
        # Initialize arrays
        P_ideal = np.zeros(n_days)
        P_thermal = np.zeros(n_days)
        P_soiled = np.zeros(n_days)
        P_spectral = np.zeros(n_days)
        P_inverter = np.zeros(n_days)
        P_final = np.zeros(n_days)
        
        soiling_losses = np.zeros(n_days)
        
        for t in range(n_days):
            age_years = t / 365.25
            
            # === Component 1: Ideal DC power ===
            # Cell temperature model (NOCT-based)
            T_cell = t2m[t] + 20 + 0.05 * ghi[t] * (1 - ws10m[t]/10)
            
            P_ideal[t] = (
                ghi[t] * self.capacity * self.eta_stc *
                (1 + self.beta_effective * (T_cell - 25))
            )
            
            # === Component 2: Thermal losses (already in P_ideal) ===
            P_thermal[t] = P_ideal[t]
            
            # === Component 3: Soiling dynamics ===
            # Accumulation rate (season-dependent)
            day_of_year = t % 365
            if 150 <= day_of_year < 240:  # Dry season (Jun-Aug)
                accum_rate = np.random.normal(0.003, 0.0007)
            elif day_of_year >= 300 or day_of_year < 60:  # Monsoon
                accum_rate = np.random.normal(0.001, 0.0003)
            else:
                accum_rate = np.random.normal(0.002, 0.0005)
            
            self.soiling_loss += accum_rate
            
            # Rain cleaning events
            if rain[t] > 5:  # Threshold for cleaning
                clean_eff = np.random.beta(8, 2)  # Mean ~80%
                self.soiling_loss *= (1 - clean_eff)
            
            # Cap maximum soiling
            self.soiling_loss = np.clip(self.soiling_loss, 0, 0.15)
            soiling_losses[t] = self.soiling_loss
            
            P_soiled[t] = P_thermal[t] * (1 - self.soiling_loss)
            
            # === Component 4: Spectral mismatch ===
            # Cloud-induced spectral shift
            spectral_factor = 1.0 - 0.05 * (cloud_amt[t]/100)
            spectral_factor += np.random.normal(0, 0.02)  # Stochastic variation
            spectral_factor = np.clip(spectral_factor, 0.85, 1.05)
            
            P_spectral[t] = P_soiled[t] * spectral_factor
            
            # === Component 5: MPPT + Inverter ===
            # MPPT efficiency (stochastic)
            eta_mppt = np.random.beta(95, 5) / 100  # ~95% mean
            
            # DC power post-MPPT
            P_dc = P_spectral[t] * eta_mppt
            
            # Inverter efficiency (Sandia model)
            p_norm = P_dc / self.capacity
            if p_norm > 0:
                eta_inv = 0.98 * (p_norm / (p_norm + 0.05))
                eta_inv *= np.random.normal(1.0, 0.01)  # ±1% variation
            else:
                eta_inv = 0
            
            # AC power with clipping
            P_ac = min(P_dc * eta_inv, 1.05 * self.capacity)
            P_inverter[t] = P_ac
            
            # === Component 6: Degradation ===
            R_deg = (1 - self.degradation_rate) ** age_years
            
            # === Component 7: Wiring losses ===
            L_wire = 0.015 + 0.0002 * (t2m[t] - 25)
            
            # === Final power ===
            P_final[t] = P_inverter[t] * R_deg * (1 - L_wire)
            
            # Add measurement noise (±1%)
            P_final[t] *= np.random.normal(1.0, 0.01)
            P_final[t] = max(0, P_final[t])  # Physical constraint
        
        # Calculate Performance Ratio
        PR = P_final / (P_ideal + 1e-10)
        
        # Package results
        results = {
            'P_ideal': P_ideal,
            'P_final': P_final,
            'PR': PR,
            'soiling_losses': soiling_losses,
            'degradation_rate': self.degradation_rate,
            'thermal_losses': P_ideal - P_thermal,  # Included in model
            'inverter_losses': P_spectral - P_inverter,
            'total_losses': P_ideal - P_final
        }
        
        return results


# ============================================================================
# SECTION 2: DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath='nasa_power_data.csv'):
    """
    Load NASA POWER data and perform preprocessing
    
    Returns:
    --------
    df : DataFrame
        Preprocessed dataframe with all features
    """
    
    # Load data
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Feature engineering
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['season'] = df['month'].apply(
        lambda m: 'dry' if 4 <= m <= 9 else 'wet'
    )
    
    # ENSO index (if available, otherwise placeholder)
    # In real implementation, merge with actual Niño3.4 index
    df['nino34'] = 0  # Placeholder
    
    # Interaction terms for CORRECTED OLS
    df['GHI_T2M'] = df['ALLSKY_SFC_SW_DWN'] * df['T2M']
    df['GHI_CLOUD'] = df['ALLSKY_SFC_SW_DWN'] * df['CLOUD_AMT']
    
    return df


# ============================================================================
# SECTION 3: GENERATE SYNTHETIC TARGET (STOCHASTIC)
# ============================================================================

def generate_stochastic_target(df, system_capacity=1.0):
    """
    Generate stochastic PV power target using physics-informed model
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with NASA POWER features
    system_capacity : float
        System capacity in kW_p
        
    Returns:
    --------
    df : DataFrame
        Dataframe with added 'P_final' target column
    """
    
    # Initialize simulator
    simulator = StochasticPVSimulator(
        system_capacity_kw=system_capacity,
        eta_stc=0.18,
        beta=-0.0045
    )
    
    # Extract features
    ghi = df['ALLSKY_SFC_SW_DWN'].values
    t2m = df['T2M'].values
    cloud = df['CLOUD_AMT'].values
    rain = df['IMERG_PRECTOT'].values
    ws10m = df['WS10M'].values
    
    n_days = len(df)
    
    # Run simulation
    results = simulator.simulate_power(
        ghi=ghi,
        t2m=t2m,
        cloud_amt=cloud,
        rain=rain,
        ws10m=ws10m,
        n_days=n_days
    )
    
    # Add to dataframe
    df['P_ideal'] = results['P_ideal']
    df['P_final'] = results['P_final']  # THIS IS THE NEW TARGET
    df['PR'] = results['PR']
    df['soiling_loss'] = results['soiling_losses']
    
    print(f"\n{'='*70}")
    print("STOCHASTIC TARGET GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"Mean Performance Ratio: {df['PR'].mean():.3f}")
    print(f"PR Std Dev: {df['PR'].std():.3f}")
    print(f"Mean Soiling Loss: {df['soiling_loss'].mean():.4f}")
    print(f"Degradation Rate: {results['degradation_rate']:.4f} per year")
    print(f"{'='*70}\n")
    
    return df, results


# ============================================================================
# SECTION 4: CORRECTED OLS MODEL (WITH INTERACTIONS)
# ============================================================================

class CorrectedOLSModel:
    """
    OLS model with proper specification including interaction terms
    
    This model corrects the original flaw:
    - Includes GHI × T2M interaction
    - Polynomial terms if needed
    - Proper residual diagnostics
    """
    
    def __init__(self, include_interactions=True):
        self.include_interactions = include_interactions
        self.model = None
        self.results = None
        self.feature_names = None
        
    def fit(self, X, y):
        """
        Fit OLS model with statsmodels for full diagnostics
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable (P_final, NOT P_ideal)
        """
        
        # Add constant
        X_with_const = sm.add_constant(X)
        
        # Fit OLS
        self.model = sm.OLS(y, X_with_const)
        self.results = self.model.fit()
        self.feature_names = X_with_const.columns.tolist()
        
        return self
    
    def predict(self, X):
        """Predict using fitted model"""
        X_with_const = sm.add_constant(X)
        # Ensure column order matches
        X_with_const = X_with_const[self.feature_names]
        return self.results.predict(X_with_const)
    
    def summary(self):
        """Print model summary"""
        return self.results.summary()
    
    def diagnostics(self, y_true, y_pred):
        """
        Comprehensive diagnostic tests
        
        Returns:
        --------
        diagnostics : dict
            Dictionary of diagnostic test results
        """
        
        residuals = y_true - y_pred
        
        # 1. Durbin-Watson
        dw = durbin_watson(residuals)
        
        # 2. Breusch-Godfrey for autocorrelation
        bg_test = acorr_breusch_godfrey(self.results, nlags=2)
        
        # 3. Shapiro-Wilk for normality
        sw_stat, sw_pval = stats.shapiro(residuals)
        
        # 4. White test for heteroskedasticity
        white_test = het_white(residuals, self.results.model.exog)
        
        # 5. Residual skewness and kurtosis
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        # 6. VIF (Variance Inflation Factor)
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X_for_vif = self.results.model.exog[:, 1:]  # Exclude constant
        vif_data = pd.DataFrame({
            'Feature': self.feature_names[1:],
            'VIF': [variance_inflation_factor(X_for_vif, i) 
                    for i in range(X_for_vif.shape[1])]
        })
        
        diagnostics = {
            'durbin_watson': dw,
            'breusch_godfrey': {
                'lm_stat': bg_test[0],
                'lm_pval': bg_test[1],
                'f_stat': bg_test[2],
                'f_pval': bg_test[3]
            },
            'shapiro_wilk': {
                'statistic': sw_stat,
                'pvalue': sw_pval
            },
            'white_test': {
                'lm_stat': white_test[0],
                'lm_pval': white_test[1],
                'f_stat': white_test[2],
                'f_pval': white_test[3]
            },
            'residual_stats': {
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'vif': vif_data
        }
        
        return diagnostics
    
    def print_diagnostics(self, diagnostics):
        """Pretty print diagnostics"""
        
        print(f"\n{'='*70}")
        print("OLS DIAGNOSTIC TESTS (CORRECTED MODEL)")
        print(f"{'='*70}\n")
        
        # Durbin-Watson
        dw = diagnostics['durbin_watson']
        print(f"1. Durbin-Watson Test:")
        print(f"   Statistic: {dw:.4f}")
        if dw < 1.5:
            print(f"   → Positive autocorrelation detected")
        elif dw > 2.5:
            print(f"   → Negative autocorrelation detected")
        else:
            print(f"   → No strong autocorrelation")
        
        # Breusch-Godfrey
        bg = diagnostics['breusch_godfrey']
        print(f"\n2. Breusch-Godfrey Test (AR(2)):")
        print(f"   LM Statistic: {bg['lm_stat']:.4f}")
        print(f"   p-value: {bg['lm_pval']:.6f}")
        if bg['lm_pval'] < 0.05:
            print(f"   → REJECT H0: Autocorrelation detected ***")
        else:
            print(f"   → ACCEPT H0: No autocorrelation")
        
        # Shapiro-Wilk
        sw = diagnostics['shapiro_wilk']
        print(f"\n3. Shapiro-Wilk Test (Normality):")
        print(f"   W Statistic: {sw['statistic']:.4f}")
        print(f"   p-value: {sw['pvalue']:.6e}")
        if sw['pvalue'] < 0.05:
            print(f"   → REJECT H0: Residuals NOT normally distributed ***")
        else:
            print(f"   → ACCEPT H0: Residuals normally distributed")
        
        # White Test
        white = diagnostics['white_test']
        print(f"\n4. White Test (Heteroskedasticity):")
        print(f"   LM Statistic: {white['lm_stat']:.4f}")
        print(f"   p-value: {white['lm_pval']:.6f}")
        if white['lm_pval'] < 0.05:
            print(f"   → REJECT H0: Heteroskedasticity detected ***")
        else:
            print(f"   → ACCEPT H0: Homoskedastic residuals")
        
        # Residual stats
        rs = diagnostics['residual_stats']
        print(f"\n5. Residual Moments:")
        print(f"   Skewness: {rs['skewness']:.4f}")
        print(f"   Excess Kurtosis: {rs['kurtosis']:.4f}")
        
        # VIF
        print(f"\n6. Variance Inflation Factors (VIF):")
        print(diagnostics['vif'].to_string(index=False))
        
        print(f"\n{'='*70}\n")


# ============================================================================
# SECTION 5: MACHINE LEARNING MODELS (FAIR BENCHMARKING)
# ============================================================================

class MLBenchmark:
    """
    Fair machine learning benchmarking on stochastic data
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Initialize ML models with reasonable hyperparameters"""
        
        self.models = {
            'LinearRegression': LinearRegression(),
            
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            
            'SVR': SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.01
            ),
            
            'XGBoost': XGBRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        }
        
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate
        
        Returns:
        --------
        results : DataFrame
            Comparison of model performances
        """
        
        results_list = []
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Standardize features for SVR
            if name == 'SVR':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'Model': name,
                'Train_MAE': mean_absolute_error(y_train, y_pred_train),
                'Test_MAE': mean_absolute_error(y_test, y_pred_test),
                'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'Train_R2': r2_score(y_train, y_pred_train),
                'Test_R2': r2_score(y_test, y_pred_test),
                'Train_MAPE': np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100,
                'Test_MAPE': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            }
            
            results_list.append(metrics)
            
            # Store predictions
            self.results[name] = {
                'model': model,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
        
        results_df = pd.DataFrame(results_list)
        
        return results_df


# ============================================================================
# SECTION 6: TIME SERIES CROSS-VALIDATION
# ============================================================================

def time_series_cv(X, y, n_splits=5):
    """
    Time series cross-validation with proper temporal ordering
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    n_splits : int
        Number of splits
        
    Returns:
    --------
    cv_results : DataFrame
        Cross-validation results for each fold
    """
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    models = {
        'OLS': LinearRegression(),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100, max_depth=8, random_state=42
        )
    }
    
    cv_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
        
        for model_name, model in models.items():
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            
            cv_results.append({
                'Fold': fold + 1,
                'Model': model_name,
                'MAE': mean_absolute_error(y_test_fold, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test_fold, y_pred)),
                'R2': r2_score(y_test_fold, y_pred)
            })
    
    return pd.DataFrame(cv_results)


# ============================================================================
# SECTION 7: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    
    print("\n" + "="*70)
    print("STOCHASTIC PV POWER ESTIMATION PIPELINE")
    print("="*70 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print("Step 1: Loading NASA POWER data...")
    
    # REPLACE THIS with actual data loading
    # For demonstration, create synthetic NASA POWER-like data
    np.random.seed(42)
    n_months = 132  # 11 years
    
    df = pd.DataFrame({
        'date': pd.date_range('2015-01-01', periods=n_months, freq='MS'),
        'ALLSKY_SFC_SW_DWN': np.random.gamma(5, 1, n_months),  # GHI
        'T2M': 25 + 3 * np.sin(np.arange(n_months) * 2 * np.pi / 12) + np.random.normal(0, 1, n_months),
        'CLOUD_AMT': 50 + 20 * np.sin(np.arange(n_months) * 2 * np.pi / 12) + np.random.normal(0, 10, n_months),
        'IMERG_PRECTOT': np.abs(np.random.gamma(2, 3, n_months)),
        'WS10M': 3 + np.random.gamma(2, 1, n_months),
        'PS': 101.3 + np.random.normal(0, 0.5, n_months),
        'ALLSKY_SFC_SW_DNI': np.random.gamma(4, 1, n_months)
    })
    
    df = load_and_preprocess_data_inplace(df)
    
    print(f"✓ Loaded {len(df)} monthly observations")
    
    # -------------------------------------------------------------------------
    # Step 2: Generate stochastic target
    # -------------------------------------------------------------------------
    print("\nStep 2: Generating stochastic PV power target...")
    
    df, sim_results = generate_stochastic_target(df, system_capacity=1.0)
    
    # -------------------------------------------------------------------------
    # Step 3: Train-test split (temporal)
    # -------------------------------------------------------------------------
    print("\nStep 3: Splitting data (80-20 temporal split)...")
    
    split_idx = int(0.8 * len(df))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Features to use
    feature_cols = [
        'ALLSKY_SFC_SW_DWN',
        'T2M',
        'CLOUD_AMT',
        'IMERG_PRECTOT',
        'WS10M',
        'GHI_T2M',  # INTERACTION TERM (CRITICAL)
        'GHI_CLOUD'
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df['P_final']
    
    X_test = test_df[feature_cols]
    y_test = test_df['P_final']
    
    print(f"✓ Train: {len(train_df)} obs, Test: {len(test_df)} obs")
    
    # -------------------------------------------------------------------------
    # Step 4: Fit corrected OLS model
    # -------------------------------------------------------------------------
    print("\nStep 4: Training corrected OLS model (with interactions)...")
    
    ols_model = CorrectedOLSModel()
    ols_model.fit(X_train, y_train)
    
    print("\n" + "="*70)
    print("OLS MODEL SUMMARY")
    print("="*70)
    print(ols_model.summary())
    
    # Predictions
    y_pred_train_ols = ols_model.predict(X_train)
    y_pred_test_ols = ols_model.predict(X_test)
    
    # Diagnostics
    diagnostics = ols_model.diagnostics(y_train, y_pred_train_ols)
    ols_model.print_diagnostics(diagnostics)
    
    # -------------------------------------------------------------------------
    # Step 5: ML Benchmarking
    # -------------------------------------------------------------------------
    print("\nStep 5: Training Machine Learning models...")
    
    ml_bench = MLBenchmark()
    ml_bench.initialize_models()
    
    results_df = ml_bench.train_and_evaluate(
        X_train, y_train, X_test, y_test
    )
    
    print("\n" + "="*70)
    print("MACHINE LEARNING BENCHMARK RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 6: Time Series Cross-Validation
    # -------------------------------------------------------------------------
    print("\nStep 6: Time series cross-validation...")
    
    cv_results = time_series_cv(
        X_train, y_train, n_splits=5
    )
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS (5-Fold TimeSeriesSplit)")
    print("="*70)
    print(cv_results.groupby('Model')[['MAE', 'RMSE', 'R2']].agg(['mean', 'std']))
    print("="*70 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 7: Visualization
    # -------------------------------------------------------------------------
    print("\nStep 7: Generating visualizations...")
    
    create_diagnostic_plots(
        y_test, 
        y_pred_test_ols,
        ml_bench.results,
        diagnostics,
        save_path='/home/claude/'
    )
    
    print("\n✓ Pipeline completed successfully!")
    print("="*70 + "\n")
    
    return df, ols_model, ml_bench, results_df


def load_and_preprocess_data_inplace(df):
    """Preprocess synthetic dataframe"""
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['GHI_T2M'] = df['ALLSKY_SFC_SW_DWN'] * df['T2M']
    df['GHI_CLOUD'] = df['ALLSKY_SFC_SW_DWN'] * df['CLOUD_AMT']
    return df


def create_diagnostic_plots(y_test, y_pred_ols, ml_results, diagnostics, save_path=''):
    """
    Create comprehensive diagnostic visualizations
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Actual vs Predicted (OLS)
    axes[0, 0].scatter(y_test, y_pred_ols, alpha=0.6, s=50)
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Power (kWh/day)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Power (kWh/day)', fontsize=12)
    axes[0, 0].set_title('OLS: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Residuals vs Fitted
    residuals = y_test - y_pred_ols
    axes[0, 1].scatter(y_pred_ols, residuals, alpha=0.6, s=50)
    axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Fitted Values', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    axes[0, 2].grid(alpha=0.3)
    
    # Plot 4: Histogram of Residuals
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 5: ACF of Residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=24, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('ACF of Residuals', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # Plot 6: Model Comparison (R² scores)
    model_names = ['OLS']
    r2_scores = [r2_score(y_test, y_pred_ols)]
    
    for name, result in ml_results.items():
        model_names.append(name)
        r2_scores.append(r2_score(y_test, result['y_pred_test']))
    
    axes[1, 2].barh(model_names, r2_scores, color='steelblue')
    axes[1, 2].set_xlabel('R² Score', fontsize=12)
    axes[1, 2].set_title('Model Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].grid(alpha=0.3, axis='x')
    axes[1, 2].set_xlim([0.7, 1.0])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}diagnostic_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Diagnostic plots saved to {save_path}diagnostic_plots.png")


# ============================================================================
# RUN PIPELINE
# ============================================================================

if __name__ == "__main__":
    df, ols_model, ml_bench, results_df = main()
