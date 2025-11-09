#!/usr/bin/env python
# coding: utf-8

"""
Media Mix Modeling (MMM) - Cleaned Version
Focuses on 3 key validations:
1. R² Score
2. Actual vs Predicted Trendline
3. ROI for Q2 Period (using actual spend)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pymc as pm
import arviz as az

if __name__ == '__main__':
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================

    print("Loading data...")
    input_df = pd.read_excel("input_data1.xlsx")

    # ============================================================================
    # 2. DEFINE VARIABLES
    # ============================================================================

    # Media variables
    media_vars = ['CDC+DIGITAL', 'CDC+TV', 'EQUITY+DIGITAL',
       'EQUITY+IN STORE', 'EQUITY+OOH',
       'EQUITY+OTHERADVERTISING_NEWSPAPER_RADIO', 'EQUITY+TV',
       'PANJAVED+DIGITAL', 'PANJAVED+IN STORE', 'PANJAVED+OOH', 'PANJAVED+TV',
       'SALT+DIGITAL', 'SALT+IN STORE',
       'SALT+OTHERADVERTISING_NEWSPAPER_RADIO', 'SALT+TV', 'TOTAL+DIGITAL',
       'TOTAL+IN STORE', 'TOTAL+OTHERADVERTISING_NEWSPAPER_RADIO', 'TOTAL+TV']

# Control variables
control_vars = ['aided_awareness', 'bigpack', 'buyget', 'chim_shop_chai',
 'closeness_top_3_box', 'consumer_confidence_index',
 'darlie_cdc_basepriceratio', 'darlie_discount', 'extravol',
 'extravol_darlie', 'free_promo', 'free_promo_darlie',
 'free_with_others', 'free_with_others_darlie', 'freepremium', 'loyalty',
 'nd', 'near_pack', 'panic_pantry_loading', 'pmd', 'rl',
 'target_baseprice', 'target_discount', 'target_price', 'theme',
 'top_of_mind_brand_awareness', 'total_unaided_brand_awareness',
 'tourist_arrivals_in_million', 'tpr', 'tpr_darlie', 'trendline', 'wd',
 'week13_si', 'week5_si', 'welfare', 'welfare_additional_boost']

# Data columns
data_cols = ['week_date']

# Revenue column
rev_cols = ['weekly_revenue']

# All required columns
req_cols = data_cols + media_vars + control_vars + rev_cols

# Keep only required columns
input_df = input_df[req_cols]

print(f"Data shape: {input_df.shape}")
print(f"Date range: {input_df['week_date'].min()} to {input_df['week_date'].max()}")

# ============================================================================
# 3. ADSTOCK TRANSFORMATION
# ============================================================================

def apply_geometric_adstock(input_df, media_columns, decay_rate):
    """Apply geometric adstock transformation to media variables"""
    df_adstocked = input_df.copy()
    
    if not 0 <= decay_rate <= 1:
        raise ValueError("Decay rate must be between 0 and 1")
    
    for media_col in media_columns:
        adstocked_values = []
        adstock = 0
        
        for value in input_df[media_col]:
            adstock = value + decay_rate * adstock
            adstocked_values.append(adstock)
        
        df_adstocked[media_col] = adstocked_values
    
    return df_adstocked

print("\nApplying adstock transformation...")
decay_rate = 0.5
adstocked_df = apply_geometric_adstock(input_df, media_vars, decay_rate)

# ============================================================================
# 4. SATURATION TRANSFORMATION (HILL FUNCTION)
# ============================================================================

def apply_hill_saturation(input_df, media_columns, alpha, gamma):
    """Apply Hill saturation transformation to media variables"""
    df_transformed = input_df.copy()
    
    if alpha <= 0:
        raise ValueError("Alpha (half-saturation point) must be positive")
    if gamma <= 0:
        raise ValueError("Gamma (shape parameter) must be positive")
    
    for media_col in media_columns:
        x = input_df[media_col].values
        x = np.maximum(x, 1e-10)  # Avoid division by zero
        
        # Hill transformation
        numerator = alpha * (x ** gamma)
        denominator = (alpha ** gamma) + (x ** gamma)
        hill_transformed = numerator / denominator
        
        df_transformed[media_col] = hill_transformed
    
    return df_transformed

print("Applying Hill saturation transformation...")
alpha = 100
gamma = 2
hill_df = apply_hill_saturation(adstocked_df, media_vars, alpha, gamma)

# ============================================================================
# 5. Z-SCALING (STANDARDIZATION)
# ============================================================================

print("Applying z-scaling...")
all_features = media_vars + control_vars

# Initialize scaler
feature_scaler = StandardScaler()

# Create copy for scaling
final_df = hill_df.copy()

# Scale features
scaled_features = feature_scaler.fit_transform(final_df[all_features])
final_df[all_features] = scaled_features

print("✅ Data preprocessing complete")

# ============================================================================
# 6. PREPARE DATA FOR MODELING
# ============================================================================

df = final_df.copy()

# Extract arrays
y = df['weekly_revenue'].values
X_media = df[media_vars].values
X_control = df[control_vars].values

n_obs = len(y)
n_media = len(media_vars)
n_control = len(control_vars)

print(f"\nModel dimensions:")
print(f"  Observations: {n_obs}")
print(f"  Media channels: {n_media}")
print(f"  Control variables: {n_control}")

# ============================================================================
# 7. BUILD BAYESIAN MMM MODEL
# ============================================================================

print("\nBuilding Bayesian MMM model...")

with pm.Model() as mmm_model:
    
    # Priors
    intercept = pm.Normal('intercept', mu=y.mean(), sigma=y.std())
    beta_media = pm.HalfNormal('beta_media', sigma=1, shape=n_media)
    beta_control = pm.Normal('beta_control', mu=0, sigma=1, shape=n_control)
    sigma = pm.HalfNormal('sigma', sigma=y.std())
    
    # Likelihood
    mu = (intercept + 
          pm.math.dot(X_media, beta_media) + 
          pm.math.dot(X_control, beta_control))
    
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    
    # Sampling
    print("Sampling from posterior (this may take a few minutes)...")
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        return_inferencedata=True,
        random_seed=42
    )
    
    # Posterior predictive
    post_pred = pm.sample_posterior_predictive(trace, random_seed=42)

print("✅ Model training complete")

# ============================================================================
# 8. MODEL DIAGNOSTICS
# ============================================================================

print("\n" + "="*80)
print("MODEL CONVERGENCE DIAGNOSTICS")
print("="*80)

# R-hat (should be < 1.01)
rhat = az.rhat(trace)
print(f"Max R-hat: {rhat.to_array().max().item():.4f}")
print(f"Variables with R-hat > 1.01: {(rhat.to_array() > 1.01).sum().item()}")

# Effective sample size
ess = az.ess(trace)
print(f"Min ESS: {ess.to_array().min().item():.0f}")

# Save trace plots
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
az.plot_trace(trace, var_names=['intercept', 'sigma'], axes=axes[:2])
plt.tight_layout()
plt.savefig('trace_plots_base.png', dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(20, 12))
az.plot_trace(trace, var_names=['beta_media'], compact=False)
plt.tight_layout()
plt.savefig('trace_plots_media.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Trace plots saved")

# ============================================================================
# 9. VALIDATION METRIC 1: R² SCORE
# ============================================================================

print("\n" + "="*80)
print("VALIDATION 1: MODEL PERFORMANCE METRICS")
print("="*80)

# Get predictions
y_pred_mean = post_pred.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values

# Calculate metrics
r2 = r2_score(y, y_pred_mean)
mae = mean_absolute_error(y, y_pred_mean)
rmse = np.sqrt(mean_squared_error(y, y_pred_mean))
mape = np.mean(np.abs((y - y_pred_mean) / y)) * 100

print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAPE: {mape:.2f}%")

# ============================================================================
# 10. VALIDATION METRIC 2: ACTUAL VS PREDICTED TRENDLINE
# ============================================================================

print("\n" + "="*80)
print("VALIDATION 2: ACTUAL VS PREDICTED TRENDLINE")
print("="*80)

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y, y_pred_mean, alpha=0.6, edgecolors='k', linewidth=0.5)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
ax.set_xlabel('Actual Revenue', fontsize=12)
ax.set_ylabel('Predicted Revenue', fontsize=12)
ax.set_title(f'Actual vs Predicted Revenue (R² = {r2:.4f})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# Time-series trendline plot
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df['week_date'], y, label='Actual Revenue', linewidth=2, marker='o', markersize=4, alpha=0.7)
ax.plot(df['week_date'], y_pred_mean, label='Predicted Revenue', linewidth=2, marker='x', markersize=4, alpha=0.7)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Weekly Revenue', fontsize=12)
ax.set_title(f'Actual vs Predicted Revenue Over Time (R² = {r2:.4f})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('actual_vs_predicted_trendline.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Trendline plots saved")
print("   - actual_vs_predicted_scatter.png")
print("   - actual_vs_predicted_trendline.png")

# ============================================================================
# 11. VALIDATION METRIC 3: ROI FOR Q2 PERIOD (USING ACTUAL SPEND)
# ============================================================================

print("\n" + "="*80)
print("VALIDATION 3: ROI ANALYSIS FOR Q2 PERIOD")
print("="*80)

# Define Q2 period
q2_start_date = '2019-07-01'
q2_end_date = '2019-09-30'

# Filter Q2 data from ORIGINAL input_df (before transformations)
q2_mask = (input_df['week_date'] >= q2_start_date) & (input_df['week_date'] <= q2_end_date)
q2_data = input_df[q2_mask].copy()

print(f"\nQ2 Period: {q2_start_date} to {q2_end_date}")
print(f"Q2 weeks: {len(q2_data)}")

# Get CDC channels only
cdc_channels = [ch for ch in media_vars if ch.startswith('CDC+')]
print(f"CDC Channels: {cdc_channels}")

# Get posterior mean coefficients
beta_media_mean = trace.posterior['beta_media'].mean(dim=['chain', 'draw']).values

# Get Q2 indices in the full dataset
q2_indices = np.where(q2_mask)[0]

# Calculate ROI for each CDC channel
roi_results = []

for i, channel in enumerate(media_vars):
    if channel in cdc_channels:
        # Get ACTUAL SPEND for Q2 period (from original input_df)
        actual_spend_q2 = q2_data[channel].sum()
        
        if actual_spend_q2 > 0:
            # Calculate revenue contribution for Q2 period
            # Using transformed features from X_media but only for Q2 weeks
            revenue_generated_q2 = (beta_media_mean[i] * X_media[q2_indices, i]).sum()
            
            # Calculate ROAS and ROI
            roas = revenue_generated_q2 / actual_spend_q2
            roi_pct = ((revenue_generated_q2 - actual_spend_q2) / actual_spend_q2) * 100
            profit = revenue_generated_q2 - actual_spend_q2
            
            roi_results.append({
                'Channel': channel,
                'Q2_Actual_Spend': actual_spend_q2,
                'Q2_Revenue_Generated': revenue_generated_q2,
                'ROAS': roas,
                'ROI_%': roi_pct,
                'Profit': profit
            })

# Create ROI dataframe
roi_df = pd.DataFrame(roi_results).sort_values('ROAS', ascending=False)

print("\n" + "-"*80)
print("Q2 ROI RESULTS (CDC CHANNELS)")
print("-"*80)
print(roi_df.to_string(index=False))

# Calculate aggregate metrics
if len(roi_df) > 0:
    total_spend = roi_df['Q2_Actual_Spend'].sum()
    total_revenue = roi_df['Q2_Revenue_Generated'].sum()
    total_profit = roi_df['Profit'].sum()
    
    aggregate_roas = total_revenue / total_spend if total_spend > 0 else 0
    aggregate_roi = ((total_revenue - total_spend) / total_spend * 100) if total_spend > 0 else 0
    
    print("\n" + "-"*80)
    print("AGGREGATE Q2 CDC METRICS")
    print("-"*80)
    print(f"Total Q2 Spend:              ${total_spend:,.2f}")
    print(f"Total Q2 Revenue Generated:  ${total_revenue:,.2f}")
    print(f"Total Q2 Profit:             ${total_profit:,.2f}")
    print(f"Aggregate ROAS:              {aggregate_roas:.2f}")
    print(f"Aggregate ROI:               {aggregate_roi:.2f}%")

# Save ROI results
roi_df.to_csv('q2_roi_analysis.csv', index=False)
print("\n✅ ROI results saved to: q2_roi_analysis.csv")

# Visualize Q2 ROI
if len(roi_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROAS chart
    colors_roas = ['green' if x >= 1 else 'red' for x in roi_df['ROAS']]
    axes[0].barh(roi_df['Channel'], roi_df['ROAS'], color=colors_roas, alpha=0.7)
    axes[0].axvline(x=1, color='black', linestyle='--', linewidth=2, label='Break-even (ROAS=1)')
    axes[0].set_xlabel('Return on Ad Spend (ROAS)', fontsize=12)
    axes[0].set_title('Q2 CDC Channels - ROAS', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # ROI % chart
    colors_roi = ['green' if x >= 0 else 'red' for x in roi_df['ROI_%']]
    axes[1].barh(roi_df['Channel'], roi_df['ROI_%'], color=colors_roi, alpha=0.7)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Break-even (ROI=0%)')
    axes[1].set_xlabel('Return on Investment (%)', fontsize=12)
    axes[1].set_title('Q2 CDC Channels - ROI %', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('q2_roi_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ ROI visualization saved to: q2_roi_visualization.png")

# ============================================================================
# 12. SAVE MODEL OUTPUTS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODEL OUTPUTS")
print("="*80)

# Save model summary
summary = az.summary(trace, var_names=['intercept', 'beta_media', 'beta_control', 'sigma'])
summary.to_csv('mmm_summary.csv')
print("✅ Model summary saved to: mmm_summary.csv")

# Save media contributions
beta_media_samples = trace.posterior['beta_media'].values.reshape(-1, n_media)
contributions = []
for i, channel in enumerate(media_vars):
    contrib = beta_media_samples[:, i].mean() * X_media[:, i].mean()
    contributions.append(contrib)

contribution_df = pd.DataFrame({
    'Channel': media_vars,
    'Contribution': contributions,
    'Contribution_pct': 100 * np.array(contributions) / np.sum(contributions)
}).sort_values('Contribution', ascending=False)

contribution_df.to_csv('media_contributions.csv', index=False)
print("✅ Media contributions saved to: media_contributions.csv")

# ============================================================================
# 13. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("MMM MODELING COMPLETE!")
print("="*80)

print("\n📊 VALIDATION RESULTS:")
print(f"   1. R² Score: {r2:.4f}")
print(f"   2. Trendline plots: Generated ✅")
print(f"   3. Q2 ROI Analysis: Complete ✅")

print("\n📁 OUTPUT FILES:")
print("   - mmm_summary.csv")
print("   - media_contributions.csv")
print("   - q2_roi_analysis.csv")
print("   - actual_vs_predicted_scatter.png")
print("   - actual_vs_predicted_trendline.png")
print("   - q2_roi_visualization.png")
print("   - trace_plots_base.png")
print("   - trace_plots_media.png")

print("\n" + "="*80)
