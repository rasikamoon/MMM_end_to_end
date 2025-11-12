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
from sklearn.linear_model import LinearRegression
import pymc as pm
import arviz as az
import json


def load_config(config_path='mmm_config.json'):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def apply_geometric_adstock(input_df, media_columns, decay_rates):
    """
    Apply geometric adstock transformation to media variables.
    decay_rates can be a single float or a dict with channel-specific rates.
    """
    df_adstocked = input_df.copy()
    
    for media_col in media_columns:
        # Get decay rate for this channel (use default if not specified)
        if isinstance(decay_rates, dict):
            decay_rate = decay_rates.get(media_col, 0.5)
        else:
            decay_rate = decay_rates
            
        if not 0 <= decay_rate <= 1:
            raise ValueError(f"Decay rate for {media_col} must be between 0 and 1")
        
        adstocked_values = []
        adstock = 0
        
        for value in input_df[media_col]:
            adstock = value + decay_rate * adstock
            adstocked_values.append(adstock)
        
        df_adstocked[media_col] = adstocked_values
    
    return df_adstocked


def apply_hill_saturation(input_df, media_columns, saturation_params):
    """
    Apply Hill saturation transformation to media variables.
    saturation_params can be a dict with 'alpha' and 'gamma', or a dict of dicts for channel-specific params.
    """
    df_transformed = input_df.copy()
    
    for media_col in media_columns:
        # Get parameters for this channel
        if 'alpha' in saturation_params and 'gamma' in saturation_params:
            # Single set of parameters for all channels
            alpha = saturation_params['alpha']
            gamma = saturation_params['gamma']
        elif media_col in saturation_params:
            # Channel-specific parameters
            alpha = saturation_params[media_col]['alpha']
            gamma = saturation_params[media_col]['gamma']
        else:
            # Use default
            alpha = 100
            gamma = 2
        
        if alpha <= 0:
            raise ValueError(f"Alpha for {media_col} must be positive")
        if gamma <= 0:
            raise ValueError(f"Gamma for {media_col} must be positive")
        
        x = input_df[media_col].values
        x = np.maximum(x, 1e-10)  # Avoid division by zero
        
        # Hill transformation
        numerator = alpha * (x ** gamma)
        denominator = (alpha ** gamma) + (x ** gamma)
        hill_transformed = numerator / denominator
        
        df_transformed[media_col] = hill_transformed
    
    return df_transformed


if __name__ == '__main__':
    
    # ============================================================================
    # 0. LOAD CONFIGURATION
    # ============================================================================
    
    print("Loading configuration from mmm_config.json...")
    config = load_config('mmm_config.json')
    
    # Extract config sections
    model_config = config['model_settings']  # Load full model_settings (includes priors and mcmc_sampling)
    mcmc_config = model_config['mcmc_sampling']  # Extract MCMC settings separately
    adstock_config = config['adstock_parameters']
    saturation_config = config['saturation_parameters']
    roi_config = config['roi_analysis']
    data_config = config['data_settings']
    output_config = config['output_settings']
    
    print("✅ Configuration loaded")
    
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    
    print(f"\nLoading data from {data_config['input_file']}...")
    input_df = pd.read_excel(data_config['input_file'])
    
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
    date_cols = ['week_date']
    
    # Revenue column
    rev_cols = [data_config['revenue_column']]
    
    # All required columns
    req_cols = date_cols + media_vars + control_vars + rev_cols
    
    # Keep only required columns
    input_df = input_df[req_cols]
    
    print(f"Data shape: {input_df.shape}")
    print(f"Date range: {input_df[data_config['date_column']].min()} to {input_df[data_config['date_column']].max()}")
    
    # ============================================================================
    # 3. ADSTOCK TRANSFORMATION
    # ============================================================================
    
    print("\nApplying adstock transformation...")
    print(f"  Using channel-specific decay rates from config")
    
    # Use channel-specific decay rates
    decay_rates = adstock_config['channel_specific']
    adstocked_df = apply_geometric_adstock(input_df, media_vars, decay_rates)
    
    # ============================================================================
    # 4. SATURATION TRANSFORMATION (HILL FUNCTION)
    # ============================================================================
    
    print("Applying Hill saturation transformation...")
    print(f"  Using channel-specific saturation parameters from config")
    
    # Use channel-specific saturation parameters
    saturation_params = saturation_config['channel_specific']
    hill_df = apply_hill_saturation(adstocked_df, media_vars, saturation_params)
    
    # ============================================================================
    # 5. Z-SCALING (STANDARDIZATION)
    # ============================================================================
    
    print("Applying z-scaling...")
    all_features = media_vars + control_vars
    
    # Initialize scaler
    feature_scaler = StandardScaler()
    
    # Create copy for scaling
    final_df = hill_df.copy()
    
    # IMPORTANT: Save unscaled data for revenue contribution calculation
    # Z-scaled values can be negative, which would give wrong revenue attribution
    X_media_unscaled = hill_df[media_vars].values
    X_control_unscaled = hill_df[control_vars].values
    
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
    X_media = df[media_vars].values  # Scaled for modeling
    X_control = df[control_vars].values
    
    n_obs = len(y)
    n_media = len(media_vars)
    n_control = len(control_vars)
    
    print(f"\nModel dimensions:")
    print(f"  Observations: {n_obs}")
    print(f"  Media channels: {n_media}")
    print(f"  Control variables: {n_control}")
    
    # ============================================================================
    # 7. LINEAR REGRESSION BASELINE MODEL
    # ============================================================================
    
    print("\n" + "="*80)
    print("BASELINE: LINEAR REGRESSION MODEL")
    print("="*80)
    
    # Combine all features for linear regression
    X_all = np.concatenate([X_media, X_control], axis=1)
    
    # Fit OLS linear regression
    lr_model = LinearRegression()
    lr_model.fit(X_all, y)
    
    # Get predictions
    y_pred_lr = lr_model.predict(X_all)
    
    # Calculate metrics
    lr_r2 = r2_score(y, y_pred_lr)
    lr_mae = mean_absolute_error(y, y_pred_lr)
    lr_rmse = np.sqrt(mean_squared_error(y, y_pred_lr))
    lr_mape = np.mean(np.abs((y - y_pred_lr) / y)) * 100
    
    print(f"\nLinear Regression Performance:")
    print(f"  R² Score: {lr_r2:.4f}")
    print(f"  MAE: {lr_mae:,.2f}")
    print(f"  RMSE: {lr_rmse:,.2f}")
    print(f"  MAPE: {lr_mape:.2f}%")
    
    # Extract coefficients
    lr_intercept = lr_model.intercept_
    lr_coef_media = lr_model.coef_[:n_media]
    lr_coef_control = lr_model.coef_[n_media:]
    
    print(f"\nLinear Regression Coefficients:")
    print(f"  Intercept: {lr_intercept:,.2f}")
    print(f"  Media coefficients range: [{lr_coef_media.min():.2f}, {lr_coef_media.max():.2f}]")
    print(f"  Control coefficients range: [{lr_coef_control.min():.2f}, {lr_coef_control.max():.2f}]")
    print(f"  Negative media coefficients: {(lr_coef_media < 0).sum()} out of {n_media}")
    print(f"  Negative control coefficients: {(lr_coef_control < 0).sum()} out of {n_control}")
    
    print("\n✅ Linear regression baseline complete")
    
    # ============================================================================
    # 8. BUILD BAYESIAN MMM MODEL
    # ============================================================================
    
    print("\n" + "="*80)
    print("BAYESIAN MMM MODEL")
    print("="*80)
    
    print("\nBuilding Bayesian MMM model...")
    
    # ============================================================================
    # 6B. LOAD PRIORS FROM CONFIG
    # ============================================================================
    
    print("\nLoading priors from config...")
    prior_config = model_config.get('priors', {})
    
    # Intercept prior
    intercept_config = prior_config.get('intercept', {})
    intercept_mu = y.mean()
    intercept_sigma = y.std() * intercept_config.get('sigma_multiplier', 0.5)
    
    # Observation noise prior
    noise_config = prior_config.get('observation_noise', {})
    noise_sigma = y.std() * noise_config.get('sigma_multiplier', 0.5)
    
    # Control variables prior
    control_config = prior_config.get('control_variables', {})
    control_mu = control_config.get('default_mu', 0)
    control_sigma = control_config.get('default_sigma', 0.1)
    
    # Media channels prior - hierarchical by media type with channel-specific priors
    media_config = prior_config.get('media_channels', {})
    default_media_sigma = media_config.get('default_sigma', 0.5)
    by_media_type = media_config.get('by_media_type', {})
    channel_specific = media_config.get('channel_specific', {})
    
    # Build sigma array for each media channel
    media_sigmas = []
    channels_with_specific_priors = 0
    channels_with_type_priors = 0
    channels_with_default = 0
    
    for channel in media_vars:
        # Check for channel-specific prior first
        if channel in channel_specific:
            sigma = channel_specific[channel]['sigma']
            channels_with_specific_priors += 1
        else:
            # Extract media type from channel name (e.g., "CDC+TV" -> "TV")
            media_type = channel.split('+')[-1]
            # Use media type sigma, or default if not found
            if media_type in by_media_type:
                sigma = by_media_type[media_type]
                channels_with_type_priors += 1
            else:
                sigma = default_media_sigma
                channels_with_default += 1
        media_sigmas.append(sigma)
    
    # DEBUG: Show breakdown
    print(f"\nPrior Assignment Breakdown:")
    print(f"  Channels with specific priors: {channels_with_specific_priors}")
    print(f"  Channels using media type priors: {channels_with_type_priors}")
    print(f"  Channels using default: {channels_with_default}")
    
    print(f"  Intercept: Normal(mu={intercept_mu:.2f}, sigma={intercept_sigma:.2f})")
    print(f"  Media channels: HalfNormal with channel-specific sigmas")
    print(f"    - {len(channel_specific)} channels with specific priors")
    print(f"    - Remaining use media type defaults (TV={by_media_type.get('TV', default_media_sigma)}, DIGITAL={by_media_type.get('DIGITAL', default_media_sigma)}, etc.)")
    print(f"  Control variables: Normal(mu={control_mu}, sigma={control_sigma})")
    print(f"  Observation noise: HalfNormal(sigma={noise_sigma:.2f})")
    
    with pm.Model() as mmm_model:
        
        # Priors from config
        intercept = pm.Normal('intercept', mu=intercept_mu, sigma=intercept_sigma)
        
        # Media coefficients: hierarchical priors by media type
        beta_media = pm.HalfNormal('beta_media', sigma=np.array(media_sigmas), shape=n_media)
        
        # Control variables: tight regularization
        beta_control = pm.Normal('beta_control', mu=control_mu, sigma=control_sigma, shape=n_control)
        
        # Observation noise
        sigma = pm.HalfNormal('sigma', sigma=noise_sigma)
        
        # Likelihood
        mu = (intercept + 
              pm.math.dot(X_media, beta_media) + 
              pm.math.dot(X_control, beta_control))
        
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        # Sampling
        print("Sampling from posterior (this may take a few minutes)...")
        print(f"  Draws: {mcmc_config['draws']}, Tune: {mcmc_config['tune']}, Chains: {mcmc_config['chains']}")
        trace = pm.sample(
            draws=mcmc_config['draws'],
            tune=mcmc_config['tune'],
            chains=mcmc_config['chains'],
            target_accept=mcmc_config['target_accept'],
            return_inferencedata=True,
            random_seed=mcmc_config['random_seed']
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
    plt.savefig(output_config['output_files']['trace_base'], dpi=output_config['plot_dpi'], bbox_inches='tight')
    plt.close()
    
    fig = plt.figure(figsize=(20, 12))
    az.plot_trace(trace, var_names=['beta_media'], compact=False)
    plt.tight_layout()
    plt.savefig(output_config['output_files']['trace_media'], dpi=output_config['plot_dpi'], bbox_inches='tight')
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
    plt.savefig(output_config['output_files']['scatter_plot'], dpi=output_config['plot_dpi'], bbox_inches='tight')
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
    plt.savefig(output_config['output_files']['trendline_plot'], dpi=output_config['plot_dpi'], bbox_inches='tight')
    plt.close()
    
    print("✅ Trendline plots saved")
    print("   - actual_vs_predicted_scatter.png")
    print("   - actual_vs_predicted_trendline.png")
    
    # ============================================================================
    # 11. VALIDATION METRIC 3: ROAS ANALYSIS (USING ACTUAL SPEND)
    # ============================================================================
    
    print("\n" + "="*80)
    print("VALIDATION 3: ROAS ANALYSIS")
    print("="*80)
    
    # Load analysis period from config
    period_config = roi_config['analysis_period']
    start_date = period_config['start_date']
    end_date = period_config['end_date']
    
    # Get period name (use from config or auto-generate)
    if 'period_name' in period_config:
        period_name = period_config['period_name']
    else:
        period_name = f"{start_date} to {end_date}"
    
    # Filter data for analysis period
    period_mask = (input_df[data_config['date_column']] >= start_date) & (input_df[data_config['date_column']] <= end_date)
    period_data = input_df[period_mask].copy()
    
    print(f"\nAnalysis Period: {period_name}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Number of weeks: {len(period_data)}")
    
    # Get target channels from config
    target_channels = roi_config['target_channels']
    
    # If "all", use all media channels
    if target_channels == "all":
        target_channels = media_vars
        
    print(f"Analyzing {len(target_channels)} channels")
    
    # Get posterior mean coefficients
    beta_media_mean = trace.posterior['beta_media'].mean(dim=['chain', 'draw']).values
    
    # Get period indices in the full dataset
    period_indices = np.where(period_mask)[0]
    
    # Calculate ROAS for each channel
    roas_results = []
    
    for i, channel in enumerate(media_vars):
        if channel in target_channels:
            # Get ACTUAL SPEND for analysis period (from original input_df)
            actual_spend_period = period_data[channel].sum()
            
            if actual_spend_period > 0:
                # Calculate revenue contribution for analysis period
                # Using UNSCALED transformed features (after adstock & saturation, before z-scaling)
                # This ensures positive coefficients × positive features = positive revenue
                revenue_generated_period = (beta_media_mean[i] * X_media_unscaled[period_indices, i]).sum()
                
                # Calculate ROAS
                roas = revenue_generated_period / actual_spend_period
                
                roas_results.append({
                    'Channel': channel,
                    'Actual_Spend': actual_spend_period,
                    'Revenue_Generated': revenue_generated_period,
                    'ROAS': roas
                })
    
    # Create ROAS dataframe
    roas_df = pd.DataFrame(roas_results).sort_values('ROAS', ascending=False)
    
    print("\n" + "-"*80)
    print(f"ROAS RESULTS - {period_name.upper()}")
    print("-"*80)
    print(roas_df.to_string(index=False))
    
    # Calculate aggregate metrics
    if len(roas_df) > 0:
        total_spend = roas_df['Actual_Spend'].sum()
        total_revenue = roas_df['Revenue_Generated'].sum()
        
        aggregate_roas = total_revenue / total_spend if total_spend > 0 else 0
        
        print("\n" + "-"*80)
        print(f"AGGREGATE METRICS - {period_name.upper()}")
        print("-"*80)
        print(f"Total Spend:              ${total_spend:,.2f}")
        print(f"Total Revenue Generated:  ${total_revenue:,.2f}")
        print(f"Aggregate ROAS:           {aggregate_roas:.2f}")
    
    # Save ROAS results
    roas_df.to_csv(output_config['output_files']['roi_analysis'], index=False)
    print(f"\n✅ ROAS results saved to: {output_config['output_files']['roi_analysis']}")
    
    # Visualize ROAS
    if len(roas_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 10))  # Taller for all channels
        
        # ROAS chart
        colors_roas = ['green' if x >= 1 else 'red' for x in roas_df['ROAS']]
        ax.barh(roas_df['Channel'], roas_df['ROAS'], color=colors_roas, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Break-even (ROAS=1)')
        ax.set_xlabel('Return on Ad Spend (ROAS)', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        ax.set_title(f'ROAS - All Media Channels ({period_name})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (channel, roas) in enumerate(zip(roas_df['Channel'], roas_df['ROAS'])):
            ax.text(roas, i, f' {roas:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_config['output_files']['roi_plot'], dpi=output_config['plot_dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROAS visualization saved to: {output_config['output_files']['roi_plot']}")
    
    # ============================================================================
    # 11B. REVENUE
    
    # ============================================================================
    
    print("\n" + "="*80)
    print(f"REVENUE DECOMPOSITION ANALYSIS - {period_name.upper()}")
    print("="*80)
    
    # Calculate all contributions for the analysis period
    if len(roas_df) > 0:
        # Get coefficients
        intercept_mean = trace.posterior['intercept'].mean().item()
        beta_control_mean = trace.posterior['beta_control'].mean(dim=['chain', 'draw']).values
        
        # Calculate contributions for the period
        # 1. Base (intercept)
        base_contribution = intercept_mean * len(period_indices)
        
        # 2. Media contribution (already calculated)
        total_media_contribution = roas_df['Revenue_Generated'].sum()
        
        # 3. Control variables contribution
        # IMPORTANT: Use UNSCALED control data for accurate contribution calculation
        X_control_period_unscaled = X_control_unscaled[period_indices, :]
        
        # Get scaling parameters
        control_means = feature_scaler.mean_[n_media:]
        control_stds = feature_scaler.scale_[n_media:]
        
        # Calculate control contribution using unscaled data
        # Need to account for z-scaling: contribution = beta * (X_unscaled - mean) / std
        control_contribution = 0
        positive_control_contrib = 0
        negative_control_contrib = 0
        
        for i in range(n_control):
            # Transform: scaled_value = (unscaled - mean) / std
            # So: contribution = beta * scaled_value = beta * (unscaled - mean) / std
            scaled_values = (X_control_period_unscaled[:, i] - control_means[i]) / control_stds[i]
            contrib = (beta_control_mean[i] * scaled_values).sum()
            control_contribution += contrib
            if contrib > 0:
                positive_control_contrib += contrib
            else:
                negative_control_contrib += contrib
        
        # Total predicted revenue for period
        total_predicted_revenue = base_contribution + total_media_contribution + control_contribution
        
        # Calculate percentages
        media_pct = (total_media_contribution / total_predicted_revenue * 100) if total_predicted_revenue != 0 else 0
        control_pct = (control_contribution / total_predicted_revenue * 100) if total_predicted_revenue != 0 else 0
        base_pct = (base_contribution / total_predicted_revenue * 100) if total_predicted_revenue != 0 else 0
        
        # Print decomposition summary
        print("\n" + "-"*80)
        print("REVENUE DECOMPOSITION SUMMARY")
        print("-"*80)
        print(f"Total Predicted Revenue:     ${total_predicted_revenue:,.2f}")
        print(f"")
        print(f"POSITIVE DRIVERS:")
        print(f"  Base (Intercept):          ${base_contribution:,.2f} ({base_pct:.1f}%)")
        print(f"  Media Total:               ${total_media_contribution:,.2f} ({media_pct:.1f}%)")
        if positive_control_contrib > 0:
            pos_ctrl_pct = (positive_control_contrib / total_predicted_revenue * 100) if total_predicted_revenue != 0 else 0
            print(f"  Positive Controls:         ${positive_control_contrib:,.2f} ({pos_ctrl_pct:.1f}%)")
        print(f"")
        if negative_control_contrib < 0:
            neg_ctrl_pct = (negative_control_contrib / total_predicted_revenue * 100) if total_predicted_revenue != 0 else 0
            print(f"NEGATIVE DRIVERS:")
            print(f"  Negative Controls:         ${negative_control_contrib:,.2f} ({neg_ctrl_pct:.1f}%)")
            print(f"")
        print(f"NET CONTROL CONTRIBUTION:    ${control_contribution:,.2f} ({control_pct:.1f}%)")
        
        # ========================================================================
        # PIE CHART 1: Media vs Non-Media
        # ========================================================================
        print("\n" + "-"*80)
        print("Creating Pie Chart 1: Media vs Non-Media Split")
        print("-"*80)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        non_media_contribution = control_contribution + base_contribution
        non_media_pct = control_pct + base_pct
        
        labels = ['Media', 'Non-Media\n(Controls + Base)']
        sizes = [media_pct, non_media_pct]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)  # Explode media slice
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_weight('bold')
        
        ax.set_title(f'Media vs Non-Media Revenue Contribution ({period_name})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend with values
        legend_labels = [
            f'Media: ${total_media_contribution:,.0f}',
            f'Non-Media: ${non_media_contribution:,.0f}'
        ]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(output_config['output_files']['decomp_media_vs_nonmedia'], dpi=output_config['plot_dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"✅ Media vs Non-Media pie chart saved")
        
        # ========================================================================
        # WATERFALL CHART: Revenue Build-Up
        # ========================================================================
        print("\n" + "-"*80)
        print("Creating Waterfall Chart: Revenue Build-Up")
        print("-"*80)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare waterfall data
        categories = ['Base', 'Media', 'Positive\nControls', 'Negative\nControls', 'Total']
        values = [
            base_contribution,
            total_media_contribution,
            positive_control_contrib if positive_control_contrib > 0 else 0,
            negative_control_contrib if negative_control_contrib < 0 else 0,
            total_predicted_revenue
        ]
        
        # Calculate cumulative values for waterfall
        cumulative = [0]
        for i, val in enumerate(values[:-1]):
            cumulative.append(cumulative[-1] + val)
        
        # Colors: green for positive, red for negative, blue for total
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#34495e']
        
        # Plot bars
        for i in range(len(categories)):
            if i == len(categories) - 1:  # Total bar
                ax.bar(i, values[i], color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)
            else:
                if values[i] >= 0:
                    ax.bar(i, values[i], bottom=cumulative[i], color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
                else:
                    ax.bar(i, -values[i], bottom=cumulative[i] + values[i], color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add connecting lines
        for i in range(len(categories) - 1):
            ax.plot([i + 0.4, i + 0.6], [cumulative[i+1], cumulative[i+1]], 'k--', linewidth=1, alpha=0.5)
        
        # Add value labels
        for i, (cat, val) in enumerate(zip(categories, values)):
            if i == len(categories) - 1:  # Total
                label_y = val / 2
            else:
                if val >= 0:
                    label_y = cumulative[i] + val / 2
                else:
                    label_y = cumulative[i] + val / 2
            
            ax.text(i, label_y, f'${val:,.0f}', ha='center', va='center', 
                   fontweight='bold', fontsize=10, color='white' if abs(val) > total_predicted_revenue * 0.1 else 'black')
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title(f'Revenue Waterfall: Build-Up Analysis ({period_name})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if abs(x) >= 1e6 else f'${x/1e3:.0f}K'))
        
        plt.tight_layout()
        plt.savefig(output_config['output_files']['decomp_waterfall'], dpi=output_config['plot_dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"✅ Waterfall chart saved to: {output_config['output_files']['decomp_waterfall']}")
    
    # ============================================================================
    # 11C. MEDIA CHANNEL CONTRIBUTION DISTRIBUTION
    # ============================================================================
    
    print("\n" + "="*80)
    print(f"MEDIA CHANNEL BREAKDOWN - {period_name.upper()}")
    print("="*80)
    
    # Calculate contribution distribution for media channels
    if len(roas_df) > 0:
        # Create contribution dataframe
        contribution_dist_df = roas_df.copy()
        
        # Calculate percentage contribution
        total_revenue_generated = contribution_dist_df['Revenue_Generated'].sum()
        contribution_dist_df['Revenue_Contribution_%'] = (
            contribution_dist_df['Revenue_Generated'] / total_revenue_generated * 100
        )
        
        # Sort by revenue contribution
        contribution_dist_df = contribution_dist_df.sort_values('Revenue_Generated', ascending=False)
        
        # Display table
        print("\n" + "-"*80)
        print("CHANNEL REVENUE CONTRIBUTION")
        print("-"*80)
        display_df = contribution_dist_df[['Channel', 'Revenue_Generated', 'Revenue_Contribution_%']].copy()
        display_df['Revenue_Generated'] = display_df['Revenue_Generated'].apply(lambda x: f"${x:,.2f}")
        display_df['Revenue_Contribution_%'] = display_df['Revenue_Contribution_%'].apply(lambda x: f"{x:.2f}%")
        print(display_df.to_string(index=False))
        
        # Summary stats
        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)
        print(f"Total Revenue Generated:     ${total_revenue_generated:,.2f}")
        print(f"Number of Channels:          {len(contribution_dist_df)}")
        print(f"Top Channel:                 {contribution_dist_df.iloc[0]['Channel']}")
        print(f"Top Channel Contribution:    ${contribution_dist_df.iloc[0]['Revenue_Generated']:,.2f} ({contribution_dist_df.iloc[0]['Revenue_Contribution_%']:.2f}%)")
        
        # Save contribution distribution CSV
        contribution_csv = contribution_dist_df[['Channel', 'Revenue_Generated', 'Revenue_Contribution_%']].copy()
        contribution_csv.to_csv(output_config['output_files']['contribution_distribution'], index=False)
        print(f"\n✅ Contribution distribution saved to: {output_config['output_files']['contribution_distribution']}")
        
        # ========================================================================
        # PIE CHART 3: Media Channel Breakdown (only positive contributions)
        # ========================================================================
        print("\n" + "-"*80)
        print("Creating Pie Chart 3: Media Channel Breakdown")
        print("-"*80)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Filter to only positive contributions for pie chart
        positive_contrib = contribution_dist_df[contribution_dist_df['Revenue_Generated'] > 0].copy()
        
        if len(positive_contrib) > 0:
            # Recalculate percentages for positive contributions only
            total_positive_revenue = positive_contrib['Revenue_Generated'].sum()
            positive_contrib['Positive_Contribution_%'] = (
                positive_contrib['Revenue_Generated'] / total_positive_revenue * 100
            )
            
            # Filter out very small contributions for better visualization
            min_percentage = 1.0  # Show only channels with >1% contribution
            major_channels = positive_contrib[positive_contrib['Positive_Contribution_%'] >= min_percentage].copy()
            other_channels = positive_contrib[positive_contrib['Positive_Contribution_%'] < min_percentage].copy()
            
            if len(other_channels) > 0:
                # Add "Others" category
                other_row = pd.DataFrame({
                    'Channel': ['Others'],
                    'Revenue_Generated': [other_channels['Revenue_Generated'].sum()],
                    'Positive_Contribution_%': [other_channels['Positive_Contribution_%'].sum()]
                })
                plot_df = pd.concat([major_channels, other_row], ignore_index=True)
            else:
                plot_df = major_channels
            
            # Create pie chart
            colors = plt.cm.Set3(range(len(plot_df)))
            wedges, texts, autotexts = ax.pie(
                plot_df['Positive_Contribution_%'],
                labels=plot_df['Channel'],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 10}
            )
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            ax.set_title(f'Positive Revenue Contribution by Channel ({period_name})', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(output_config['output_files']['contribution_pie'], dpi=output_config['plot_dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"✅ Contribution pie chart saved to: {output_config['output_files']['contribution_pie']}")
        else:
            print("⚠️  No positive contributions to plot in pie chart")
        
        # Create horizontal bar chart for better readability
        fig, ax = plt.subplots(figsize=(14, 10))
        
        colors_contrib = plt.cm.viridis(np.linspace(0.3, 0.9, len(contribution_dist_df)))
        bars = ax.barh(
            contribution_dist_df['Channel'],
            contribution_dist_df['Revenue_Contribution_%'],
            color=colors_contrib,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('Revenue Contribution (%)', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        ax.set_title(f'Revenue Contribution Distribution ({period_name})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (channel, pct) in enumerate(zip(contribution_dist_df['Channel'], contribution_dist_df['Revenue_Contribution_%'])):
            ax.text(pct, i, f' {pct:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_config['output_files']['contribution_bar'], dpi=output_config['plot_dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"✅ Contribution bar chart saved to: {output_config['output_files']['contribution_bar']}")
    
    # ============================================================================
    # 12. SAVE MODEL OUTPUTS
    # ============================================================================
    
    print("\n" + "="*80)
    print("SAVING MODEL OUTPUTS")
    print("="*80)
    
    # Save model summary
    summary = az.summary(trace, var_names=['intercept', 'beta_media', 'beta_control', 'sigma'])
    summary.to_csv(output_config['output_files']['model_summary'])
    print(f"✅ Model summary saved to: {output_config['output_files']['model_summary']}")
    
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
    
    contribution_df.to_csv(output_config['output_files']['media_contributions'], index=False)
    print(f"✅ Media contributions saved to: {output_config['output_files']['media_contributions']}")
    
    # ============================================================================
    # 13. MODEL COMPARISON: LINEAR REGRESSION vs BAYESIAN MMM
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL COMPARISON: LINEAR REGRESSION vs BAYESIAN MMM")
    print("="*80)
    
    # Create comparison table
    comparison_data = {
        'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE (%)'],
        'Linear Regression': [
            f"{lr_r2:.4f}",
            f"${lr_mae:,.2f}",
            f"${lr_rmse:,.2f}",
            f"{lr_mape:.2f}%"
        ],
        'Bayesian MMM': [
            f"{r2:.4f}",
            f"${mae:,.2f}",
            f"${rmse:,.2f}",
            f"{mape:.2f}%"
        ],
        'Winner': [
            'Bayesian MMM' if r2 > lr_r2 else 'Linear Regression',
            'Bayesian MMM' if mae < lr_mae else 'Linear Regression',
            'Bayesian MMM' if rmse < lr_rmse else 'Linear Regression',
            'Bayesian MMM' if mape < lr_mape else 'Linear Regression'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "-"*80)
    print("PERFORMANCE COMPARISON")
    print("-"*80)
    print(comparison_df.to_string(index=False))
    
    # Calculate improvement
    r2_improvement = ((r2 - lr_r2) / abs(lr_r2) * 100) if lr_r2 != 0 else 0
    mae_improvement = ((lr_mae - mae) / lr_mae * 100) if lr_mae != 0 else 0
    rmse_improvement = ((lr_rmse - rmse) / lr_rmse * 100) if lr_rmse != 0 else 0
    mape_improvement = ((lr_mape - mape) / lr_mape * 100) if lr_mape != 0 else 0
    
    print("\n" + "-"*80)
    print("IMPROVEMENT SUMMARY")
    print("-"*80)
    print(f"R² Improvement:    {r2_improvement:+.2f}%")
    print(f"MAE Improvement:   {mae_improvement:+.2f}%")
    print(f"RMSE Improvement:  {rmse_improvement:+.2f}%")
    print(f"MAPE Improvement:  {mape_improvement:+.2f}%")
    
    # Key differences
    print("\n" + "-"*80)
    print("KEY DIFFERENCES")
    print("-"*80)
    print(f"Linear Regression:")
    print(f"  ✓ Fast training (instant)")
    print(f"  ✓ Simple interpretation")
    print(f"  ✗ No uncertainty quantification")
    print(f"  ✗ Can produce negative media coefficients: {(lr_coef_media < 0).sum()}/{n_media}")
    print(f"  ✗ No prior knowledge incorporation")
    print(f"  ✗ Overfitting risk with many features")
    
    print(f"\nBayesian MMM:")
    print(f"  ✓ Uncertainty quantification (credible intervals)")
    print(f"  ✓ Enforces positive media coefficients (HalfNormal prior)")
    print(f"  ✓ Incorporates domain knowledge via priors")
    print(f"  ✓ Better regularization")
    print(f"  ✗ Slower training (~{mcmc_config['draws']} draws)")
    print(f"  ✗ More complex interpretation")
    
    # Save comparison
    comparison_df.to_csv('model_comparison.csv', index=False)
    print(f"\n✅ Model comparison saved to: model_comparison.csv")
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. R² Comparison
    ax1 = axes[0, 0]
    models = ['Linear\nRegression', 'Bayesian\nMMM']
    r2_scores = [lr_r2, r2]
    colors_r2 = ['#3498db', '#2ecc71']
    bars1 = ax1.bar(models, r2_scores, color=colors_r2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(r2_scores) - 0.1, max(r2_scores) + 0.1])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 2. MAE Comparison
    ax2 = axes[0, 1]
    mae_scores = [lr_mae, mae]
    colors_mae = ['#e74c3c' if lr_mae > mae else '#3498db', '#2ecc71' if mae < lr_mae else '#e74c3c']
    bars2 = ax2.bar(models, mae_scores, color=colors_mae, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('MAE ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Absolute Error Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if abs(x) >= 1e6 else f'${x/1e3:.0f}K'))
    for bar, score in zip(bars2, mae_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${score:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. RMSE Comparison
    ax3 = axes[1, 0]
    rmse_scores = [lr_rmse, rmse]
    colors_rmse = ['#e74c3c' if lr_rmse > rmse else '#3498db', '#2ecc71' if rmse < lr_rmse else '#e74c3c']
    bars3 = ax3.bar(models, rmse_scores, color=colors_rmse, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
    ax3.set_title('Root Mean Squared Error Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if abs(x) >= 1e6 else f'${x/1e3:.0f}K'))
    for bar, score in zip(bars3, rmse_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${score:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. MAPE Comparison
    ax4 = axes[1, 1]
    mape_scores = [lr_mape, mape]
    colors_mape = ['#e74c3c' if lr_mape > mape else '#3498db', '#2ecc71' if mape < lr_mape else '#e74c3c']
    bars4 = ax4.bar(models, mape_scores, color=colors_mape, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Mean Absolute Percentage Error Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars4, mape_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('model_comparison_metrics.png', dpi=output_config['plot_dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model comparison visualization saved to: model_comparison_metrics.png")
    
    # ============================================================================
    # 14. FINAL SUMMARY
    # ============================================================================
    
    print("\n" + "="*80)
    print("MMM MODELING COMPLETE!")
    print("="*80)
    
    print("\n📊 VALIDATION RESULTS:")
    print(f"   1. Linear Regression R²: {lr_r2:.4f}")
    print(f"   2. Bayesian MMM R²: {r2:.4f}")
    print(f"   2. Trendline plots: Generated ✅")
    print(f"   3. ROAS Analysis ({period_name}): Complete ✅")
    print(f"   4. Revenue Decomposition (Media vs Non-Media): Complete ✅")
    print(f"   5. Media Channel Breakdown: Complete ✅")
    
    print("\n📁 OUTPUT FILES:")
    print(f"\n  Model Comparison: ⭐")
    print(f"   - model_comparison.csv (Metrics comparison table)")
    print(f"   - model_comparison_metrics.png (Visual comparison)")
    print(f"\n  Model Diagnostics:")
    print(f"   - {output_config['output_files']['model_summary']}")
    print(f"   - {output_config['output_files']['trace_base']}")
    print(f"   - {output_config['output_files']['trace_media']}")
    print(f"\n  Model Performance:")
    print(f"   - {output_config['output_files']['scatter_plot']}")
    print(f"   - {output_config['output_files']['trendline_plot']}")
    print(f"\n  ROAS Analysis:")
    print(f"   - {output_config['output_files']['roi_analysis']}")
    print(f"   - {output_config['output_files']['roi_plot']}")
    print(f"\n  Revenue Decomposition:")
    print(f"   - {output_config['output_files']['decomp_waterfall']} (Waterfall: Full Revenue Build-Up)")
    print(f"   - {output_config['output_files']['decomp_media_vs_nonmedia']} (Pie: Media vs Non-Media)")
    print(f"   - {output_config['output_files']['contribution_pie']} (Pie: Media Channel Breakdown)")
    print(f"   - {output_config['output_files']['contribution_bar']} (Bar: Media Channel Breakdown)")
    print(f"   - {output_config['output_files']['contribution_distribution']}")
    print(f"\n  Media Contributions:")
    print(f"   - {output_config['output_files']['media_contributions']}")
    
    print("\n" + "="*80)
