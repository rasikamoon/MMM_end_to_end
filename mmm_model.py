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
    model_config = config['model_settings']['mcmc_sampling']
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
    data_cols = ['week_date']
    
    # Revenue column
    rev_cols = [data_config['revenue_column']]
    
    # All required columns
    req_cols = data_cols + media_vars + control_vars + rev_cols
    
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
    
    # IMPORTANT: Save unscaled media data for ROAS calculation
    # Z-scaled values can be negative, which would give wrong revenue attribution
    X_media_unscaled = hill_df[media_vars].values
    
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
    # 7. BUILD BAYESIAN MMM MODEL
    # ============================================================================
    
    print("\nBuilding Bayesian MMM model...")
    
    with pm.Model() as mmm_model:
        
        # Priors - Improved for better convergence and domain constraints
        # Intercept: centered at mean revenue with reasonable variance
        intercept = pm.Normal('intercept', mu=y.mean(), sigma=y.std() * 0.5)
        
        # Media coefficients: MUST be positive (media spend increases revenue)
        # Using HalfNormal with smaller sigma for regularization
        beta_media = pm.HalfNormal('beta_media', sigma=2, shape=n_media)
        
        # Control variables: can be positive or negative
        # Using tighter prior for regularization
        beta_control = pm.Normal('beta_control', mu=0, sigma=0.5, shape=n_control)
        
        # Observation noise: positive only
        sigma = pm.HalfNormal('sigma', sigma=y.std() * 0.5)
        
        # Likelihood
        mu = (intercept + 
              pm.math.dot(X_media, beta_media) + 
              pm.math.dot(X_control, beta_control))
        
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        # Sampling
        print("Sampling from posterior (this may take a few minutes)...")
        print(f"  Draws: {model_config['draws']}, Tune: {model_config['tune']}, Chains: {model_config['chains']}")
        trace = pm.sample(
            draws=model_config['draws'],
            tune=model_config['tune'],
            chains=model_config['chains'],
            target_accept=model_config['target_accept'],
            return_inferencedata=True,
            random_seed=model_config['random_seed']
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
    # 11B. REVENUE DECOMPOSITION ANALYSIS
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
        # Need to get unscaled control data for period
        X_control_period = X_control[period_indices, :]
        control_contribution = (beta_control_mean * X_control_period).sum()
        
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
        print(f"Base (Intercept):            ${base_contribution:,.2f} ({base_pct:.1f}%)")
        print(f"Media Contribution:          ${total_media_contribution:,.2f} ({media_pct:.1f}%)")
        print(f"Control Variables:           ${control_contribution:,.2f} ({control_pct:.1f}%)")
        
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
        plt.savefig('revenue_decomp_media_vs_nonmedia.png', dpi=output_config['plot_dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"✅ Media vs Non-Media pie chart saved")
        
        # ========================================================================
        # PIE CHART 2: Full Decomposition (Base + Media + Controls)
        # ========================================================================
        print("\n" + "-"*80)
        print("Creating Pie Chart 2: Full Decomposition")
        print("-"*80)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = ['Base\n(Intercept)', 'Media', 'Control\nVariables']
        sizes = [base_pct, media_pct, control_pct]
        colors = ['#99ff99', '#ff9999', '#ffcc99']
        explode = (0, 0.1, 0)  # Explode media slice
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)
            autotext.set_weight('bold')
        
        ax.set_title(f'Revenue Decomposition: Base + Media + Controls ({period_name})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend with values
        legend_labels = [
            f'Base: ${base_contribution:,.0f}',
            f'Media: ${total_media_contribution:,.0f}',
            f'Controls: ${control_contribution:,.0f}'
        ]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig('revenue_decomp_full.png', dpi=output_config['plot_dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"✅ Full decomposition pie chart saved")
    
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
    # 13. FINAL SUMMARY
    # ============================================================================
    
    print("\n" + "="*80)
    print("MMM MODELING COMPLETE!")
    print("="*80)
    
    print("\n📊 VALIDATION RESULTS:")
    print(f"   1. R² Score: {r2:.4f}")
    print(f"   2. Trendline plots: Generated ✅")
    print(f"   3. ROAS Analysis ({period_name}): Complete ✅")
    print(f"   4. Revenue Decomposition (Media vs Non-Media): Complete ✅")
    print(f"   5. Media Channel Breakdown: Complete ✅")
    
    print("\n📁 OUTPUT FILES:")
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
    print(f"   - {output_config['output_files']['decomp_media_vs_nonmedia']} (Pie Chart 1: Media vs Non-Media)")
    print(f"   - {output_config['output_files']['decomp_full']} (Pie Chart 2: Full Decomposition)")
    print(f"   - {output_config['output_files']['contribution_pie']} (Pie Chart 3: Media Channels)")
    print(f"   - {output_config['output_files']['contribution_bar']} (Bar Chart: Media Channels)")
    print(f"   - {output_config['output_files']['contribution_distribution']}")
    print(f"\n  Media Contributions:")
    print(f"   - {output_config['output_files']['media_contributions']}")
    
    print("\n" + "="*80)
