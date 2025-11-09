# Media Mix Modeling (MMM) - End-to-End Project

## 📊 Project Overview

This repository contains a comprehensive **Media Mix Modeling (MMM)** implementation for analyzing the effectiveness of marketing channels and optimizing media spend allocation. The project focuses on the **CDC brand** in Thailand, analyzing media performance across multiple channels from 2017-2020.

Media Mix Modeling is a statistical analysis technique that helps quantify the impact of various marketing channels on sales/revenue, enabling data-driven budget allocation decisions.

## 🎯 Business Objective

- **Measure the effectiveness** of different media channels (TV, Digital, OOH, etc.)
- **Calculate ROI and ROAS** for each marketing channel
- **Optimize media spend** allocation across channels
- **Understand adstock effects** (carryover impact of advertising)
- **Model saturation curves** to identify diminishing returns
- **Provide actionable insights** for marketing budget planning

## 📁 Project Structure

```
MMM_end_to_end/
│
├── Data Files
│   ├── media_data.csv              # Raw media spend data (32,405 records)
│   ├── control_data.csv            # Control variables & revenue data (240 records)
│   ├── input_data1.xlsx            # Processed input for modeling
│   ├── media_mrd.xlsx              # Media data pivoted by week
│   ├── col_info.csv                # Column metadata
│   └── control_var_mapping.json    # Control variable categorization
│
├── Notebooks
│   ├── 0.analysis.ipynb            # Initial data exploration
│   ├── 1.eda.ipynb                 # Exploratory data analysis
│   ├── 3.preprocessing.ipynb       # Data preprocessing & transformation
│   ├── modelling.ipynb             # Main modeling notebook
│   ├── modelling_new.ipynb         # Updated modeling approach
│   └── Copy of modelling.ipynb     # Backup modeling notebook
│
├── Scripts
│   └── modelling_new.py            # Production modeling script
│
├── Visualizations
│   ├── actual_vs_predicted.png     # Model performance
│   ├── media_contributions.png     # Channel contribution analysis
│   ├── media_coefficients_forest.png # Coefficient forest plot
│   ├── trace_plots_base.png        # MCMC diagnostics (base variables)
│   ├── trace_plots_media.png       # MCMC diagnostics (media variables)
│   ├── cdc_roas_by_channel.png     # ROAS by channel
│   ├── cdc_roi_by_channel.png      # ROI by channel
│   ├── cdc_roi_distributions.png   # ROI uncertainty distributions
│   ├── cdc_spend_vs_revenue.png    # Spend vs revenue scatter
│   └── cdc_profit_by_channel.png   # Profit/loss by channel
│
├── Documentation
│   ├── Media Mix Modeling.docx     # Project documentation
│   └── Top 50 Media Mix Modeling Interview Que.docx
│
└── requirements.txt                # Python dependencies
```

## 🔧 Technical Stack

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning utilities
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualizations
- **pymc** - Bayesian modeling (used in notebooks)
- **arviz** - Bayesian inference diagnostics
- **cvxpy** - Convex optimization
- **openpyxl** - Excel file handling

## 📈 Data Description

### Media Data (`media_data.csv`)
- **32,405 records** spanning 2017-2020
- **Time granularity**: Weekly
- **Brands**: CDC, SALT, EQUITY, TOTAL, PANJAVED
- **Media channels**: 19 unique brand+vehicle combinations
  - TV (CDC, EQUITY, SALT, TOTAL, PANJAVED)
  - Digital (CDC, EQUITY, SALT, TOTAL, PANJAVED)
  - In-Store (EQUITY, SALT, TOTAL, PANJAVED)
  - OOH (EQUITY, PANJAVED)
  - Other Advertising/Newspaper/Radio (EQUITY, SALT, TOTAL)

### Control Variables (`control_data.csv`)
- **240 monthly records** (47 months from Feb 2017 to Dec 2020)
- **54 variables** categorized into:

#### 1. Distribution (2 variables)
- Numeric Distribution (ND)
- Weighted Distribution (WD)

#### 2. Brand Equity (8 variables)
- Aided Awareness
- Closeness Top 3 Box
- Loyalty
- Top of Mind Brand Awareness
- Total Unaided Brand Awareness
- Chim Shop Chai
- Panic Pantry Loading

#### 3. Promotions & Discounts (19 variables)
- Welfare, Welfare Additional Boost
- BigPack, BuyGet, Extravol
- Free Promo, Free with Others, Freepremium
- Near Pack, PMD, RL, Theme, TPR
- Target Price, Target Base Price, Target Discount
- Darlie Discount, Darlie CDC Base Price Ratio
- Extravol Darlie, Free Promo Darlie, etc.

#### 4. Seasonality (3 variables)
- Week 5 Seasonality Index
- Week 13 Seasonality Index
- Trendline

#### 5. Macro-Economic (2 variables)
- Consumer Confidence Index
- Tourist Arrivals in Million

#### 6. Target Variables
- **VALUE**: Revenue (in local currency)
- **VOLUME**: Sales volume

## 🧮 Modeling Approach

### 1. Data Preprocessing
- **Time alignment**: Convert monthly control data to weekly granularity
- **Revenue distribution**: Divide monthly revenue equally across weeks
- **Feature engineering**: Create brand+vehicle combinations
- **Data filtering**: Focus on CDC brand for modeling

### 2. Adstock Transformation
Captures the **carryover effect** of advertising over time using geometric adstock:

```python
adstock_t = spend_t + decay_rate × adstock_(t-1)
```

- **Decay rate**: 0.5 (configurable per channel)
- Models how advertising impact persists beyond the initial exposure

### 3. Saturation Transformation
Models **diminishing returns** using the Hill function:

```python
saturated_spend = (alpha × spend^gamma) / (alpha^gamma + spend^gamma)
```

- **Alpha**: Half-saturation point (100)
- **Gamma**: Shape parameter (2)
- Captures the S-curve relationship between spend and impact

### 4. Bayesian Regression Model
Uses **PyMC** for Bayesian linear regression:

```
Revenue = Intercept + 
          Σ(β_media × Media_Channels) + 
          Σ(β_control × Control_Variables) + 
          ε
```

**Benefits of Bayesian approach**:
- Uncertainty quantification for all coefficients
- Probabilistic ROI/ROAS estimates
- Robust to multicollinearity
- Natural handling of prior knowledge

### 5. Model Diagnostics
- **Trace plots**: MCMC convergence analysis
- **R-hat statistics**: Chain convergence metrics
- **Posterior predictive checks**: Model fit validation
- **Actual vs Predicted**: Visual performance assessment

## 📊 Key Metrics & Outputs

### 1. Channel Contribution
- **Absolute contribution**: Revenue attributed to each channel
- **Percentage contribution**: Relative importance of channels
- **Incremental lift**: Revenue above baseline

### 2. ROI Analysis
- **ROAS** (Return on Ad Spend): Revenue / Spend
- **ROI %**: ((Revenue - Spend) / Spend) × 100
- **Profit**: Revenue - Spend
- **Efficiency Score**: Revenue per dollar spent

### 3. Weekly Performance
- **Weekly ROAS trends**: Time-series analysis
- **Profitability rate**: % of weeks with ROAS > 1
- **Volatility metrics**: Standard deviation of weekly ROAS

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MMM_end_to_end

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

#### Option 1: Using Jupyter Notebooks
```bash
jupyter notebook
# Open and run notebooks in sequence:
# 1. 0.analysis.ipynb
# 2. 1.eda.ipynb
# 3. 3.preprocessing.ipynb
# 4. modelling_new.ipynb
```

#### Option 2: Using Python Script
```bash
python modelling_new.py
```

## 📉 Key Findings

### Media Channels Analyzed (CDC Brand)
1. **CDC+TV** - Television advertising
2. **CDC+DIGITAL** - Digital marketing channels

### Control Variables Impact
The model accounts for:
- **Distribution effects**: Product availability impact
- **Brand health**: Awareness and loyalty metrics
- **Promotional activities**: Various discount schemes
- **Seasonality**: Weekly and quarterly patterns
- **External factors**: Economic indicators

## 🎨 Visualizations Generated

1. **Model Performance**
   - Actual vs Predicted revenue plot
   - R² score and error metrics

2. **Channel Analysis**
   - Media contribution bar charts
   - Coefficient forest plots with credible intervals

3. **ROI Dashboards**
   - ROAS by channel (bar chart)
   - ROI percentage by channel
   - ROI distribution (violin plots)
   - Spend vs Revenue scatter plot
   - Profit/Loss by channel

4. **Diagnostics**
   - MCMC trace plots for convergence
   - Posterior distribution plots

## 💡 Business Insights

### Optimization Recommendations
Based on the model outputs, businesses can:

1. **Reallocate budget** from low-ROI to high-ROI channels
2. **Identify optimal spend levels** before saturation
3. **Plan seasonal campaigns** using seasonality insights
4. **Measure incrementality** of each marketing channel
5. **Forecast revenue** under different budget scenarios

### Decision Support
- **Which channels to scale up/down?**
- **What is the optimal media mix?**
- **How much to spend on each channel?**
- **What is the expected return on investment?**

## 📝 Model Outputs

The modeling pipeline generates:

- `mmm_trace.nc` - Posterior samples from Bayesian model
- `mmm_summary.csv` - Coefficient summaries with credible intervals
- `media_contributions.csv` - Channel-wise revenue contributions
- `roi_analysis.csv` - ROI & ROAS metrics by channel
- `weekly_roi_stats.csv` - Weekly performance statistics
- `roi_distributions.csv` - Posterior ROI distributions
- Multiple PNG visualizations

## 🔍 Advanced Features

### 1. Uncertainty Quantification
- Credible intervals for all metrics
- Probabilistic ROI estimates
- Risk assessment for budget decisions

### 2. Time-Varying Effects
- Weekly ROAS trends
- Seasonal pattern detection
- Long-term vs short-term impact

### 3. Multi-Brand Analysis
Data includes multiple brands (CDC, SALT, EQUITY, TOTAL, PANJAVED) for comparative analysis

## 🛠️ Customization

### Adjusting Model Parameters

**Adstock decay rate** (in `modelling_new.py`):
```python
decay_rate = 0.5  # Adjust between 0-1
```

**Saturation parameters**:
```python
alpha = 100  # Half-saturation point
gamma = 2    # Shape parameter
```

**Bayesian priors**: Modify in the PyMC model definition

## 📚 References & Resources

### Key Concepts
- **Adstock Effect**: Carryover impact of advertising
- **Saturation Curve**: Diminishing returns on ad spend
- **Bayesian Inference**: Probabilistic modeling approach
- **MCMC**: Markov Chain Monte Carlo sampling

### Industry Applications
- Marketing budget optimization
- Channel performance measurement
- Media planning and forecasting
- Attribution modeling

## 🤝 Contributing

This is an end-to-end MMM implementation suitable for:
- Marketing analysts
- Data scientists
- Media planners
- Business strategists

## 📧 Contact & Support

For questions about the methodology or implementation, refer to:
- `Media Mix Modeling.docx` - Detailed documentation
- `Top 50 Media Mix Modeling Interview Que.docx` - Common questions

## ⚠️ Important Notes

1. **Data Privacy**: Ensure compliance with data protection regulations
2. **Model Validation**: Always validate results with business stakeholders
3. **Assumptions**: Review model assumptions for your specific use case
4. **Causality**: MMM shows correlation; validate causal relationships
5. **Time Period**: Model trained on 2017-2020 data; retrain for current periods

## 🎯 Future Enhancements

Potential improvements:
- [ ] Automated hyperparameter tuning
- [ ] Cross-validation framework
- [ ] Real-time dashboard integration
- [ ] Multi-touch attribution
- [ ] Competitive spend analysis
- [ ] Geographic segmentation
- [ ] Prophet/LSTM for time-series forecasting
- [ ] Budget optimizer with constraints

---

**Project Status**: ✅ Complete - Production Ready

**Last Updated**: 2024

**License**: [Specify License]

---

## 🏆 Project Highlights

✨ **Comprehensive MMM pipeline** from raw data to actionable insights  
✨ **Bayesian approach** for robust uncertainty quantification  
✨ **19 media channels** analyzed across 5 brands  
✨ **54 control variables** for accurate attribution  
✨ **4 years** of historical data (2017-2020)  
✨ **Production-ready** code with extensive visualizations  

---

*Built with ❤️ for data-driven marketing optimization*
