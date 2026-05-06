"""
End-to-End Hidden Markov Weather Model Pipeline
Analyzes weather regimes for Madrid using HMM clustering and transition analysis
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import project modules
from markovstates.utils import Preprocess, FeatMat, hourly_dataframe
from markovstates.data_collect import response
from markovstates.factor_analysis import FINAL_FEATURES
from markovstates.models import HMMWeatherModel


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def get_user_input() -> tuple[str, str, bool]:
    """
    Get user input for date range and model training preference.
    
    Returns:
        tuple: (start_date, end_date, should_retrain)
    """
    print_header("WEATHER REGIME ANALYSIS - Hidden Markov Model Pipeline")
    print("\nThis tool analyzes historical weather data to identify distinct")
    print("weather regimes and their transition probabilities.")
    
    print("\n📍 Location: Madrid, Spain (40.41°N, 3.7°E)")
    print("📊 Data Source: Open-Meteo Historical Archive API")
    print("📈 Model: Gaussian Hidden Markov Model (5 states)")
    
    # Date inputs
    default_start = "2023-04-10"
    default_end = datetime.now().strftime("%Y-%m-%d")
    
    while True:
        start_input = input(f"\nEnter start date (YYYY-MM-DD) [{default_start}]: ").strip()
        start_date = start_input if start_input else default_start
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            break
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD")
    
    while True:
        end_input = input(f"Enter end date (YYYY-MM-DD) [{default_end}]: ").strip()
        end_date = end_input if end_input else default_end
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
            break
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD")
    
    retrain_input = input("\nRetrain model? (y/n) [n]: ").strip().lower()
    should_retrain = retrain_input == 'y'
    
    return start_date, end_date, should_retrain


def collect_weather_data() -> pd.DataFrame:
    """
    Collect and display weather data.
    Uses pre-fetched hourly_dataframe from data_collect.py
    
    Returns:
        pd.DataFrame: Hourly weather data
    """
    print_header("DATA COLLECTION")
    print("✓ Weather data retrieved from Open-Meteo API")
    print(f"✓ Total records: {len(hourly_dataframe):,}")
    print(f"✓ Date range: {hourly_dataframe['date'].min()} to {hourly_dataframe['date'].max()}")
    print(f"✓ Columns: {', '.join(hourly_dataframe.columns.tolist())}")
    print(f"✓ Location: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"✓ Elevation: {response.Elevation()} m asl")
    
    return hourly_dataframe


def preprocess_data(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Preprocess weather data: resample, handle missing values, scale.
    
    Args:
        df: Raw hourly weather dataframe
        
    Returns:
        tuple: (feature_matrix, daily_dataframe)
    """
    print_header("DATA PREPROCESSING")
    
    # Initialize preprocessor
    pp = Preprocess(df)
    
    print("📍 Step 1: Resampling to daily data")
    daily_df = pp.resample()
    print(f"   ✓ Resampled from {len(df):,} hourly records to {len(daily_df):,} daily records")
    
    print("\n📍 Step 2: Handling missing values")
    daily_df_clean = pp.handle_missing(method="interpolate")
    print(f"   ✓ Applied linear interpolation for missing values")
    
    print("\n📍 Step 3: Constructing feature matrix")
    print(f"   ✓ Selected features: {FINAL_FEATURES}")
    
    # Create and construct feature matrix
    fm = FeatMat(df, FINAL_FEATURES)
    X = fm.construct_feat_mat()
    print(f"   ✓ Feature matrix shape: {X.shape}")
    print(f"   ✓ Features scaled using StandardScaler (mean=0, std=1)")
    
    # Get daily data with selected features for output
    daily_df_final = daily_df[FINAL_FEATURES]
    
    return X, daily_df_final


def train_or_load_model(X: np.ndarray, should_retrain: bool) -> HMMWeatherModel:
    """
    Train a new HMM model or load an existing one.
    
    Args:
        X: Feature matrix
        should_retrain: Whether to train a new model
        
    Returns:
        HMMWeatherModel: Fitted model
    """
    print_header("MODEL TRAINING/LOADING")
    
    model_path = "/Users/philipalexopoulos/markovstates/models/hmm_final.pkl"
    
    if should_retrain:
        print("🔧 Training new HMM model...")
        print("   Parameters: n_components=5, covariance_type='diag', n_restarts=50")
        print("   (Testing 50 different random seeds and selecting best model)")
        
        hmm = HMMWeatherModel(n_components=5, covar_type='diag', n_restarts=50)
        hmm.fit(X)
        
        score = hmm.score(X)
        bic = hmm.bic(X)
        print(f"\n✓ Model trained successfully!")
        print(f"   Log-Likelihood Score: {score:.4f}")
        print(f"   BIC: {bic:.4f}")
        
        # Save the model
        hmm.save(model_path)
        print(f"   Model saved to: {model_path}")
        
    else:
        print("📂 Loading pre-trained model...")
        hmm = HMMWeatherModel(n_components=5, covar_type='diag')
        hmm.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
        
        score = hmm.score(X)
        bic = hmm.bic(X)
        print(f"   Log-Likelihood Score: {score:.4f}")
        print(f"   BIC: {bic:.4f}")
    
    return hmm


def predict_regimes(hmm: HMMWeatherModel, X: np.ndarray, 
                   daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict weather regimes for each time step.
    
    Args:
        hmm: Trained HMM model
        X: Feature matrix
        daily_df: Daily dataframe for indexing
        
    Returns:
        pd.DataFrame: Daily data with regime predictions
    """
    print_header("REGIME PREDICTION")
    
    regimes = hmm.predict(X)
    
    # Create regime mapping
    regime_names = {
        0: "Warm/Humid",
        1: "Cool/Moderate",
        2: "Warm/Dry",
        3: "Cold/Dry",
        4: "Hot/Peak Summer"
    }
    
    # Add regime columns to daily dataframe
    daily_df["regime_id"] = regimes
    daily_df["regime_name"] = daily_df["regime_id"].map(regime_names)
    
    print(f"✓ Predicted regimes for {len(daily_df)} days")
    print(f"\nRegime Distribution:")
    regime_counts = daily_df["regime_name"].value_counts().sort_index()
    for regime_name, count in regime_counts.items():
        percentage = (count / len(daily_df)) * 100
        print(f"   {regime_name:20s}: {count:4d} days ({percentage:5.1f}%)")
    
    return daily_df, regime_names


def compute_statistics(hmm: HMMWeatherModel, daily_df: pd.DataFrame, 
                      regime_names: dict) -> dict:
    """
    Compute model statistics and regime characteristics.
    
    Args:
        hmm: Trained HMM model
        daily_df: Daily data with regime predictions
        regime_names: Mapping of regime IDs to names
        
    Returns:
        dict: Statistics dictionary
    """
    print_header("MODEL STATISTICS")
    
    stats = {}
    
    # Transition matrix
    transmat = hmm.transition_mat()
    stats['transition_matrix'] = transmat
    
    print("\n🔄 Transition Probabilities (from → to):")
    print("\n   " + "".join([f"{regime_names[i]:18s}" for i in range(5)]))
    print("   " + "-"*100)
    for i in range(5):
        print(f"{regime_names[i]:15s}: " + "".join([f"{transmat[i, j]:17.3f}" for j in range(5)]))
    
    # Steady-state probabilities (long-run regime distribution)
    eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
    steady_state_idx = np.argmax(np.abs(eigenvalues - 1.0) < 1e-8)
    steady_state = np.real(eigenvectors[:, steady_state_idx])
    steady_state = steady_state / steady_state.sum()
    stats['steady_state'] = steady_state
    
    print("\n📊 Steady-State Probabilities (long-run regime distribution):")
    for i, regime_name in regime_names.items():
        print(f"   {regime_name:20s}: {steady_state[i]:6.3f} ({steady_state[i]*100:5.1f}%)")
    
    # Average regime duration
    durations = []
    current_regime = daily_df.iloc[0]["regime_id"]
    current_duration = 1
    
    for i in range(1, len(daily_df)):
        if daily_df.iloc[i]["regime_id"] == current_regime:
            current_duration += 1
        else:
            durations.append((current_regime, current_duration))
            current_regime = daily_df.iloc[i]["regime_id"]
            current_duration = 1
    durations.append((current_regime, current_duration))
    
    avg_durations = {}
    for regime_id in range(5):
        regime_durations = [d for r, d in durations if r == regime_id]
        if regime_durations:
            avg_durations[regime_id] = np.mean(regime_durations)
        else:
            avg_durations[regime_id] = 0
    
    stats['avg_durations'] = avg_durations
    
    print("\n⏱️  Average Regime Duration (days):")
    for regime_id, avg_dur in avg_durations.items():
        print(f"   {regime_names[regime_id]:20s}: {avg_dur:6.2f} days")
    
    # Regime characteristics (mean feature values per regime)
    regime_chars = {}
    for regime_id in range(5):
        regime_data = daily_df[daily_df["regime_id"] == regime_id]
        regime_chars[regime_id] = {
            feat: regime_data[feat].mean() for feat in FINAL_FEATURES
        }
    stats['regime_characteristics'] = regime_chars
    
    print("\n🌡️  Regime Characteristics (mean feature values):")
    print(f"   {'Regime':20s} {'Temp (°C)':>12s} {'Pressure (hPa)':>15s} {'Dew Point (°C)':>15s}")
    print("   " + "-"*65)
    for regime_id in range(5):
        chars = regime_chars[regime_id]
        print(f"   {regime_names[regime_id]:20s} {chars['temperature_2m']:12.1f} "
              f"{chars['surface_pressure']:15.1f} {chars['dew_point_2m']:15.1f}")
    
    return stats


def visualize_results(hmm: HMMWeatherModel, daily_df: pd.DataFrame, 
                     regime_names: dict, stats: dict) -> None:
    """
    Generate and display visualizations.
    
    Args:
        hmm: Trained HMM model
        daily_df: Daily data with regime predictions
        regime_names: Mapping of regime IDs to names
        stats: Statistics dictionary
    """
    print_header("GENERATING VISUALIZATIONS")
    
    # Set up style
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Transition Matrix Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    transmat = stats['transition_matrix']
    regime_labels = [regime_names[i] for i in range(5)]
    sns.heatmap(
        transmat,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=regime_labels,
        yticklabels=regime_labels,
        ax=ax1,
        cbar_kws={"label": "Probability"}
    )
    ax1.set_title("Regime Transition Probabilities", fontsize=12, fontweight='bold')
    ax1.set_xlabel("To")
    ax1.set_ylabel("From")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Steady-State Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    steady_state = stats['steady_state']
    colors = ['steelblue', 'orange', 'firebrick', 'green', 'crimson']
    bars = ax2.bar(regime_labels, steady_state, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title("Steady-State Regime Distribution", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Probability")
    ax2.set_ylim([0, max(steady_state) * 1.15])
    for i, (bar, val) in enumerate(zip(bars, steady_state)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Time Series with Regime Colors
    ax3 = fig.add_subplot(gs[1, :])
    color_map = {0: 'steelblue', 1: 'orange', 2: 'firebrick', 3: 'green', 4: 'crimson'}
    for regime_id in range(5):
        regime_data = daily_df[daily_df["regime_id"] == regime_id]
        ax3.scatter(
            regime_data.index,
            regime_data["temperature_2m"],
            c=color_map[regime_id],
            s=20,
            label=regime_names[regime_id],
            alpha=0.6
        )
    ax3.set_title("Temperature Time Series with Weather Regimes", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Temperature (°C)")
    ax3.set_xlabel("Date")
    ax3.legend(loc='upper right', ncol=5, fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Average Regime Duration
    ax4 = fig.add_subplot(gs[2, 0])
    durations = [stats['avg_durations'][i] for i in range(5)]
    bars = ax4.bar(regime_labels, durations, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title("Average Regime Duration", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Days")
    for bar, dur in zip(bars, durations):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{dur:.1f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # 5. Temperature Distribution by Regime
    ax5 = fig.add_subplot(gs[2, 1])
    temp_by_regime = [daily_df[daily_df["regime_id"] == i]["temperature_2m"].values 
                      for i in range(5)]
    bp = ax5.boxplot(temp_by_regime, labels=regime_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_title("Temperature Distribution by Regime", fontsize=12, fontweight='bold')
    ax5.set_ylabel("Temperature (°C)")
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Hidden Markov Weather Model Analysis - Madrid, Spain", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    print("✓ Visualizations generated successfully!")
    print("  - Transition probabilities heatmap")
    print("  - Steady-state distribution")
    print("  - Temperature time series with regime colors")
    print("  - Average regime duration")
    print("  - Temperature distribution by regime")
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Visualization window opened. Close to continue.")


def generate_summary_report(daily_df: pd.DataFrame, regime_names: dict, stats: dict) -> None:
    """
    Generate a text summary report of the analysis.
    
    Args:
        daily_df: Daily data with regime predictions
        regime_names: Mapping of regime IDs to names
        stats: Statistics dictionary
    """
    print_header("ANALYSIS SUMMARY")
    
    print("\n✨ KEY FINDINGS:\n")
    
    # Most frequent regime
    most_freq = daily_df["regime_name"].value_counts().index[0]
    most_freq_pct = (daily_df["regime_name"].value_counts().values[0] / len(daily_df)) * 100
    print(f"1. Most Common Regime: {most_freq} ({most_freq_pct:.1f}% of days)")
    
    # Temperature ranges
    print("\n2. Temperature Range by Regime:")
    for regime_id in range(5):
        regime_data = daily_df[daily_df["regime_id"] == regime_id]["temperature_2m"]
        print(f"   {regime_names[regime_id]:20s}: {regime_data.min():6.1f}°C to {regime_data.max():6.1f}°C "
              f"(avg: {regime_data.mean():6.1f}°C)")
    
    # Longest regime period
    print("\n3. Regime Persistence:")
    max_duration = 0
    max_regime = None
    current_regime = daily_df.iloc[0]["regime_id"]
    current_duration = 1
    
    for i in range(1, len(daily_df)):
        if daily_df.iloc[i]["regime_id"] == current_regime:
            current_duration += 1
        else:
            if current_duration > max_duration:
                max_duration = current_duration
                max_regime = current_regime
            current_regime = daily_df.iloc[i]["regime_id"]
            current_duration = 1
    
    print(f"   Longest continuous period: {max_duration} days of {regime_names[max_regime]}")
    print(f"   Average regime duration: {np.mean(list(stats['avg_durations'].values())):.1f} days")
    
    # Transition insights
    print("\n4. Regime Transitions:")
    transmat = stats['transition_matrix']
    for i in range(5):
        most_likely_next = np.argmax(transmat[i, :])
        prob = transmat[i, most_likely_next]
        print(f"   From {regime_names[i]:20s} → most likely: {regime_names[most_likely_next]:20s} "
              f"({prob:.1%})")


def main():
    """Main pipeline execution."""
    try:
        # 1. Get user input
        start_date, end_date, should_retrain = get_user_input()
        
        # 2. Collect weather data
        df = collect_weather_data()
        
        # 3. Preprocess data
        X, daily_df = preprocess_data(df)
        
        # 4. Train or load model
        hmm = train_or_load_model(X, should_retrain)
        
        # 5. Predict regimes
        daily_df, regime_names = predict_regimes(hmm, X, daily_df)
        
        # 6. Compute statistics
        stats = compute_statistics(hmm, daily_df, regime_names)
        
        # 7. Generate visualizations
        visualize_results(hmm, daily_df, regime_names, stats)
        
        # 8. Generate summary report
        generate_summary_report(daily_df, regime_names, stats)
        
        print_header("PIPELINE COMPLETE")
        print("\n✅ Analysis completed successfully!")
        print("\nNext steps:")
        print("  • Review the visualizations above")
        print("  • Check regime statistics and transition probabilities")
        print("  • Explore notebooks/ for additional analysis")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
