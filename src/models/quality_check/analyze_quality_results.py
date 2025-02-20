import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare the CSV data for analysis"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    print("Original columns:", df.columns.tolist())  # For debugging
    print("Number of rows:", len(df))  # For debugging
    
    # Fill empty model_version with 'base'
    df['model_version'] = df['model_version'].fillna('base')
    df['checkpoint'] = df['checkpoint'].fillna('base')
    
    # Melt the dataframe to get all metrics in one column
    metric_columns = [col for col in df.columns if any(metric in col for metric in ['tone', 'approach', 'format', 'logic'])]
    print("Metric columns found:", metric_columns)  # For debugging
    
    if not metric_columns:
        raise ValueError("No metric columns found in the CSV file")
    
    # Print sample of data before melting
    print("\nSample of original data:")
    print(df[['model_version', 'checkpoint'] + metric_columns].head())
    
    df_melted = pd.melt(
        df,
        id_vars=['model_version', 'checkpoint'],
        value_vars=metric_columns,
        var_name='metric_pair',
        value_name='score'
    )
    
    # Print sample of melted data
    print("\nSample of melted data:")
    print(df_melted.head())
    
    # Split metric_pair into metric_type and pair_number
    df_melted[['metric_type', 'pair_num']] = df_melted['metric_pair'].str.extract(r'(\w+)_pair(\d+)')
    
    # Convert score to numeric, coercing errors to NaN
    df_melted['score'] = pd.to_numeric(df_melted['score'], errors='coerce')
    
    # Remove any rows with NaN values
    df_melted = df_melted.dropna()
    
    print("\nAfter processing:")
    print("Number of valid data points:", len(df_melted))  # Debug use
    print("Unique model versions:", df_melted['model_version'].unique())  # Debug use
    print("Unique metrics:", df_melted['metric_type'].unique())  # Debug use
    
    # Print sample of final data
    print("\nSample of final processed data:")
    print(df_melted.head())
    
    if len(df_melted) == 0:
        raise ValueError("No valid data points after processing")
    
    return df_melted

def calculate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for each model version, checkpoint, and metric type"""
    summary = df.groupby(['model_version', 'checkpoint', 'metric_type'])['score'].agg([
        'mean',
        'std',
        'min',
        'max',
        'count'
    ]).round(3)
    
    return summary

def calculate_pair_penalty(pair_diff: float) -> float:
    """
    Calculate penalty/bonus based on difference between pair1 and pair2
    
    Args:
        pair_diff: Score difference between pair1 and pair2
    Returns:
        float: Penalty (negative value) or bonus (positive value)
    """
    if pair_diff == 0:
        return 0
    elif pair_diff > 0:
        # For positive difference (pair1>pair2), apply proportional penalty
        return -0.2 * pair_diff  # Example: difference of 1.0 reduces score by 0.2
    else:
        # For negative difference (pair1<pair2), apply proportional bonus
        return 0.1 * abs(pair_diff)  # Example: difference of -1.0 increases score by 0.1

def plot_metric_comparisons(df: pd.DataFrame, output_dir: str):
    """Create visualization plots for metric comparisons"""
    if df.empty:
        print("Error: No data to plot")
        return
        
    # Set style - using a default matplotlib style
    plt.style.use('default')
    
    # Set the figure size and font sizes
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14
    })
    
    # Plot 1: Box plot comparing metrics across model versions
    plt.figure()
    
    # Prepare data for boxplot
    metrics = sorted(df['metric_type'].unique())
    models = sorted(df['model_version'].unique())
    
    if not metrics or not models:
        print("Error: No metrics or models found in data")
        return
    
    # Update weight definitions (excluding pair_consistency)
    metric_weights = {
        'tone': 0.35 / 0.85,      # 35% → 41.2%
        'logic': 0.20 / 0.85,     # 20% → 23.5%
        'approach': 0.20 / 0.85,  # 20% → 23.5%
        'format': 0.10 / 0.85     # 10% → 11.8%
    }
    
    # Modify best checkpoint identification section
    best_checkpoints = {}
    for model in df[df['model_version'] != 'base']['model_version'].unique():
        model_data = df[df['model_version'] == model]
        checkpoint_scores = {}
        
        for checkpoint in model_data['checkpoint'].unique():
            checkpoint_data = model_data[model_data['checkpoint'] == checkpoint]
            
            # Calculate basic metric scores
            metric_scores = checkpoint_data.groupby('metric_type')['score'].mean()
            base_weighted_score = sum(
                metric_scores[metric] * weight 
                for metric, weight in metric_weights.items()
            )
            
            # Calculate and evaluate pair differences
            pair_penalties = []
            for metric in ['tone', 'logic', 'approach', 'format']:
                pair1_scores = checkpoint_data[
                    checkpoint_data['metric_pair'] == f'{metric}_pair1'
                ]['score']
                pair2_scores = checkpoint_data[
                    checkpoint_data['metric_pair'] == f'{metric}_pair2'
                ]['score']
                
                if not pair1_scores.empty and not pair2_scores.empty:
                    pair_diff = pair1_scores.mean() - pair2_scores.mean()
                    pair_penalties.append(calculate_pair_penalty(pair_diff))
            
            # Apply pair difference evaluation to total score (15% weight)
            pair_consistency_score = sum(pair_penalties) / len(pair_penalties) if pair_penalties else 0
            pair_weighted_score = pair_consistency_score * 0.15  # 15% weight
            
            # Calculate final score (85% + 15%)
            final_score = base_weighted_score * 0.85 + pair_weighted_score
            checkpoint_scores[checkpoint] = final_score
        
        # Select checkpoint with highest score
        best_checkpoints[model] = max(checkpoint_scores.items(), key=lambda x: x[1])[0]
    
    # Preparing box plot data
    positions = []
    data = []
    labels = []
    colors = ['lightblue', 'lightgreen', 'lightpink']
    
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            if model == 'base':
                # Use all data for base model
                mask = (df['metric_type'] == metric) & (df['model_version'] == model)
            else:
                # Use only optimal checkpoint data for fine-tuned models
                mask = ((df['metric_type'] == metric) & 
                       (df['model_version'] == model) & 
                       (df['checkpoint'] == best_checkpoints[model]))
            
            scores = df[mask]['score'].dropna().tolist()
            if scores:
                data.append(scores)
                positions.append(i * (len(models) + 1) + j)
                labels.append(f"{model}")
    
    if not data:
        print("Error: No valid data for plotting")
        return
    
    # Create boxplot
    bp = plt.boxplot(data, positions=positions, patch_artist=True)
    
    # Color the boxes
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=colors[i % len(colors)])
    
    # Set the style of the chart
    plt.xticks([i * (len(models) + 1) + (len(models) - 1) / 2 for i in range(len(metrics))],
               metrics, rotation=45)
    plt.title('Metric Scores Distribution by Model Version\n' +
             '(Fine-tuned models shown at their best checkpoints)', 
             pad=20)
    plt.ylabel('Score')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], label=model)
                      for i, model in enumerate(models)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add explanation text
    plt.figtext(0.02, 0.02, 
                "Note: For fine-tuned models, only the best checkpoint data is shown.\n"
                "Best checkpoints selected based on weighted average score across metrics.\n"
                "Weights: Tone(41.2%), Logic(23.5%), Approach(23.5%), Format(11.8%)",
                fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metric_distribution.png', bbox_inches='tight')
    plt.close()
    
    # Plot 2: Line plot showing progression across checkpoints
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))  # Create 2x2 subplots
    axes = axes.flatten()  # Convert to 1D array for easier handling
    
    # Handle checkpoint numbers for non-base models
    df['checkpoint_num'] = df.apply(lambda x: 
        int(x['checkpoint'].replace('checkpoint-', '')) if 'checkpoint-' in str(x['checkpoint'])
        else 0 if x['checkpoint'] == 'base'  # Treat base model as checkpoint 0
        else None, axis=1)
    
    # Define colors for each model
    colors = {
        'attention-tuned': '#FF6347',    # Vermillion
        'standard-tuned': '#DAA520',  # Goldenrod
        'base': '#4169E1'               # Royal Blue
    }
    
    metrics = ['approach', 'format', 'logic', 'tone']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot for each model
        for model in ['attention-tuned', 'standard-tuned', 'base']:
            model_data = df[df['model_version'] == model]
            if not model_data.empty:
                means = model_data.groupby(['checkpoint_num', 'metric_type'])['score'].mean().unstack()
                if metric in means.columns:
                    if model == 'base':
                        # Display base model as horizontal line
                        base_score = means[metric].iloc[0]
                        ax.axhline(y=base_score, color=colors[model], 
                                 linestyle='-', label=f'{model}', linewidth=2)
                    else:
                        # Display other models with lines and points
                        ax.plot(means.index, means[metric], 
                               marker='o', label=f'{model}',
                               color=colors[model], linewidth=2)
        
        ax.set_title(f'{metric.capitalize()} Score Progression', pad=10)
        ax.set_xlabel('Checkpoint Number (0 = base model)')
        ax.set_ylabel('Average Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Set Y-axis range to 0-4
        ax.set_ylim(1.5, 4.0)
    
    plt.suptitle('Score Progression Across Checkpoints', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/checkpoint_progression.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_improvement_from_base(df: pd.DataFrame, output_dir: str):
    """Plot improvements of fine-tuned models compared to base model"""
    # Set larger figure size with top margin
    plt.figure(figsize=(12, 8))
    
    # Adjust subplot position to add top margin
    plt.subplots_adjust(top=0.9, bottom=0.2)
    
    # Calculate base model means for each metric
    base_means = df[df['model_version'] == 'base'].groupby('metric_type')['score'].mean()
    
    # Update weight definitions (excluding pair_consistency)
    metric_weights = {
        'tone': 0.35 / 0.85,      # 35% → 41.2%
        'logic': 0.20 / 0.85,     # 20% → 23.5%
        'approach': 0.20 / 0.85,  # 20% → 23.5%
        'format': 0.10 / 0.85     # 10% → 11.8%
    }
    
    # Get the best checkpoint scores for each fine-tuned model
    ft_models = df[df['model_version'] != 'base']['model_version'].unique()
    improvements = []
    
    for model in ft_models:
        model_data = df[df['model_version'] == model]
        
        # Calculate weighted average score for each checkpoint
        checkpoint_scores = {}
        for checkpoint in model_data['checkpoint'].unique():
            checkpoint_data = model_data[model_data['checkpoint'] == checkpoint]
            metric_scores = checkpoint_data.groupby('metric_type')['score'].mean()
            
            # Calculate basic metric scores
            base_weighted_score = sum(
                metric_scores[metric] * weight 
                for metric, weight in metric_weights.items()
            )
            
            # Calculate and evaluate pair differences
            pair_penalties = []
            for metric in ['tone', 'logic', 'approach', 'format']:
                pair1_scores = checkpoint_data[
                    checkpoint_data['metric_pair'] == f'{metric}_pair1'
                ]['score']
                pair2_scores = checkpoint_data[
                    checkpoint_data['metric_pair'] == f'{metric}_pair2'
                ]['score']
                
                if not pair1_scores.empty and not pair2_scores.empty:
                    pair_diff = pair1_scores.mean() - pair2_scores.mean()
                    pair_penalties.append(calculate_pair_penalty(pair_diff))
            
            # Apply pair difference evaluation to total score (15% weight)
            pair_consistency_score = sum(pair_penalties) / len(pair_penalties) if pair_penalties else 0
            pair_weighted_score = pair_consistency_score * 0.15  # 15% weight
            
            # Calculate final score (85% + 15%)
            final_score = base_weighted_score * 0.85 + pair_weighted_score
            checkpoint_scores[checkpoint] = final_score
        
        # Identify checkpoint with highest weighted score
        best_checkpoint = max(checkpoint_scores.items(), key=lambda x: x[1])[0]
        
        # Get scores for selected checkpoint
        best_checkpoint_data = model_data[model_data['checkpoint'] == best_checkpoint]
        best_scores = best_checkpoint_data.groupby('metric_type')['score'].mean()
        
        improvement = best_scores - base_means
        improvements.append((model, improvement, best_checkpoint))
    
    # Plot
    x = np.arange(len(base_means.index))
    width = 0.35
    
    for i, (model, improvement, best_checkpoint) in enumerate(improvements):
        plt.bar(x + i*width, improvement, width, label=f"{model}\n(Best: {best_checkpoint})",
               color=['lightblue', 'lightgreen'][i])
        
        # Add value labels on bars
        for j, v in enumerate(improvement):
            plt.text(x[j] + i*width, v + (0.1 if v >= 0 else -0.1),
                    f'{v:+.2f}',
                    ha='center', va='bottom' if v >= 0 else 'top')
    
    # Place graph elements
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Metrics', labelpad=10)  # Adjust axis label position with labelpad
    plt.ylabel('Improvement from Base Model\n(Score Difference)', labelpad=10)  # Add line break for two lines
    plt.title('Best Improvement in Socratic Elements from Base Model', pad=20)
    plt.xticks(x + width/2, base_means.index)
    plt.legend(bbox_to_anchor=(1.02, 1))  # Adjust legend position
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add baseline scores as text (adjust position)
    plt.text(-0.2, -1.2, f'Base Model Scores:', 
            fontsize=10, color='gray', ha='left')
    for i, (metric, score) in enumerate(base_means.items()):
        plt.text(i-0.2, -1.4, f'{metric}: {score:.2f}', 
                fontsize=9, color='gray', ha='left')
    
    # Add explanation text (add weight explanation)
    plt.figtext(0.02, 0.02, 
                "Note: Improvements shown are from the best performing checkpoint\n"
                "for each model, selected based on weighted average score across metrics.\n"
                "Weights: Tone(41.2%), Logic(23.5%), Approach(23.5%), Format(11.8%)",
                fontsize=8, style='italic')
    
    # Set Y-axis range explicitly
    plt.ylim(-1.5, 1.5)  # Adjust lower limit to show baseline score text
    
    plt.tight_layout()  # Adjust layout automatically
    plt.savefig(f'{output_dir}/improvement_from_base.png', 
                bbox_inches='tight',  # Adjust margins appropriately
                dpi=300)
    plt.close()

def plot_pair_differences(df: pd.DataFrame, output_dir: str):
    """Plot the differences between pair1 and pair2 for each metric across models and checkpoints"""
    # Create 2x2 subplots (for 4 metrics)
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()  # Convert to 1D array for easier handling
    
    metrics = ['tone', 'approach', 'format', 'logic']
    colors = {
        'attention-tuned': '#FF6347',    # Vermillion
        'standard-tuned': '#DAA520',  # Goldenrod
        'base': '#4169E1'                # Royal Blue
    }
    
    # Handle checkpoint numbers for non-base models
    df['checkpoint_num'] = df.apply(lambda x: 
        int(x['checkpoint'].replace('checkpoint-', '')) if 'checkpoint-' in str(x['checkpoint'])
        else 0 if x['checkpoint'] == 'base'
        else None, axis=1)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # For each model and checkpoint
        for model in ['attention-tuned', 'standard-tuned', 'base']:
            model_diffs = []
            checkpoint_nums = []
            
            # Get unique checkpoints for this model
            model_checkpoints = sorted(df[df['model_version'] == model]['checkpoint'].unique())
            
            for checkpoint in model_checkpoints:
                mask = (df['model_version'] == model) & (df['checkpoint'] == checkpoint)
                
                # Get scores for pair1 and pair2
                pair1_scores = df[mask & (df['metric_pair'] == f'{metric}_pair1')]['score']
                pair2_scores = df[mask & (df['metric_pair'] == f'{metric}_pair2')]['score']
                
                if not pair1_scores.empty and not pair2_scores.empty:
                    # Calculate difference
                    diff = pair1_scores.mean() - pair2_scores.mean()
                    model_diffs.append(diff)
                    
                    # Get checkpoint number for x-axis
                    if checkpoint == 'base':
                        checkpoint_nums.append(0)
                    else:
                        checkpoint_nums.append(int(checkpoint.replace('checkpoint-', '')))
            
            if model_diffs:
                if model == 'base':
                    # Display base model as horizontal line
                    ax.axhline(y=model_diffs[0], color=colors[model], 
                             linestyle='-', label=f'{model}', linewidth=2)
                else:
                    # Plot line with markers
                    ax.plot(checkpoint_nums, model_diffs, 
                           marker='o', label=model,
                           color=colors[model], linewidth=2)
        
        # Customize subplot
        ax.set_title(f'{metric.capitalize()} Pair Difference\n(Pair1 - Pair2)', pad=10)
        ax.set_xlabel('Checkpoint Number (0 = base model)')
        ax.set_ylabel('Score Difference')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Set y-axis limits to show small differences clearly
        ax.set_ylim(-1.0, 1.0)
        
        # Add horizontal line at y=0 to show neutral difference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('Differences Between Pair1 and Pair2 Across Checkpoints', y=1.02, fontsize=16)
    
    # Add explanation text
    plt.figtext(0.02, 0.02, 
                "Note: Positive values indicate Pair1 scored higher than Pair2\n"
                "Base model shown as horizontal line for reference",
                fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pair_differences.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def analyze_quality_results(csv_path: str, output_dir: str):
    """Main function to analyze quality check results"""
    # Load and prepare data
    df = load_and_prepare_data(csv_path)
    
    # Calculate summary statistics
    summary_stats = calculate_summary_stats(df)
    
    # Save summary statistics to CSV
    summary_stats.to_csv(f'{output_dir}/summary_statistics.csv')
    
    # Create visualization plots
    plot_metric_comparisons(df, output_dir)
    plot_improvement_from_base(df, output_dir)
    plot_pair_differences(df, output_dir)
    
    # Print overall findings
    print("\nOverall Analysis Results:")
    print("-" * 50)
    
    # Compare model versions
    for metric in df['metric_type'].unique():
        print(f"\n{metric.upper()} Metric Summary:")
        for model in df['model_version'].unique():
            model_metric_mean = df[
                (df['model_version'] == model) & 
                (df['metric_type'] == metric)
            ]['score'].mean()
            print(f"{model}: {model_metric_mean:.3f}")

    # Print detailed summary
    print("\nDetailed Summary Statistics:")
    print("=" * 80)
    print(summary_stats)
    
    print("\nKey Findings:")
    print("-" * 80)
    print("1. Best performing metrics by model:")
    for model in df['model_version'].unique():
        model_means = df[df['model_version'] == model].groupby('metric_type')['score'].mean()
        best_metric = model_means.idxmax()
        print(f"   {model}: {best_metric} ({model_means[best_metric]:.3f})")

    # Add stability analysis
    print("\n2. Stability Analysis (Standard Deviation):")
    for model in df['model_version'].unique():
        print(f"\n{model}:")
        model_data = df[df['model_version'] == model]
        for metric in df['metric_type'].unique():
            metric_std = model_data[model_data['metric_type'] == metric]['score'].std()
            print(f"   {metric}: {metric_std:.3f}")
    
    # Add improvement analysis
    print("\n3. Improvement Analysis (First to Last Checkpoint):")
    for model in df['model_version'].unique():
        print(f"\n{model}:")
        for metric in df['metric_type'].unique():
            first_checkpoint = df[df['checkpoint'] == 'checkpoint-100']
            last_checkpoint = df[df['checkpoint'] == 'checkpoint-990']
            
            first_score = first_checkpoint[
                (first_checkpoint['model_version'] == model) & 
                (first_checkpoint['metric_type'] == metric)
            ]['score'].mean()
            
            last_score = last_checkpoint[
                (last_checkpoint['model_version'] == model) & 
                (last_checkpoint['metric_type'] == metric)
            ]['score'].mean()
            
            improvement = last_score - first_score
            print(f"   {metric}: {improvement:+.3f}")

    # Add improvement analysis from base model
    print("\n4. Improvement Analysis from Base Model:")
    base_scores = df[df['model_version'] == 'base'].groupby('metric_type')['score'].mean()
    
    for model in df[df['model_version'] != 'base']['model_version'].unique():
        print(f"\n{model}:")
        model_scores = df[df['model_version'] == model].groupby('metric_type')['score'].mean()
        for metric in base_scores.index:
            improvement = model_scores[metric] - base_scores[metric]
            print(f"   {metric}: {improvement:+.3f} ({base_scores[metric]:.2f} → {model_scores[metric]:.2f})")

def main():
    csv_path = "data/config/automation_gemma.csv"
    output_dir = "data/analysis"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyze_quality_results(csv_path, output_dir)

if __name__ == "__main__":
    main() 