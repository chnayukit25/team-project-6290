#!/usr/bin/env python3
"""
Create presentation slides for accuracy and latency comparison.
Generates two PNG files: accuracy_slide.png and latency_slide.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path(__file__).parent
SUMMARY_CSV = RESULTS_DIR / 'summary_statistics.csv'

def load_summary():
    """Load summary statistics."""
    df = pd.read_csv(SUMMARY_CSV)
    return df

def create_accuracy_slide(summary_df):
    """Create accuracy comparison slide."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Separate HE and plain
    he_df = summary_df[summary_df['method'] == 'HE']
    plain_df = summary_df[summary_df['method'] == 'plain']
    
    # Dataset labels (a, b, c)
    datasets = ['a', 'b', 'c']
    
    # HE and plain accuracy values
    he_acc = he_df['accuracy(%)'].values
    plain_acc = plain_df['accuracy(%)'].values
    
    # Bar positions
    x = np.arange(len(datasets))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, he_acc, width, label='Homomorphic Encryption (HE)', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, plain_acc, width, label='Plaintext', color='#ff7f0e', alpha=0.8)
    
    # Add value labels on bars
    for bars, values in [(bars1, he_acc), (bars2, plain_acc)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=11)
    
    # Set labels and title
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Accuracy Comparison: HE vs Plaintext\n(First 100 Samples, Identical Performance)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=13)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add average accuracy text box
    he_avg = he_acc.mean()
    plain_avg = plain_acc.mean()
    textstr = f'Average Accuracy:\nHE: {he_avg:.1f}%\nPlaintext: {plain_avg:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            # verticalalignment='top', bbox=props)
    
    # Add note about sample size
    ax.text(0.5, -0.15, 'Note: HE datasets contain 100 samples each; plaintext uses first 100 samples of 879 total.',
            transform=ax.transAxes, fontsize=11, ha='center', va='top', style='italic')
    
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'accuracy_slide.svg', dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print("Accuracy slide saved to accuracy_slide.png")

def create_latency_slide(summary_df):
    """Create latency comparison slide with log scale."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Separate HE and plain
    he_df = summary_df[summary_df['method'] == 'HE']
    plain_df = summary_df[summary_df['method'] == 'plain']
    
    # Dataset labels
    datasets = ['a', 'b', 'c']
    
    # Calculate total HE time (encrypt + infer)
    he_total = he_df['encrypt_mean'] + he_df['infer_mean']
    plain_infer = plain_df['infer_mean']
    
    # Convert to microseconds for better readability on log scale
    he_total_us = he_total * 1000  # ms to μs
    plain_infer_us = plain_infer * 1000  # ms to μs
    
    # Plot 1: Bar chart with log scale y-axis
    x = np.arange(len(datasets))
    width = 0.35
    
    # Use log scale
    ax1.set_yscale('log')
    
    # Plot bars
    bars1 = ax1.bar(x - width/2, he_total_us, width, label='HE Total (encrypt + infer)', color='#d62728', alpha=0.8)
    bars2 = ax1.bar(x + width/2, plain_infer_us, width, label='Plaintext Inference', color='#2ca02c', alpha=0.8)
    
    # Add value labels (in ms for readability)
    for bar, value_ms in zip(bars1, he_total.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                f'{value_ms:.1f} ms', ha='center', va='bottom', fontsize=10, rotation=0)
    
    for bar, value_ms in zip(bars2, plain_infer.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                f'{value_ms:.4f} ms', ha='center', va='bottom', fontsize=10, rotation=0)
    
    ax1.set_xlabel('Dataset', fontsize=14)
    ax1.set_ylabel('Time (μs, log scale)', fontsize=14)
    ax1.set_title('Latency Comparison: HE vs Plaintext\n(Log Scale)', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=13)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, which='both')
    
    # Plot 2: Speedup factor (HE slower by factor)
    speedup_factors = he_total.values / plain_infer.values
    
    bars3 = ax2.bar(x, speedup_factors, width=0.6, color='#9467bd', alpha=0.8)
    ax2.set_xlabel('Dataset', fontsize=14)
    ax2.set_ylabel('Slowdown Factor (HE/Plain)', fontsize=14)
    ax2.set_title('HE Slowdown Factor Relative to Plaintext\n(How many times slower)', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=13)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels on bars
    for bar, factor in zip(bars3, speedup_factors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{factor:,.0f}x', ha='center', va='bottom', fontsize=11)
    
    # Add summary statistics
    avg_he_total = he_total.mean()
    avg_plain = plain_infer.mean()
    avg_speedup = speedup_factors.mean()
    he_encrypt_avg = he_df['encrypt_mean'].mean()
    he_infer_avg = he_df['infer_mean'].mean()
    
    summary_text = (
        f'Average Times:\n'
        f'• HE Total: {avg_he_total:.2f} ms\n'
        f'  (Encrypt: {he_encrypt_avg:.2f} ms + Infer: {he_infer_avg:.2f} ms)\n'
        f'• Plaintext: {avg_plain:.4f} ms\n'
        f'• Average Slowdown: {avg_speedup:,.0f}x'
    )
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    # ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, fontsize=12,
            # verticalalignment='top', bbox=props)
    
    # Add note
    fig.text(0.5, 0.01, 'Note: HE inference includes homomorphic encryption overhead. Plaintext inference is on unencrypted data.',
             ha='center', fontsize=11, style='italic')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(RESULTS_DIR / 'latency_slide.svg', dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print("Latency slide saved to latency_slide.png")

def main():
    print("Loading summary statistics...")
    summary_df = load_summary()
    
    print("Creating accuracy slide...")
    create_accuracy_slide(summary_df)
    
    print("Creating latency slide...")
    create_latency_slide(summary_df)
    
    print("\nSlides created successfully:")
    print("• accuracy_slide.png - Accuracy comparison")
    print("• latency_slide.png - Latency comparison with log scale")

if __name__ == '__main__':
    main()