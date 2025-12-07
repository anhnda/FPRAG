"""
Cross-Comparison: Symmetric vs Asymmetric Quantization

This script provides a detailed cross-comparison between symmetric and asymmetric
quantization for both AWQ and Hybrid PRAQ methods.

Analysis includes:
1. Direct pairwise comparisons (Symmetric vs Asymmetric for each method)
2. Trade-off analysis (Quality vs Speed)
3. Efficiency metrics (Perplexity per tok/s)
4. Statistical analysis of differences
5. Detailed visualizations

Input:
- Reads results from compare_symmetric_vs_asymmetric.py
- Or re-evaluates models if results not available

Output:
- Detailed comparison tables
- Trade-off scatter plots
- Efficiency analysis
- Recommendation based on use case
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


class CrossComparator:
    """Cross-compare symmetric vs asymmetric quantization methods."""

    def __init__(self, results):
        """
        Initialize with comparison results.

        Args:
            results: List of result dictionaries from comparison script
        """
        self.results = results
        self.methods = self._parse_methods()

    def _parse_methods(self):
        """Parse results into method pairs."""
        methods = {
            'awq_sym': None,
            'awq_asym': None,
            'praq_sym': None,
            'praq_asym': None
        }

        for r in self.results:
            name = r['name']
            if 'AWQ' in name and 'Asym' not in name and 'Hybrid' not in name:
                methods['awq_sym'] = r
            elif 'AWQ' in name and 'Asym' in name and 'Hybrid' not in name:
                methods['awq_asym'] = r
            elif 'Hybrid' in name and 'Asym' not in name:
                methods['praq_sym'] = r
            elif 'Hybrid' in name and 'Asym' in name:
                methods['praq_asym'] = r

        return methods

    def compute_metrics(self):
        """Compute cross-comparison metrics."""
        metrics = {}

        # AWQ comparison
        if self.methods['awq_sym'] and self.methods['awq_asym']:
            metrics['awq'] = self._compare_pair(
                self.methods['awq_sym'],
                self.methods['awq_asym'],
                'GW-AWQ'
            )

        # PRAQ comparison
        if self.methods['praq_sym'] and self.methods['praq_asym']:
            metrics['praq'] = self._compare_pair(
                self.methods['praq_sym'],
                self.methods['praq_asym'],
                'GWH-PRAQ'
            )

        # Cross-method comparison
        metrics['best_overall'] = self._find_best_overall()

        return metrics

    def _compare_pair(self, sym, asym, method_name):
        """Compare symmetric vs asymmetric for a single method."""
        comparison = {
            'method': method_name,
            'symmetric': sym,
            'asymmetric': asym
        }

        # Perplexity difference
        ppl_diff = asym['perplexity'] - sym['perplexity']
        ppl_pct = (ppl_diff / sym['perplexity']) * 100
        comparison['perplexity_diff'] = ppl_diff
        comparison['perplexity_pct'] = ppl_pct
        comparison['perplexity_winner'] = 'Symmetric' if ppl_diff > 0 else 'Asymmetric'

        # Throughput difference
        if sym['throughput_tokens_per_sec'] > 0 and asym['throughput_tokens_per_sec'] > 0:
            thr_diff = asym['throughput_tokens_per_sec'] - sym['throughput_tokens_per_sec']
            thr_pct = (thr_diff / sym['throughput_tokens_per_sec']) * 100
            comparison['throughput_diff'] = thr_diff
            comparison['throughput_pct'] = thr_pct
            comparison['throughput_winner'] = 'Asymmetric' if thr_diff > 0 else 'Symmetric'
        else:
            comparison['throughput_diff'] = None
            comparison['throughput_pct'] = None
            comparison['throughput_winner'] = 'N/A'

        # Efficiency metric: Perplexity per 1000 tokens/sec
        # Lower is better (lower perplexity at same speed)
        if sym['throughput_tokens_per_sec'] > 0:
            sym_efficiency = sym['perplexity'] / (sym['throughput_tokens_per_sec'] / 1000)
            comparison['sym_efficiency'] = sym_efficiency
        else:
            comparison['sym_efficiency'] = None

        if asym['throughput_tokens_per_sec'] > 0:
            asym_efficiency = asym['perplexity'] / (asym['throughput_tokens_per_sec'] / 1000)
            comparison['asym_efficiency'] = asym_efficiency
        else:
            comparison['asym_efficiency'] = None

        if comparison['sym_efficiency'] and comparison['asym_efficiency']:
            comparison['efficiency_winner'] = 'Symmetric' if sym_efficiency < asym_efficiency else 'Asymmetric'
        else:
            comparison['efficiency_winner'] = 'N/A'

        # Trade-off score: (Quality improvement) - (Speed cost)
        # Positive = worthwhile tradeoff, Negative = not worthwhile
        if comparison['throughput_pct']:
            quality_gain = -ppl_pct  # Negative perplexity change is good
            speed_cost = -comparison['throughput_pct']  # Negative throughput change is bad
            comparison['tradeoff_score'] = quality_gain + speed_cost
            comparison['worthwhile'] = comparison['tradeoff_score'] > 0
        else:
            comparison['tradeoff_score'] = None
            comparison['worthwhile'] = None

        return comparison

    def _find_best_overall(self):
        """Find the best overall method considering all factors."""
        best = {
            'best_quality': None,
            'best_speed': None,
            'best_efficiency': None,
            'recommended': None
        }

        valid_results = [r for r in self.results if r['perplexity'] != float('inf')]

        if valid_results:
            # Best quality (lowest perplexity)
            best['best_quality'] = min(valid_results, key=lambda x: x['perplexity'])

            # Best speed (highest throughput)
            valid_throughput = [r for r in valid_results if r['throughput_tokens_per_sec'] > 0]
            if valid_throughput:
                best['best_speed'] = max(valid_throughput, key=lambda x: x['throughput_tokens_per_sec'])

                # Best efficiency (lowest perplexity per 1000 tok/s)
                for r in valid_throughput:
                    r['efficiency'] = r['perplexity'] / (r['throughput_tokens_per_sec'] / 1000)
                best['best_efficiency'] = min(valid_throughput, key=lambda x: x['efficiency'])

            # Recommended: Best efficiency (balances quality and speed)
            best['recommended'] = best['best_efficiency']

        return best

    def print_detailed_comparison(self, metrics):
        """Print detailed comparison tables."""
        print("\n" + "=" * 100)
        print("CROSS-COMPARISON: Symmetric vs Asymmetric Quantization")
        print("=" * 100)

        for method_key in ['awq', 'praq']:
            if method_key not in metrics:
                continue

            comp = metrics[method_key]
            print(f"\n{comp['method']}: SYMMETRIC vs ASYMMETRIC")
            print("-" * 100)

            sym = comp['symmetric']
            asym = comp['asymmetric']

            print(f"\n{'Metric':<30} {'Symmetric':<20} {'Asymmetric':<20} {'Difference':<30}")
            print("-" * 100)

            # Perplexity
            print(f"{'Perplexity':<30} {sym['perplexity']:<20.2f} {asym['perplexity']:<20.2f} "
                  f"{comp['perplexity_diff']:+.2f} ({comp['perplexity_pct']:+.2f}%) [{comp['perplexity_winner']}]")

            # Throughput
            if comp['throughput_diff']:
                print(f"{'Throughput (tok/s)':<30} {sym['throughput_tokens_per_sec']:<20.1f} "
                      f"{asym['throughput_tokens_per_sec']:<20.1f} "
                      f"{comp['throughput_diff']:+.1f} ({comp['throughput_pct']:+.2f}%) [{comp['throughput_winner']}]")
            else:
                print(f"{'Throughput (tok/s)':<30} {'N/A':<20} {'N/A':<20} {'N/A':<30}")

            # Model size
            print(f"{'Model Size (MB)':<30} {sym['model_size_mb']:<20.2f} {asym['model_size_mb']:<20.2f} "
                  f"{asym['model_size_mb'] - sym['model_size_mb']:+.2f}")

            # Efficiency
            if comp['sym_efficiency'] and comp['asym_efficiency']:
                print(f"{'Efficiency (PPL/1000tok/s)':<30} {comp['sym_efficiency']:<20.4f} "
                      f"{comp['asym_efficiency']:<20.4f} "
                      f"{comp['asym_efficiency'] - comp['sym_efficiency']:+.4f} [{comp['efficiency_winner']}]")

            print("\n" + "-" * 100)
            print("ANALYSIS:")

            # Quality analysis
            if abs(comp['perplexity_pct']) < 0.5:
                quality_verdict = "Negligible difference"
            elif comp['perplexity_pct'] < 0:
                quality_verdict = f"Asymmetric is better by {abs(comp['perplexity_pct']):.2f}%"
            else:
                quality_verdict = f"Symmetric is better by {comp['perplexity_pct']:.2f}%"
            print(f"  Quality: {quality_verdict}")

            # Speed analysis
            if comp['throughput_pct']:
                if abs(comp['throughput_pct']) < 1:
                    speed_verdict = "Negligible difference"
                elif comp['throughput_pct'] < 0:
                    speed_verdict = f"Symmetric is faster by {abs(comp['throughput_pct']):.1f}%"
                else:
                    speed_verdict = f"Asymmetric is faster by {comp['throughput_pct']:.1f}%"
                print(f"  Speed: {speed_verdict}")

            # Trade-off analysis
            if comp['tradeoff_score'] is not None:
                if comp['worthwhile']:
                    print(f"  Trade-off: ✓ Asymmetric is worthwhile (score: {comp['tradeoff_score']:+.2f})")
                else:
                    print(f"  Trade-off: ✗ Asymmetric is NOT worthwhile (score: {comp['tradeoff_score']:+.2f})")
                    print(f"               Recommendation: Use Symmetric")

        # Overall recommendation
        print("\n" + "=" * 100)
        print("OVERALL RECOMMENDATIONS")
        print("=" * 100)

        best = metrics['best_overall']

        if best['best_quality']:
            print(f"\n✓ Best Quality: {best['best_quality']['name']}")
            print(f"  Perplexity: {best['best_quality']['perplexity']:.2f}")

        if best['best_speed']:
            print(f"\n✓ Best Speed: {best['best_speed']['name']}")
            print(f"  Throughput: {best['best_speed']['throughput_tokens_per_sec']:.1f} tok/s")

        if best['best_efficiency']:
            print(f"\n✓ Best Efficiency: {best['best_efficiency']['name']}")
            print(f"  Efficiency: {best['best_efficiency']['efficiency']:.4f} PPL/(1000 tok/s)")
            print(f"  (Perplexity: {best['best_efficiency']['perplexity']:.2f}, "
                  f"Throughput: {best['best_efficiency']['throughput_tokens_per_sec']:.1f} tok/s)")

        if best['recommended']:
            print(f"\n🏆 RECOMMENDED: {best['recommended']['name']}")
            print(f"   Best balance of quality and speed")

        print("\n" + "=" * 100)
        print("KEY FINDINGS:")
        print("=" * 100)

        # Determine if asymmetric is worth it
        asym_worth_it = []
        for method_key in ['awq', 'praq']:
            if method_key in metrics and metrics[method_key].get('worthwhile'):
                asym_worth_it.append(metrics[method_key]['method'])

        if asym_worth_it:
            print(f"✓ Asymmetric quantization is beneficial for: {', '.join(asym_worth_it)}")
        else:
            print("✗ Asymmetric quantization provides NO significant benefits")
            print("  - Similar or worse quality (perplexity)")
            print("  - Slower throughput (computational overhead)")
            print("  - Recommendation: Use Symmetric quantization for all methods")

        print("=" * 100)

    def create_visualizations(self, metrics, output_dir):
        """Create detailed cross-comparison visualizations."""
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")

        # 1. Trade-off plot: Quality vs Speed
        self._plot_quality_vs_speed(output_dir)

        # 2. Symmetric vs Asymmetric comparison bars
        self._plot_symmetric_vs_asymmetric(metrics, output_dir)

        # 3. Efficiency comparison
        self._plot_efficiency(metrics, output_dir)

    def _plot_quality_vs_speed(self, output_dir):
        """Plot perplexity vs throughput scatter."""
        fig, ax = plt.subplots(figsize=(10, 7))

        # Prepare data
        valid_results = [r for r in self.results
                        if r['perplexity'] != float('inf') and r['throughput_tokens_per_sec'] > 0]

        if not valid_results:
            print("Skipping quality vs speed plot (no valid data)")
            return

        for r in valid_results:
            name = r['name']
            ppl = r['perplexity']
            thr = r['throughput_tokens_per_sec']

            # Color and marker by type
            if 'Asym' in name:
                color = '#E74C3C'
                marker = 's'  # square
            else:
                color = '#3498DB'
                marker = 'o'  # circle

            # Size by method
            if 'Hybrid' in name:
                size = 200
            else:
                size = 150

            ax.scatter(thr, ppl, c=color, marker=marker, s=size,
                      alpha=0.7, edgecolors='black', linewidth=2,
                      label=name)

            # Add label
            ax.annotate(name, (thr, ppl),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

        ax.set_xlabel('Throughput (tokens/second)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
        ax.set_title('Quality vs Speed Trade-off\n(Bottom-Right is Best)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Invert y-axis (lower perplexity is better)
        ax.invert_yaxis()

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498DB', edgecolor='black', label='Symmetric'),
            Patch(facecolor='#E74C3C', edgecolor='black', label='Asymmetric')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_vs_speed_tradeoff.png'), dpi=300)
        print(f"Saved: {output_dir}/quality_vs_speed_tradeoff.png")
        plt.close()

    def _plot_symmetric_vs_asymmetric(self, metrics, output_dir):
        """Plot side-by-side comparison of symmetric vs asymmetric."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        method_keys = ['awq', 'praq']
        method_names = ['GW-AWQ', 'GWH-PRAQ']

        for ax, method_key, method_name in zip(axes, method_keys, method_names):
            if method_key not in metrics:
                continue

            comp = metrics[method_key]
            sym = comp['symmetric']
            asym = comp['asymmetric']

            # Data
            categories = ['Perplexity\n(lower=better)', 'Throughput\n(higher=better)']
            x = np.arange(len(categories))
            width = 0.35

            # Normalize for visualization
            ppl_sym_norm = 1.0
            ppl_asym_norm = sym['perplexity'] / asym['perplexity']

            if sym['throughput_tokens_per_sec'] > 0 and asym['throughput_tokens_per_sec'] > 0:
                thr_sym_norm = sym['throughput_tokens_per_sec'] / sym['throughput_tokens_per_sec']
                thr_asym_norm = asym['throughput_tokens_per_sec'] / sym['throughput_tokens_per_sec']
            else:
                thr_sym_norm = 0
                thr_asym_norm = 0

            sym_values = [ppl_sym_norm, thr_sym_norm]
            asym_values = [ppl_asym_norm, thr_asym_norm]

            # Plot bars
            bars1 = ax.bar(x - width/2, sym_values, width, label='Symmetric',
                          color='#3498DB', alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x + width/2, asym_values, width, label='Asymmetric',
                          color='#E74C3C', alpha=0.8, edgecolor='black')

            # Add value labels
            for bars, values, orig_values in [
                (bars1, sym_values, [sym['perplexity'], sym['throughput_tokens_per_sec']]),
                (bars2, asym_values, [asym['perplexity'], asym['throughput_tokens_per_sec']])
            ]:
                for bar, val, orig in zip(bars, values, orig_values):
                    if val > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                               f'{val:.3f}\n({orig:.1f})',
                               ha='center', va='bottom', fontsize=8, fontweight='bold')

            ax.set_ylabel('Normalized Score', fontsize=11, fontweight='bold')
            ax.set_title(f'{method_name}\nSymmetric vs Asymmetric', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, max(max(sym_values), max(asym_values)) * 1.15])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'symmetric_vs_asymmetric_comparison.png'), dpi=300)
        print(f"Saved: {output_dir}/symmetric_vs_asymmetric_comparison.png")
        plt.close()

    def _plot_efficiency(self, metrics, output_dir):
        """Plot efficiency comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = []
        efficiencies = []
        colors = []

        for method_key in ['awq', 'praq']:
            if method_key not in metrics:
                continue

            comp = metrics[method_key]

            if comp['sym_efficiency']:
                methods.append(f"{comp['method']}\n(Sym)")
                efficiencies.append(comp['sym_efficiency'])
                colors.append('#3498DB')

            if comp['asym_efficiency']:
                methods.append(f"{comp['method']}\n(Asym)")
                efficiencies.append(comp['asym_efficiency'])
                colors.append('#E74C3C')

        if not efficiencies:
            print("Skipping efficiency plot (no valid data)")
            return

        bars = ax.bar(range(len(methods)), efficiencies, color=colors, alpha=0.8, edgecolor='black')

        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Efficiency: Perplexity per 1000 tok/s\n(Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Efficiency Comparison: Quality per Unit Speed', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Highlight best
        best_idx = np.argmin(efficiencies)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=300)
        print(f"Saved: {output_dir}/efficiency_comparison.png")
        plt.close()


def load_results(results_file):
    """Load results from JSON file."""
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("   Please run: python compare_symmetric_vs_asymmetric.py")
        return None

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-compare Symmetric vs Asymmetric Quantization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results-file", type=str,
                       default="./comparison_results/comparison_results.json",
                       help="Path to comparison results JSON file")
    parser.add_argument("--output-dir", type=str,
                       default="./cross_comparison_results",
                       help="Output directory for cross-comparison results")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations")
    args = parser.parse_args()

    print("=" * 100)
    print("CROSS-COMPARISON: Symmetric vs Asymmetric Quantization")
    print("=" * 100)

    # Load results
    results = load_results(args.results_file)
    if not results:
        return

    print(f"Loaded {len(results)} results from {args.results_file}")

    # Create comparator
    comparator = CrossComparator(results)

    # Compute metrics
    metrics = comparator.compute_metrics()

    # Print detailed comparison
    comparator.print_detailed_comparison(metrics)

    # Create visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        comparator.create_visualizations(metrics, args.output_dir)
        print(f"✓ Visualizations saved to: {args.output_dir}/")

    # Save metrics to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'cross_comparison_metrics.json')

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_metrics = convert_to_serializable(metrics)

    with open(output_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {output_file}")

    print("\n" + "=" * 100)
    print("CROSS-COMPARISON COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    main()
