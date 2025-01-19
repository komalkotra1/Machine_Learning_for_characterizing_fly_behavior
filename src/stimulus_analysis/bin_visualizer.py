# src/stimulus_analysis/bin_visualizer.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class PlotStyles:
    """Configuration for plot styling."""
    figsize: Tuple[int, int] = (18, 6)
    before_color: str = 'orange'
    after_color: str = 'red'
    scatter_color: str = 'purple'
    bar_alpha: float = 0.5
    dpi: int = 300


class BinComparisonVisualizer:
    """
    Creates comparative visualizations for before/after stimulus analysis.
    
    Generates multiple plot types to visualize changes in heading angles
    before and after stimulus presentation.
    """
    
    def __init__(self, styles: Optional[PlotStyles] = None):
        """
        Initialize the visualizer.
        
        Args:
            styles: Configuration for plot appearance
        """
        self.styles = styles or PlotStyles()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load comparison data from CSV."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
            
    def create_line_plot(self, 
                        ax: plt.Axes,
                        before: np.ndarray,
                        after: np.ndarray) -> None:
        """Create line plot comparing before/after values."""
        ax.plot(before, 
               label='Before Stimulus',
               color=self.styles.before_color,
               marker='o')
        ax.plot(after,
               label='After Stimulus',
               color=self.styles.after_color,
               marker='o')
        ax.set_title("Line Plot: Saccade Angles")
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Heading Angle Difference (Degrees)')
        ax.legend()
        
    def create_scatter_plot(self,
                          ax: plt.Axes,
                          before: np.ndarray,
                          after: np.ndarray) -> None:
        """Create scatter plot of before vs after values."""
        ax.scatter(before, 
                  after,
                  label="Turns",
                  color=self.styles.scatter_color)
        
        # Add reference line
        min_val = min(before.min(), after.min())
        max_val = max(before.max(), after.max())
        ax.plot([min_val, max_val],
                [min_val, max_val],
                color='black',
                linestyle='--',
                label='y=x (No Change)')
                
        ax.set_title("Scatter Plot: Before vs After Stimulus")
        ax.set_xlabel('Before Stimulus')
        ax.set_ylabel('After Stimulus')
        ax.legend()
        
    def create_bar_plot(self,
                       ax: plt.Axes,
                       before: np.ndarray,
                       after: np.ndarray) -> None:
        """Create bar plot showing magnitude changes."""
        x = np.arange(len(before))
        
        ax.bar(x, 
               before,
               color=self.styles.before_color,
               label='Before Stimulus')
        ax.bar(x,
               after,
               color=self.styles.after_color,
               alpha=self.styles.bar_alpha,
               label='After Stimulus')
               
        ax.set_title("Bar Plot: Magnitude of Change")
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Heading Angle Difference (Degrees)')
        ax.legend()
        
    def create_comparison_plots(self,
                              data: pd.DataFrame,
                              before_col: str = 'diff_heading_Angle_degree_x',
                              after_col: str = 'diff_heading_Angle_degree_y') -> plt.Figure:
        """
        Create all comparison plots.
        
        Args:
            data: DataFrame containing before/after data
            before_col: Column name for before stimulus data
            after_col: Column name for after stimulus data
            
        Returns:
            matplotlib Figure object
        """
        before = data[before_col].values
        after = data[after_col].values
        
        fig, axs = plt.subplots(1, 3, figsize=self.styles.figsize)
        
        self.create_line_plot(axs[0], before, after)
        self.create_scatter_plot(axs[1], before, after)
        self.create_bar_plot(axs[2], before, after)
        
        plt.tight_layout()
        return fig
        
    def visualize_comparison(self,
                           input_file: str,
                           output_file: str,
                           show_plot: bool = True) -> None:
        """
        Complete visualization pipeline.
        
        Args:
            input_file: Path to comparison data CSV
            output_file: Path to save visualization
            show_plot: Whether to display plot
        """
        # Load and process data
        data = self.load_data(input_file)
        
        # Create visualization
        fig = self.create_comparison_plots(data)
        
        # Save plot
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.styles.dpi)
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)


def main():
    """Command line interface for bin comparison visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create comparative visualizations for bin analysis"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to comparison data CSV"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save visualization"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plot"
    )
    
    args = parser.parse_args()
    
    visualizer = BinComparisonVisualizer()
    
    try:
        visualizer.visualize_comparison(
            args.input,
            args.output,
            show_plot=not args.no_show
        )
        print(f"Visualization saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
