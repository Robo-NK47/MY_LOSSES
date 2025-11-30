"""
Loss Functions Visualizer
Creates visualizations for all loss functions listed in the Excel file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
import torch.nn.functional as F
from pathlib import Path
import warnings
import textwrap
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import io
warnings.filterwarnings('ignore')

# Set up beautiful, modern styling
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#E0E0E0',
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#E8E8E8',
    'grid.linewidth': 0.8,
    'grid.linestyle': '--',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'text.color': '#2C3E50',
    'axes.labelcolor': '#34495E',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#CCCCCC',
    'legend.facecolor': '#FFFFFF',
    'legend.shadow': True,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'mathtext.default': 'regular',  # Enable mathtext
    'mathtext.fontset': 'dejavusans'  # Use DejaVu Sans for math
})

# Beautiful color palettes
COLOR_PALETTES = {
    'regression': ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6'],
    'classification': ['#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3'],
    'segmentation': ['#00BCD4', '#009688', '#4CAF50', '#8BC34A'],
    'detection': ['#FF5722', '#FF9800', '#FFC107', '#FFEB3B'],
    'metric': ['#795548', '#607D8B', '#455A64', '#263238'],
    'other': ['#F44336', '#E91E63', '#9C27B0', '#673AB7']
}

# Gradient function for beautiful line plots
def create_gradient_line(ax, x, y, color, linewidth=3, alpha=0.8):
    """Create a gradient line effect"""
    for i in range(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], 
                color=color, linewidth=linewidth, alpha=alpha, 
                solid_capstyle='round', solid_joinstyle='round')

def style_axes(ax, title, xlabel, ylabel, color='#3498DB'):
    """Apply beautiful styling to axes"""
    ax.set_xlabel(xlabel, fontsize=13, fontweight='600', color='#34495E', labelpad=12)
    ax.set_ylabel(ylabel, fontsize=13, fontweight='600', color='#34495E', labelpad=10)
    ax.set_title(title, fontsize=17, fontweight='700', color='#2C3E50', pad=15)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, color='#E0E0E0', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    # Ensure x-axis ticks and labels are visible
    ax.tick_params(axis='x', which='major', pad=8, labelsize=10)
    ax.tick_params(axis='y', which='major', pad=5, labelsize=10)
    
def create_beautiful_figure(size=(12, 10)):
    """Create a beautifully styled figure for graph only (no text box)"""
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor('#FAFAFA')
    # Full graph area - no text box space needed, x-axis fully visible
    fig.subplots_adjust(bottom=0.12, top=0.95, left=0.1, right=0.95)
    # Ensure x-axis labels and ticks are visible with proper padding
    ax.tick_params(axis='x', which='major', pad=8, labelsize=10)
    ax.tick_params(axis='y', which='major', pad=5, labelsize=10)
    return fig, ax

def create_beautiful_figure_2subplots(size=(16, 10)):
    """Create a beautifully styled figure with 2 subplots for graph only (no text box)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size)
    fig.patch.set_facecolor('#FAFAFA')
    # Full graph area - no text box space needed, x-axis fully visible
    fig.subplots_adjust(bottom=0.12, top=0.95, left=0.08, right=0.95)
    # Ensure x-axis labels and ticks are visible for both subplots with proper padding
    ax1.tick_params(axis='x', which='major', pad=8, labelsize=10)
    ax1.tick_params(axis='y', which='major', pad=5, labelsize=10)
    ax2.tick_params(axis='x', which='major', pad=8, labelsize=10)
    ax2.tick_params(axis='y', which='major', pad=5, labelsize=10)
    return fig, ax1, ax2

def add_separator_line(fig):
    """Add elegant separator line between zones - DISABLED"""
    # Separator line removed per user request
    pass

# Dictionary for "what it does to inputs" information
WHAT_IT_DOES = {
    'MSE (Mean Squared Error)': 'Computes the average of squared differences between predictions and true values. Squares errors, making large errors much more significant.',
    'MSE': 'Computes the average of squared differences between predictions and true values. Squares errors, making large errors much more significant.',
    'MAE (Mean Absolute Error)': 'Computes the average of absolute differences between predictions and true values. Treats all errors equally regardless of magnitude.',
    'MAE': 'Computes the average of absolute differences between predictions and true values. Treats all errors equally regardless of magnitude.',
    'RMSE (Root Mean Squared Error)': 'Takes the square root of MSE to get errors back in the same units as the target variable. Still penalizes large errors heavily.',
    'RMSE': 'Takes the square root of MSE to get errors back in the same units as the target variable. Still penalizes large errors heavily.',
    'Huber Loss': 'Uses squared loss for small errors and linear loss for large errors. Smoothly transitions between MSE and MAE behavior.',
    'Huber': 'Uses squared loss for small errors and linear loss for large errors. Smoothly transitions between MSE and MAE behavior.',
    'Log-Cosh Loss': 'Applies logarithm to hyperbolic cosine of errors. Produces smooth, twice-differentiable loss that behaves like squared loss for small errors.',
    'Log-Cosh': 'Applies logarithm to hyperbolic cosine of errors. Produces smooth, twice-differentiable loss that behaves like squared loss for small errors.',
    'Smooth L1 Loss': 'Uses squared loss for small errors and linear loss for large errors. Provides smooth gradient everywhere, useful for bounding box regression.',
    'Smooth L1': 'Uses squared loss for small errors and linear loss for large errors. Provides smooth gradient everywhere, useful for bounding box regression.',
    'Binary Cross-Entropy (BCE)': 'Measures the difference between predicted probabilities and true binary labels. Applies negative log-likelihood to probability outputs.',
    'Binary Cross-Entropy': 'Measures the difference between predicted probabilities and true binary labels. Applies negative log-likelihood to probability outputs.',
    'BCE With Logits': 'Applies sigmoid activation and then binary cross-entropy in a numerically stable way. Combines sigmoid and BCE to prevent overflow.',
    'Cross-Entropy (CE)': 'Measures the difference between predicted class probabilities and true class distribution. Applies negative log-likelihood to multi-class probabilities.',
    'Cross-Entropy': 'Measures the difference between predicted class probabilities and true class distribution. Applies negative log-likelihood to multi-class probabilities.',
    'Sparse Cross-Entropy': 'Same as cross-entropy but accepts integer class labels directly instead of one-hot encoded vectors. More memory efficient.',
    'Focal Loss': 'Modifies cross-entropy by down-weighting easy examples and focusing on hard examples. Multiplies CE by a focusing factor based on prediction confidence.',
    'Focal': 'Modifies cross-entropy by down-weighting easy examples and focusing on hard examples. Multiplies CE by a focusing factor based on prediction confidence.',
    'Hinge Loss': 'Penalizes predictions that are on the wrong side of the margin. Maximizes the margin between correct and incorrect predictions.',
    'Hinge': 'Penalizes predictions that are on the wrong side of the margin. Maximizes the margin between correct and incorrect predictions.',
    'Dice Loss': 'Measures overlap between predicted and true segmentation masks. Computes 1 minus the Dice coefficient (intersection over union).',
    'Dice': 'Measures overlap between predicted and true segmentation masks. Computes 1 minus the Dice coefficient (intersection over union).',
    'IoU Loss': 'Directly optimizes the Intersection over Union metric. Computes 1 minus IoU between predicted and true regions.',
    'IoU': 'Directly optimizes the Intersection over Union metric. Computes 1 minus IoU between predicted and true regions.',
    'Tversky Loss': 'Generalizes Dice loss with asymmetric penalty for false positives and false negatives. Allows control over precision-recall tradeoff.',
    'Tversky': 'Generalizes Dice loss with asymmetric penalty for false positives and false negatives. Allows control over precision-recall tradeoff.',
    'GIoU Loss': 'Extends IoU to handle non-overlapping boxes by considering the smallest enclosing box. Computes IoU minus normalized area of enclosing box.',
    'GIoU': 'Extends IoU to handle non-overlapping boxes by considering the smallest enclosing box. Computes IoU minus normalized area of enclosing box.',
    'DIoU Loss': 'Adds distance penalty between box centers to IoU loss. Considers both overlap and spatial distance between predicted and true boxes.',
    'DIoU': 'Adds distance penalty between box centers to IoU loss. Considers both overlap and spatial distance between predicted and true boxes.',
    'CIoU Loss': 'Combines DIoU with aspect ratio consistency. Considers overlap, distance, and aspect ratio similarity between boxes.',
    'CIoU': 'Combines DIoU with aspect ratio consistency. Considers overlap, distance, and aspect ratio similarity between boxes.',
    'KLDivergence': 'Measures how different one probability distribution is from another. Computes expected log-ratio between distributions.',
    'Triplet Loss': 'Pulls anchor-positive pairs together and pushes anchor-negative pairs apart in embedding space. Uses margin-based ranking.',
    'Triplet': 'Pulls anchor-positive pairs together and pushes anchor-negative pairs apart in embedding space. Uses margin-based ranking.',
    'Contrastive Loss': 'Minimizes distance for similar pairs and maximizes distance for dissimilar pairs. Uses margin to prevent collapse.',
    'Contrastive': 'Minimizes distance for similar pairs and maximizes distance for dissimilar pairs. Uses margin to prevent collapse.',
    'InfoNCE Loss': 'Maximizes mutual information between positive pairs relative to negative samples. Uses contrastive learning with temperature scaling.',
    'InfoNCE': 'Maximizes mutual information between positive pairs relative to negative samples. Uses contrastive learning with temperature scaling.',
    'CTC Loss': 'Aligns sequences without requiring explicit alignment. Computes probability of all possible alignments between input and output sequences.',
    'CTC': 'Aligns sequences without requiring explicit alignment. Computes probability of all possible alignments between input and output sequences.',
    'GAN BCE Loss': 'Trains discriminator to distinguish real from fake samples. Uses binary classification loss on discriminator outputs.',
    'WGAN Loss': 'Uses Wasserstein distance instead of JS divergence. Provides more stable gradients by using critic scores directly as loss.',
    'WGAN': 'Uses Wasserstein distance instead of JS divergence. Provides more stable gradients by using critic scores directly as loss.'
}

# Dictionary for "when to use" information
WHEN_TO_USE = {
    'MSE (Mean Squared Error)': 'Use when you want to penalize large errors heavily. Good for most regression tasks.',
    'MSE': 'Use when you want to penalize large errors heavily. Good for most regression tasks.',
    'MAE (Mean Absolute Error)': 'Use when you want equal penalty for all errors. Robust to outliers.',
    'MAE': 'Use when you want equal penalty for all errors. Robust to outliers.',
    'RMSE (Root Mean Squared Error)': 'Use when you want errors in the same units as the target variable. Penalizes large errors.',
    'RMSE': 'Use when you want errors in the same units as the target variable. Penalizes large errors.',
    'Huber Loss': 'Use when you have outliers in your data. Combines benefits of MSE and MAE.',
    'Huber': 'Use when you have outliers in your data. Combines benefits of MSE and MAE.',
    'Log-Cosh Loss': 'Use for smooth regression. Similar to Huber but twice differentiable everywhere.',
    'Log-Cosh': 'Use for smooth regression. Similar to Huber but twice differentiable everywhere.',
    'Smooth L1 Loss': 'Use for bounding box regression in object detection. Less sensitive to outliers than L2.',
    'Smooth L1': 'Use for bounding box regression in object detection. Less sensitive to outliers than L2.',
    'Binary Cross-Entropy (BCE)': 'Use for binary classification tasks (2 classes). Standard choice for binary problems.',
    'Binary Cross-Entropy': 'Use for binary classification tasks (2 classes). Standard choice for binary problems.',
    'BCE With Logits': 'Use for binary classification with numerical stability. Prevents overflow/underflow.',
    'Cross-Entropy (CE)': 'Use for multi-class classification (3+ classes). Standard choice for classification.',
    'Cross-Entropy': 'Use for multi-class classification (3+ classes). Standard choice for classification.',
    'Sparse Cross-Entropy': 'Use when labels are integers (not one-hot encoded). More memory efficient.',
    'Focal Loss': 'Use when dealing with class imbalance. Focuses learning on hard examples.',
    'Focal': 'Use when dealing with class imbalance. Focuses learning on hard examples.',
    'Hinge Loss': 'Use for Support Vector Machines (SVMs). Maximizes margin between classes.',
    'Hinge': 'Use for Support Vector Machines (SVMs). Maximizes margin between classes.',
    'Dice Loss': 'Use for image segmentation tasks. Handles class imbalance well.',
    'Dice': 'Use for image segmentation tasks. Handles class imbalance well.',
    'IoU Loss': 'Use for segmentation when you care about overlap accuracy. Directly optimizes IoU metric.',
    'IoU': 'Use for segmentation when you care about overlap accuracy. Directly optimizes IoU metric.',
    'Tversky Loss': 'Use for segmentation with imbalanced classes. Allows control over false positives/negatives.',
    'Tversky': 'Use for segmentation with imbalanced classes. Allows control over false positives/negatives.',
    'GIoU Loss': 'Use for object detection bounding box regression. Handles non-overlapping boxes.',
    'GIoU': 'Use for object detection bounding box regression. Handles non-overlapping boxes.',
    'DIoU Loss': 'Use for object detection. Considers distance between box centers.',
    'DIoU': 'Use for object detection. Considers distance between box centers.',
    'CIoU Loss': 'Use for object detection. Considers aspect ratio, distance, and overlap.',
    'CIoU': 'Use for object detection. Considers aspect ratio, distance, and overlap.',
    'KLDivergence': 'Use when matching probability distributions. Common in variational autoencoders.',
    'Triplet Loss': 'Use for learning embeddings where similar items should be close. Common in face recognition.',
    'Triplet': 'Use for learning embeddings where similar items should be close. Common in face recognition.',
    'Contrastive Loss': 'Use for learning representations where similar pairs should be close, dissimilar pairs far.',
    'Contrastive': 'Use for learning representations where similar pairs should be close, dissimilar pairs far.',
    'InfoNCE Loss': 'Use for self-supervised learning and contrastive learning. Maximizes mutual information.',
    'InfoNCE': 'Use for self-supervised learning and contrastive learning. Maximizes mutual information.',
    'CTC Loss': 'Use for sequence alignment problems like speech recognition and OCR.',
    'CTC': 'Use for sequence alignment problems like speech recognition and OCR.',
    'GAN BCE Loss': 'Use for training Generative Adversarial Networks. Standard discriminator loss.',
    'WGAN Loss': 'Use for Wasserstein GANs. Provides more stable training than standard GAN loss.',
    'WGAN': 'Use for Wasserstein GANs. Provides more stable training than standard GAN loss.'
}

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = Path('loss_visualizations')
    output_dir.mkdir(exist_ok=True)
    return output_dir

def figure_to_image(fig, dpi=300):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf)
    return img

def convert_formula_to_latex(formula):
    """Convert formula text to LaTeX format for mathematical rendering"""
    if pd.isna(formula) or formula == 'N/A':
        return 'N/A'
    
    formula = str(formula)
    
    # Fix encoding issues first
    replacements = {
        'Î£': 'Σ', 'Î´': 'δ', 'Î±': 'α', 'Î²': 'β', 'Î³': 'γ',
        'Â²': '²', 'âˆ©': '∩', 'âˆª': '∪', 'âˆš': '√',
    }
    for old, new in replacements.items():
        formula = formula.replace(old, new)
    
    # Convert to LaTeX format using matplotlib's mathtext
    import re
    
    # Handle superscripts first
    formula = formula.replace('²', '^2')
    
    # Convert Greek letters BEFORE other conversions to avoid conflicts
    # Add space after Greek letters if followed by uppercase letter to prevent merging
    formula = re.sub(r'α([A-Z])', r'\\alpha \1', formula)
    formula = re.sub(r'α', r'\\alpha', formula)
    formula = re.sub(r'β([A-Z])', r'\\beta \1', formula)
    formula = re.sub(r'β', r'\\beta', formula)
    formula = re.sub(r'γ([A-Z])', r'\\gamma \1', formula)
    formula = re.sub(r'γ', r'\\gamma', formula)
    formula = re.sub(r'δ([A-Z])', r'\\delta \1', formula)
    formula = re.sub(r'δ', r'\\delta', formula)
    formula = re.sub(r'Δ([A-Z])', r'\\Delta \1', formula)
    formula = re.sub(r'Δ', r'\\Delta', formula)
    
    # Convert sqrt with parentheses to curly braces: sqrt(...) -> \sqrt{...}
    # Do this BEFORE converting 'sum' to avoid conflicts
    formula = re.sub(r'sqrt\s*\(([^)]+)\)', r'\\sqrt{\1}', formula)
    formula = re.sub(r'√\s*\(([^)]+)\)', r'\\sqrt{\1}', formula)
    formula = re.sub(r'√\s*([^\s\)]+)', r'\\sqrt{\1}', formula)
    
    # Convert summation notation: Σ -> \sum (only standalone Σ symbol, not in words)
    # Handle patterns like "1/n Σ" or standalone "Σ"
    formula = re.sub(r'(\d+)/n\s*Σ', r'\1/n \\sum', formula)
    formula = re.sub(r'\s+Σ\s+', r' \\sum ', formula)  # Σ with spaces around
    formula = re.sub(r'Σ\s*\(', r'\\sum(', formula)  # Σ followed by (
    formula = re.sub(r'Σ([^A-Za-z])', r'\\sum\1', formula)  # Σ followed by non-letter
    
    # Convert common functions (log, exp, max, min, cosh)
    # Use word boundaries to avoid replacing parts of words like "sumexp"
    formula = re.sub(r'\blog\s*\(', r'\\log(', formula)
    formula = re.sub(r'\bexp\s*\(', r'\\exp(', formula)
    formula = re.sub(r'\bmax\s*\(', r'\\max(', formula)
    formula = re.sub(r'\bmin\s*\(', r'\\min(', formula)
    formula = re.sub(r'\bcosh\s*\(', r'\\cosh(', formula)
    
    # Handle subscripts: y_pred -> y_{pred}, y_true -> y_{true}
    formula = re.sub(r'y_pred', r'y_{pred}', formula)
    formula = re.sub(r'y_true', r'y_{true}', formula)
    
    # Handle fractions: 1/n -> \frac{1}{n}
    formula = re.sub(r'(\d+)/n', r'\\frac{\1}{n}', formula)
    
    # Handle parentheses with expressions: (y_pred - y_true)^2
    formula = re.sub(r'\(y_{pred}\s*-\s*y_{true}\)\^?2', r'(y_{pred} - y_{true})^2', formula)
    
    # Clean up spaces
    formula = formula.replace('  ', ' ').strip()
    
    # Wrap in $ signs for mathtext rendering
    if formula and formula != 'N/A':
        if not formula.startswith('$'):
            formula = f'${formula}$'
    
    return formula

def create_textbox_figure(row, width_inches=12, height_inches=2.5, accent_color='#3498DB', dpi=300):
    """Create a separate figure for the text box"""
    fig = plt.figure(figsize=(width_inches, height_inches), facecolor='#FAFAFA', dpi=dpi)
    fig.patch.set_facecolor('#FAFAFA')
    
    # Get formula and other info
    formula_raw = str(row['formula of loss function']) if pd.notna(row['formula of loss function']) else 'N/A'
    use_case = str(row['what it is used for']) if pd.notna(row['what it is used for']) else 'N/A'
    loss_name = str(row['name of loss function'])
    
    # Convert formula to LaTeX format
    formula = convert_formula_to_latex(formula_raw)
    
    # Get "what it does" information
    what_it_does = 'N/A'
    loss_name_lower = loss_name.lower()
    if loss_name in WHAT_IT_DOES:
        what_it_does = WHAT_IT_DOES[loss_name]
    else:
        matches = [(key, WHAT_IT_DOES[key]) for key in WHAT_IT_DOES if key.lower() in loss_name_lower]
        if matches:
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            what_it_does = matches[0][1]
    
    # Get "when to use" information
    when_to_use = 'N/A'
    if loss_name in WHEN_TO_USE:
        when_to_use = WHEN_TO_USE[loss_name]
    else:
        matches = [(key, WHEN_TO_USE[key]) for key in WHEN_TO_USE if key.lower() in loss_name_lower]
        if matches:
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            when_to_use = matches[0][1]
    
    # Format text content - separate formula label from formula itself
    max_width = 90
    formula_label = "📐 Formula:"
    formula_wrapped = f"{formula_label}\n{formula}"  # Put formula on next line
    what_it_does_wrapped = '\n'.join(textwrap.wrap(f"⚙️ What it does: {what_it_does}", width=max_width))
    use_case_wrapped = '\n'.join(textwrap.wrap(f"🎯 Used for: {use_case}", width=max_width))
    when_to_use_wrapped = '\n'.join(textwrap.wrap(f"💡 When to use: {when_to_use}", width=max_width))
    text_content = f"{formula_wrapped}\n\n{what_it_does_wrapped}\n\n{use_case_wrapped}\n\n{when_to_use_wrapped}"
    
    # Calculate optimal font size
    lines = text_content.split('\n')
    num_lines = len(lines)
    max_line_length = max(len(line) for line in lines) if lines else 1
    
    available_width_inches = width_inches * 0.9
    available_height_inches = height_inches * 0.9
    
    fontsize_width = (available_width_inches * 0.95 * 72) / (max_line_length * 0.55)
    fontsize_height = (available_height_inches * 0.92 * 72) / (num_lines * 1.15)
    
    optimal_fontsize = min(fontsize_width, fontsize_height, 16)
    optimal_fontsize = max(optimal_fontsize, 8)
    
    # Create text box background
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Background rectangle
    bg_rect = FancyBboxPatch(
        (0.05, 0.05), 0.9, 0.9,
        boxstyle="round,pad=0.02",
        transform=ax.transAxes,
        facecolor='#F8F9FA',
        edgecolor=accent_color,
        linewidth=2.5,
        zorder=0
    )
    ax.add_patch(bg_rect)
    
    # Shadow
    shadow_rect = FancyBboxPatch(
        (0.053, 0.047), 0.9, 0.9,
        boxstyle="round,pad=0.02",
        transform=ax.transAxes,
        facecolor='#000000',
        edgecolor='none',
        zorder=-1,
        alpha=0.1
    )
    ax.add_patch(shadow_rect)
    
    # Accent line at top
    accent_line = Rectangle(
        (0.05, 0.94), 0.9, 0.01,
        transform=ax.transAxes,
        facecolor=accent_color,
        edgecolor='none',
        zorder=1
    )
    ax.add_patch(accent_line)
    
    # Add text with LaTeX formula rendering
    # Split content to handle formula separately with math rendering
    lines = text_content.split('\n')
    
    # Find formula line and render it separately
    y_positions = []
    current_y = 0.7  # Start from top
    
    for i, line in enumerate(lines):
        if line.strip() == '':
            current_y -= 0.08  # Space between sections
            continue
            
        # Check if this line contains the formula (has $ signs for LaTeX)
        if '$' in line and 'Formula:' in line:
            # Split label and formula
            if 'Formula:' in line:
                parts = line.split('Formula:', 1)
                if len(parts) == 2:
                    label = parts[0] + 'Formula:'
                    formula_part = parts[1].strip()
                    # Render label on one line
                    ax.text(0.1, current_y, label, transform=ax.transAxes,
                           fontsize=optimal_fontsize, verticalalignment='center',
                           horizontalalignment='left', color='#2C3E50',
                           family='sans-serif', weight='normal', zorder=2)
                    # Render formula on the next line with proper spacing
                    current_y -= 0.08  # Move down for formula
                    # Wrap long formulas if needed
                    if len(formula_part) > 80:
                        # Split formula into multiple lines
                        formula_lines = textwrap.wrap(formula_part, width=80)
                        for formula_line in formula_lines:
                            ax.text(0.1, current_y, formula_line, transform=ax.transAxes,
                                   fontsize=optimal_fontsize, verticalalignment='center',
                                   horizontalalignment='left', color='#2C3E50',
                                   family='sans-serif', weight='normal', zorder=2,
                                   usetex=False)
                            current_y -= 0.06
                    else:
                        ax.text(0.1, current_y, formula_part, transform=ax.transAxes,
                               fontsize=optimal_fontsize, verticalalignment='center',
                               horizontalalignment='left', color='#2C3E50',
                               family='sans-serif', weight='normal', zorder=2,
                               usetex=False)  # Use mathtext, not full LaTeX
                        current_y -= 0.06
                    current_y -= 0.04  # Extra space after formula
                    continue
        
        # Regular text line
        ax.text(0.1, current_y, line, transform=ax.transAxes,
               fontsize=optimal_fontsize, verticalalignment='center',
               horizontalalignment='left', color='#2C3E50',
               family='sans-serif', weight='normal', zorder=2)
        current_y -= 0.06
    
    return fig

def combine_graph_and_textbox(graph_fig, textbox_fig, output_path, dpi=300):
    """Combine graph and textbox figures into a single image"""
    # Convert figures to images
    graph_img = figure_to_image(graph_fig, dpi=dpi)
    textbox_img = figure_to_image(textbox_fig, dpi=dpi)
    
    # Get dimensions
    graph_width, graph_height = graph_img.size
    textbox_width, textbox_height = textbox_img.size
    
    # Calculate combined dimensions (match widths, stack vertically)
    combined_width = max(graph_width, textbox_width)
    combined_height = graph_height + textbox_height
    
    # Create combined image
    combined_img = Image.new('RGB', (combined_width, combined_height), color='#FAFAFA')
    
    # Paste graph at top
    graph_x_offset = (combined_width - graph_width) // 2
    combined_img.paste(graph_img, (graph_x_offset, 0))
    
    # Paste textbox at bottom
    textbox_x_offset = (combined_width - textbox_width) // 2
    combined_img.paste(textbox_img, (textbox_x_offset, graph_height))
    
    # Save combined image
    combined_img.save(output_path, dpi=(dpi, dpi), quality=95)
    
    # Close figures
    plt.close(graph_fig)
    plt.close(textbox_fig)

def add_info_textbox(fig, row, text_zone_bottom=0.0, text_zone_height=0.25, accent_color='#3498DB'):
    """Add beautiful information text box in the lower zone with formula, use case, and when to use"""
    formula = str(row['formula of loss function']) if pd.notna(row['formula of loss function']) else 'N/A'
    use_case = str(row['what it is used for']) if pd.notna(row['what it is used for']) else 'N/A'
    loss_name = str(row['name of loss function'])
    
    # Fix formula display - replace common encoding issues with proper mathematical symbols
    # Handle UTF-8 encoding issues from Excel (Windows-1252 to UTF-8 conversion issues)
    replacements = {
        'Î£': 'Σ',      # Sigma (summation)
        'Î´': 'δ',      # Delta
        'Î±': 'α',      # Alpha
        'Î²': 'β',      # Beta
        'Î³': 'γ',      # Gamma
        'Â²': '²',      # Superscript 2
        'âˆ©': '∩',     # Intersection
        'âˆª': '∪',     # Union
        'âˆš': '√',     # Square root symbol
    }
    
    for old, new in replacements.items():
        formula = formula.replace(old, new)
    
    # Additional fixes for common mathematical notation
    # Don't replace 'sqrt', 'log', 'exp', 'max', 'min' as they're already correct
    # Just ensure proper spacing
    formula = formula.replace('  ', ' ')  # Remove double spaces
    formula = formula.strip()  # Remove leading/trailing whitespace
    
    # Get "what it does" information
    what_it_does = 'N/A'
    loss_name_lower = loss_name.lower()
    if loss_name in WHAT_IT_DOES:
        what_it_does = WHAT_IT_DOES[loss_name]
    else:
        matches = [(key, WHAT_IT_DOES[key]) for key in WHAT_IT_DOES if key.lower() in loss_name_lower]
        if matches:
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            what_it_does = matches[0][1]
    
    # Get "when to use" information - try exact match first, then partial
    when_to_use = 'N/A'
    
    # Try exact match first
    if loss_name in WHEN_TO_USE:
        when_to_use = WHEN_TO_USE[loss_name]
    else:
        # Try partial matches (prioritize longer matches)
        matches = [(key, WHEN_TO_USE[key]) for key in WHEN_TO_USE if key.lower() in loss_name_lower]
        if matches:
            # Sort by key length (longer = more specific)
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            when_to_use = matches[0][1]
    
    # Calculate text zone dimensions
    text_zone_left = 0.05
    text_zone_right = 0.95
    text_zone_top = text_zone_bottom + text_zone_height
    text_zone_center_y = text_zone_bottom + text_zone_height / 2
    text_zone_width = text_zone_right - text_zone_left
    
    # Create beautifully formatted text content
    # Start with a reasonable max_width and adjust based on available space
    max_width = 90  # characters per line
    
    # Format with emoji-like symbols for visual appeal
    # Separate formula label from formula itself to prevent overlap
    formula_label = "📐 Formula:"
    formula_wrapped = f"{formula_label}\n{formula}"  # Put formula on next line
    what_it_does_wrapped = '\n'.join(textwrap.wrap(f"⚙️ What it does: {what_it_does}", width=max_width))
    use_case_wrapped = '\n'.join(textwrap.wrap(f"🎯 Used for: {use_case}", width=max_width))
    when_to_use_wrapped = '\n'.join(textwrap.wrap(f"💡 When to use: {when_to_use}", width=max_width))
    
    text_content = f"{formula_wrapped}\n\n{what_it_does_wrapped}\n\n{use_case_wrapped}\n\n{when_to_use_wrapped}"
    
    # Calculate optimal font size based on text content and available space
    # Estimate text dimensions
    lines = text_content.split('\n')
    num_lines = len(lines)
    max_line_length = max(len(line) for line in lines) if lines else 1
    
    # Estimate character width and line height (approximate)
    # For sans-serif font, average char width is about 0.6 * fontsize, line height is about 1.2 * fontsize
    # Available width in figure coordinates: text_zone_width (0.9 of figure width)
    # Available height in figure coordinates: text_zone_height (0.35 of figure height)
    
    # Calculate font size based on width constraint
    fig_width_inches = fig.get_size_inches()[0]
    available_width_inches = text_zone_width * fig_width_inches
    # Use more of the available space (0.95 factor - minimal padding)
    fontsize_width = (available_width_inches * 0.95 * 72) / (max_line_length * 0.55)  # 72 points per inch, tighter char width
    
    # Calculate font size based on height constraint
    fig_height_inches = fig.get_size_inches()[1]
    available_height_inches = text_zone_height * fig_height_inches
    # Use more of the available space (0.92 factor - minimal padding)
    fontsize_height = (available_height_inches * 0.92 * 72) / (num_lines * 1.15)  # 72 points per inch, tighter line height
    
    # Use the smaller of the two to ensure text fits
    # Increased maximum cap to allow larger text
    optimal_fontsize = min(fontsize_width, fontsize_height, 16)  # Cap at 16 for maximum readability
    optimal_fontsize = max(optimal_fontsize, 8)  # Minimum 8 for readability
    
    # If text is still too long, reduce max_width and rewrap
    if optimal_fontsize < 9:
        max_width = int(max_width * 0.9)
        formula_wrapped = '\n'.join(textwrap.wrap(f"📐 Formula: {formula}", width=max_width))
        use_case_wrapped = '\n'.join(textwrap.wrap(f"🎯 Used for: {use_case}", width=max_width))
        when_to_use_wrapped = '\n'.join(textwrap.wrap(f"💡 When to use: {when_to_use}", width=max_width))
        text_content = f"{formula_wrapped}\n\n{use_case_wrapped}\n\n{when_to_use_wrapped}"
        # Recalculate with new line count
        lines = text_content.split('\n')
        num_lines = len(lines)
        max_line_length = max(len(line) for line in lines) if lines else 1
        fontsize_width = (available_width_inches * 0.95 * 72) / (max_line_length * 0.55)
        fontsize_height = (available_height_inches * 0.92 * 72) / (num_lines * 1.15)
        optimal_fontsize = min(fontsize_width, fontsize_height, 16)
        optimal_fontsize = max(optimal_fontsize, 8)
    
    # Create beautiful background with gradient effect
    bg_rect = FancyBboxPatch(
        (text_zone_left, text_zone_bottom), 
        text_zone_width, 
        text_zone_height,
        boxstyle="round,pad=0.01",
        transform=fig.transFigure,
        facecolor='#F8F9FA',
        edgecolor=accent_color,
        linewidth=2.5,
        linestyle='-',
        zorder=0,
        alpha=0.95
    )
    fig.patches.append(bg_rect)
    
    # Add subtle shadow effect
    shadow_rect = FancyBboxPatch(
        (text_zone_left + 0.003, text_zone_bottom - 0.003), 
        text_zone_width, 
        text_zone_height,
        boxstyle="round,pad=0.01",
        transform=fig.transFigure,
        facecolor='#000000',
        edgecolor='none',
        zorder=-1,
        alpha=0.1
    )
    fig.patches.append(shadow_rect)
    
    # Add accent line at top
    accent_line = Rectangle(
        (text_zone_left, text_zone_top - 0.01),
        text_zone_width,
        0.005,
        transform=fig.transFigure,
        facecolor=accent_color,
        edgecolor='none',
        zorder=1,
        alpha=0.8
    )
    fig.patches.append(accent_line)
    
    # Add text with scaled font size
    fig.text(text_zone_left + 0.02, text_zone_center_y, text_content,
            transform=fig.transFigure,
            fontsize=optimal_fontsize,
            verticalalignment='center',
            horizontalalignment='left',
            color='#2C3E50',
            family='sans-serif',
            weight='normal',
            zorder=2)

def visualize_regression_losses(df, output_dir):
    """Visualize regression loss functions"""
    regression_losses = df[df['what it is used for'].str.contains('Regression', case=False, na=False)]
    
    for idx, row in regression_losses.iterrows():
        loss_name = row['name of loss function']
        print(f"Visualizing: {loss_name}")
        
        # Create error range (difference between predicted and true)
        error = np.linspace(-5, 5, 1000)
        y_pred = error  # Assuming y_true = 0 for simplicity
        y_true = np.zeros_like(y_pred)
        
        # Convert to tensors
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
        
        # Calculate loss based on function name
        if 'MSE' in loss_name and 'RMSE' not in loss_name:
            loss = (y_pred_tensor - y_true_tensor).pow(2).mean()
            loss_values = (y_pred - y_true) ** 2
        
        elif 'MAE' in loss_name:
            loss_values = np.abs(y_pred - y_true)
        
        elif 'RMSE' in loss_name:
            loss_values = np.sqrt((y_pred - y_true) ** 2)
        
        elif 'Huber' in loss_name:
            delta = 1.0
            abs_error = np.abs(y_pred - y_true)
            loss_values = np.where(abs_error <= delta,
                                  0.5 * abs_error ** 2,
                                  delta * (abs_error - 0.5 * delta))
        
        elif 'Log-Cosh' in loss_name:
            loss_values = np.log(np.cosh(y_pred - y_true))
        
        else:
            continue
        
        # Step 1: Create graph figure separately
        fig, ax = create_beautiful_figure((12, 10))
        
        # Choose beautiful color based on loss type
        color = COLOR_PALETTES['regression'][idx % len(COLOR_PALETTES['regression'])]
        
        # Create gradient fill under the curve
        ax.fill_between(error, loss_values, alpha=0.2, color=color, zorder=1)
        
        # Plot with beautiful styling
        ax.plot(error, loss_values, linewidth=3.5, label=loss_name, color=color, 
                zorder=2, antialiased=True, solid_capstyle='round')
        
        # Beautiful axis styling
        style_axes(ax, f'{loss_name} Loss Function', 'Error (y_pred - y_true)', 'Loss', color)
        
        # Beautiful legend
        legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True,
                          loc='best', framealpha=0.95, edgecolor='#CCCCCC')
        legend.get_frame().set_facecolor('#FFFFFF')
        
        ax.set_xlim(-5, 5)
        
        # Step 2: Create textbox figure separately
        textbox_fig = create_textbox_figure(row, width_inches=12, height_inches=2.5, accent_color=color)
        
        # Step 3 & 4: Combine and save
        filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)

def visualize_classification_losses(df, output_dir):
    """Visualize classification loss functions"""
    classification_losses = df[df['what it is used for'].str.contains('classification', case=False, na=False)]
    
    for idx, row in classification_losses.iterrows():
        loss_name = row['name of loss function']
        print(f"Visualizing: {loss_name}")
        
        # Create probability range
        p = np.linspace(0.001, 0.999, 1000)  # Avoid log(0)
        
        if 'Binary Cross-Entropy' in loss_name and 'Logits' not in loss_name:
            # For BCE: y_true = 1
            y_true = 1
            loss_values = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            
            fig, ax1, ax2 = create_beautiful_figure_2subplots((16, 10))
            color1 = COLOR_PALETTES['classification'][0]  # Beautiful pink/purple
            color2 = COLOR_PALETTES['classification'][1]  # Beautiful purple
            
            # y_true = 1
            ax1.fill_between(p, loss_values, alpha=0.2, color=color1, zorder=1)
            ax1.plot(p, loss_values, linewidth=3.5, label='y_true = 1', color=color1, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            style_axes(ax1, f'{loss_name} (y_true = 1)', 'Predicted Probability p', 'Loss', color1)
            ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            ax1.get_legend().get_frame().set_facecolor('#FFFFFF')
            
            # y_true = 0
            y_true = 0
            loss_values = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            ax2.fill_between(p, loss_values, alpha=0.2, color=color2, zorder=1)
            ax2.plot(p, loss_values, linewidth=3.5, label='y_true = 0', color=color2, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            style_axes(ax2, f'{loss_name} (y_true = 0)', 'Predicted Probability p', 'Loss', color2)
            ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            ax2.get_legend().get_frame().set_facecolor('#FFFFFF')
            
            plt.suptitle(f'{loss_name} Loss Function', fontsize=18, fontweight='700', color='#2C3E50', y=0.98)
            
            # Step 2: Create textbox figure separately
            textbox_fig = create_textbox_figure(row, width_inches=16, height_inches=2.5, accent_color=color1)
            
            # Step 3 & 4: Combine and save
            filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)
        
        elif 'BCE With Logits' in loss_name or ('Binary Cross-Entropy' in loss_name and 'Logits' in loss_name):
            # BCE With Logits: same as BCE but with logits (before sigmoid)
            logits = np.linspace(-5, 5, 1000)
            y_true = 1
            # Apply sigmoid to get probabilities for visualization
            p = 1 / (1 + np.exp(-logits))
            loss_values = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            
            fig, ax1, ax2 = create_beautiful_figure_2subplots((16, 10))
            color1 = COLOR_PALETTES['classification'][2]
            color2 = COLOR_PALETTES['classification'][3]
            
            # y_true = 1
            ax1.fill_between(logits, loss_values, alpha=0.2, color=color1, zorder=1)
            ax1.plot(logits, loss_values, linewidth=3.5, label='y_true = 1', color=color1, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            style_axes(ax1, f'{loss_name} (y_true = 1)', 'Logits (before sigmoid)', 'Loss', color1)
            ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            ax1.get_legend().get_frame().set_facecolor('#FFFFFF')
            
            # y_true = 0
            y_true = 0
            loss_values = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            ax2.fill_between(logits, loss_values, alpha=0.2, color=color2, zorder=1)
            ax2.plot(logits, loss_values, linewidth=3.5, label='y_true = 0', color=color2, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            style_axes(ax2, f'{loss_name} (y_true = 0)', 'Logits (before sigmoid)', 'Loss', color2)
            ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            ax2.get_legend().get_frame().set_facecolor('#FFFFFF')
            
            plt.suptitle(f'{loss_name} Loss Function', fontsize=18, fontweight='700', color='#2C3E50', y=0.98)
            
            # Step 2: Create textbox figure separately
            textbox_fig = create_textbox_figure(row, width_inches=16, height_inches=2.5, accent_color=color1)
            
            # Step 3 & 4: Combine and save
            filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)
        
        elif 'Sparse Cross-Entropy' in loss_name:
            # Sparse Cross-Entropy: same as Cross-Entropy but with integer labels
            p_true = np.linspace(0.001, 0.999, 1000)
            loss_values = -np.log(p_true)
            
            fig, ax = create_beautiful_figure((12, 10))
            color = COLOR_PALETTES['classification'][idx % len(COLOR_PALETTES['classification'])]
            ax.fill_between(p_true, loss_values, alpha=0.2, color=color, zorder=1)
            ax.plot(p_true, loss_values, linewidth=3.5, label='Sparse CE Loss', color=color, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            style_axes(ax, f'{loss_name} Loss Function', 'Predicted Probability for True Class', 'Loss', color)
            legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            legend.get_frame().set_facecolor('#FFFFFF')
            # Step 2: Create textbox figure separately
            textbox_fig = create_textbox_figure(row, width_inches=12, height_inches=2.5, accent_color=color)
            
            # Step 3 & 4: Combine and save
            filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)
        
        elif 'Cross-Entropy' in loss_name and 'Sparse' not in loss_name and 'Binary' not in loss_name:
            # Multi-class cross-entropy: assume 3 classes, true class = 0
            num_classes = 3
            p_true = np.linspace(0.001, 0.999, 1000)
            # For simplicity, show loss when true class probability varies
            loss_values = -np.log(p_true)
            
            fig, ax = create_beautiful_figure((12, 10))
            color = COLOR_PALETTES['classification'][idx % len(COLOR_PALETTES['classification'])]
            ax.fill_between(p_true, loss_values, alpha=0.2, color=color, zorder=1)
            ax.plot(p_true, loss_values, linewidth=3.5, label='CE Loss', color=color, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            style_axes(ax, f'{loss_name} Loss Function', 'Predicted Probability for True Class', 'Loss', color)
            legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            legend.get_frame().set_facecolor('#FFFFFF')
            # Step 2: Create textbox figure separately
            textbox_fig = create_textbox_figure(row, width_inches=12, height_inches=2.5, accent_color=color)
            
            # Step 3 & 4: Combine and save
            filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)
        
        elif 'Focal' in loss_name:
            # Focal loss: -α(1-p)^γ log(p) for y_true=1
            alpha = 0.25
            gamma = 2.0
            p = np.linspace(0.001, 0.999, 1000)
            loss_values = -alpha * ((1 - p) ** gamma) * np.log(p)
            
            fig, ax = create_beautiful_figure((12, 10))
            color = COLOR_PALETTES['classification'][idx % len(COLOR_PALETTES['classification'])]
            ax.fill_between(p, loss_values, alpha=0.2, color=color, zorder=1)
            ax.plot(p, loss_values, linewidth=3.5, label=f'Focal Loss (α={alpha}, γ={gamma})', color=color, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            style_axes(ax, f'{loss_name} Loss Function', 'Predicted Probability p (y_true=1)', 'Loss', color)
            legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            legend.get_frame().set_facecolor('#FFFFFF')
            # Step 2: Create textbox figure separately
            textbox_fig = create_textbox_figure(row, width_inches=12, height_inches=2.5, accent_color=color)
            
            # Step 3 & 4: Combine and save
            filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)

def visualize_hinge_loss(df, output_dir):
    """Visualize Hinge Loss"""
    hinge_row = df[df['name of loss function'].str.contains('Hinge', case=False, na=False)]
    if hinge_row.empty:
        return
    
    loss_name = hinge_row.iloc[0]['name of loss function']
    print(f"Visualizing: {loss_name}")
    
    # Hinge loss: max(0, 1 - y * f(x))
    # Assume y = 1, f(x) ranges from -2 to 2
    fx = np.linspace(-2, 2, 1000)
    y = 1
    loss_values = np.maximum(0, 1 - y * fx)
    
    fig, ax = create_beautiful_figure((12, 10))
    color = COLOR_PALETTES['other'][0]
    ax.fill_between(fx, loss_values, alpha=0.2, color=color, zorder=1)
    ax.plot(fx, loss_values, linewidth=3.5, label='Hinge Loss (y=1)', color=color, 
            zorder=2, antialiased=True, solid_capstyle='round')
    ax.axvline(x=1, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7, label='Margin boundary', zorder=3)
    style_axes(ax, f'{loss_name} Loss Function', 'f(x)', 'Loss', color)
    legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
    legend.get_frame().set_facecolor('#FFFFFF')
    
    # Step 2: Create textbox figure separately
    textbox_fig = create_textbox_figure(hinge_row.iloc[0], width_inches=12, height_inches=2.5, accent_color=color)
    
    # Step 3 & 4: Combine and save
    filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)

def visualize_smooth_l1_loss(df, output_dir):
    """Visualize Smooth L1 Loss"""
    smooth_l1_row = df[df['name of loss function'].str.contains('Smooth L1', case=False, na=False)]
    if smooth_l1_row.empty:
        return
    
    loss_name = smooth_l1_row.iloc[0]['name of loss function']
    print(f"Visualizing: {loss_name}")
    
    error = np.linspace(-3, 3, 1000)
    beta = 1.0
    abs_error = np.abs(error)
    loss_values = np.where(abs_error < beta,
                          0.5 * error ** 2 / beta,
                          abs_error - 0.5 * beta)
    
    fig, ax = create_beautiful_figure((12, 10))
    color = COLOR_PALETTES['detection'][0]
    ax.fill_between(error, loss_values, alpha=0.2, color=color, zorder=1)
    ax.plot(error, loss_values, linewidth=3.5, label='Smooth L1 Loss', color=color, 
            zorder=2, antialiased=True, solid_capstyle='round')
    style_axes(ax, f'{loss_name} Loss Function', 'Error (y_pred - y_true)', 'Loss', color)
    legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
    legend.get_frame().set_facecolor('#FFFFFF')
    
    # Step 2: Create textbox figure separately
    textbox_fig = create_textbox_figure(smooth_l1_row.iloc[0], width_inches=12, height_inches=2.5, accent_color=color)
    
    # Step 3 & 4: Combine and save
    filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)

def visualize_segmentation_losses(df, output_dir):
    """Visualize segmentation loss functions"""
    segmentation_losses = df[df['what it is used for'].str.contains('Segmentation', case=False, na=False)]
    
    for idx, row in segmentation_losses.iterrows():
        loss_name = row['name of loss function']
        print(f"Visualizing: {loss_name}")
        
        # Create overlap scenarios (intersection over union)
        # p: predicted probability, y: ground truth (binary)
        # For simplicity, visualize as function of overlap ratio
        overlap_ratio = np.linspace(0.01, 0.99, 1000)
        
        if 'Dice' in loss_name:
            # Dice Loss: 1 - 2|A∩B| / (|A|+|B|)
            # Simplified: assuming |A| = |B|, then Dice = 2|A∩B| / 2|A| = overlap
            dice_coeff = overlap_ratio
            loss_values = 1 - dice_coeff
        
        elif 'IoU' in loss_name:
            # IoU Loss: 1 - IoU
            iou = overlap_ratio
            loss_values = 1 - iou
        
        elif 'Tversky' in loss_name:
            # Tversky Loss: 1 - TP / (TP + αFP + βFN)
            # Simplified visualization
            alpha = 0.7
            beta = 0.3
            tp = overlap_ratio
            fp = 1 - overlap_ratio
            fn = 1 - overlap_ratio
            tversky_coeff = tp / (tp + alpha * fp + beta * fn)
            loss_values = 1 - tversky_coeff
        
        else:
            continue
        
        fig, ax = create_beautiful_figure((12, 10))
        color = COLOR_PALETTES['segmentation'][idx % len(COLOR_PALETTES['segmentation'])]
        ax.fill_between(overlap_ratio, loss_values, alpha=0.2, color=color, zorder=1)
        ax.plot(overlap_ratio, loss_values, linewidth=3.5, label=loss_name, color=color, 
                zorder=2, antialiased=True, solid_capstyle='round')
        style_axes(ax, f'{loss_name} Loss Function', 'Overlap Ratio (Intersection/Union)', 'Loss', color)
        legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
        legend.get_frame().set_facecolor('#FFFFFF')
        
        # Step 2: Create textbox figure separately
        textbox_fig = create_textbox_figure(row, width_inches=12, height_inches=2.5, accent_color=color)
        
        # Step 3 & 4: Combine and save
        filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)

def visualize_kl_divergence(df, output_dir):
    """Visualize KL Divergence"""
    kl_row = df[df['name of loss function'].str.contains('KL', case=False, na=False)]
    if kl_row.empty:
        return
    
    loss_name = kl_row.iloc[0]['name of loss function']
    print(f"Visualizing: {loss_name}")
    
    # KL Divergence: Σ p log(p/q)
    # For simplicity, visualize when q varies while p is fixed
    q = np.linspace(0.01, 0.99, 1000)
    p = 0.5  # Fixed true distribution
    kl_values = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    
    fig, ax = create_beautiful_figure((12, 10))
    color = COLOR_PALETTES['other'][1]
    ax.fill_between(q, kl_values, alpha=0.2, color=color, zorder=1)
    ax.plot(q, kl_values, linewidth=3.5, label=f'KL Divergence (p={p})', color=color, 
            zorder=2, antialiased=True, solid_capstyle='round')
    ax.axvline(x=p, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7, label='p (true distribution)', zorder=3)
    style_axes(ax, f'{loss_name} Loss Function', 'q (Predicted Distribution)', 'KL Divergence', color)
    legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
    legend.get_frame().set_facecolor('#FFFFFF')
    
    # Step 2: Create textbox figure separately
    textbox_fig = create_textbox_figure(kl_row.iloc[0], width_inches=12, height_inches=2.5, accent_color=color)
    
    # Step 3 & 4: Combine and save
    filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)

def visualize_metric_learning_losses(df, output_dir):
    """Visualize metric learning loss functions"""
    metric_losses = df[df['what it is used for'].str.contains('Metric learning', case=False, na=False)]
    
    for idx, row in metric_losses.iterrows():
        loss_name = row['name of loss function']
        print(f"Visualizing: {loss_name}")
        
        if 'Triplet' in loss_name:
            # Triplet Loss: max(0, d(a,p) - d(a,n) + margin)
            # Visualize as function of distance difference
            margin = 1.0
            distance_diff = np.linspace(-2, 3, 1000)  # d(a,p) - d(a,n)
            loss_values = np.maximum(0, distance_diff + margin)
            
            fig, ax = create_beautiful_figure((12, 10))
            color = COLOR_PALETTES['metric'][idx % len(COLOR_PALETTES['metric'])]
            ax.fill_between(distance_diff, loss_values, alpha=0.2, color=color, zorder=1)
            ax.plot(distance_diff, loss_values, linewidth=3.5, label=f'Triplet Loss (margin={margin})', color=color, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            ax.axvline(x=-margin, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7, label='Margin boundary', zorder=3)
            style_axes(ax, f'{loss_name} Loss Function', 'Distance Difference (d(a,p) - d(a,n))', 'Loss', color)
            legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            legend.get_frame().set_facecolor('#FFFFFF')
            
            # Step 2: Create textbox figure separately
            textbox_fig = create_textbox_figure(row, width_inches=12, height_inches=2.5, accent_color=color)
            
            # Step 3 & 4: Combine and save
            filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)
        
        elif 'Contrastive' in loss_name:
            # Contrastive Loss: y*d² + (1-y)*max(0, margin-d)²
            margin = 1.0
            distance = np.linspace(0, 3, 1000)
            
            # y = 1 (same class)
            loss_same = distance ** 2
            # y = 0 (different class)
            loss_diff = np.maximum(0, margin - distance) ** 2
            
            fig, ax1, ax2 = create_beautiful_figure_2subplots((16, 10))
            color1 = COLOR_PALETTES['metric'][0]
            color2 = COLOR_PALETTES['metric'][1]
            
            # Same class
            ax1.fill_between(distance, loss_same, alpha=0.2, color=color1, zorder=1)
            ax1.plot(distance, loss_same, linewidth=3.5, label='Same class (y=1)', color=color1, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            style_axes(ax1, f'{loss_name} (Same Class)', 'Distance d', 'Loss', color1)
            ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            ax1.get_legend().get_frame().set_facecolor('#FFFFFF')
            
            # Different class
            ax2.fill_between(distance, loss_diff, alpha=0.2, color=color2, zorder=1)
            ax2.plot(distance, loss_diff, linewidth=3.5, label='Different class (y=0)', color=color2, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            ax2.axvline(x=margin, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7, label='Margin', zorder=3)
            style_axes(ax2, f'{loss_name} (Different Class)', 'Distance d', 'Loss', color2)
            ax2.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            ax2.get_legend().get_frame().set_facecolor('#FFFFFF')
            
            plt.suptitle(f'{loss_name} Loss Function', fontsize=18, fontweight='700', color='#2C3E50', y=0.98)
            
            # Step 2: Create textbox figure separately
            textbox_fig = create_textbox_figure(row, width_inches=16, height_inches=2.5, accent_color=color1)
            
            # Step 3 & 4: Combine and save
            filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)
            continue
        
        # Step 2: Create textbox figure separately
        textbox_fig = create_textbox_figure(row, width_inches=12, height_inches=2.5, accent_color=color)
        
        # Step 3 & 4: Combine and save
        filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)

def visualize_other_losses(df, output_dir):
    """Visualize other specialized loss functions"""
    # InfoNCE Loss
    infonce_row = df[df['name of loss function'].str.contains('InfoNCE', case=False, na=False)]
    if not infonce_row.empty:
        loss_name = infonce_row.iloc[0]['name of loss function']
        print(f"Visualizing: {loss_name}")
        
        # InfoNCE: -log(exp(sim(pos))/Σexp(sim(all)))
        # Visualize as function of positive similarity
        sim_pos = np.linspace(-2, 2, 1000)
        sim_neg = np.array([-1, 0, 1])  # Negative similarities
        # Simplified: assume fixed negative similarities
        loss_values = -np.log(np.exp(sim_pos) / (np.exp(sim_pos) + np.sum(np.exp(sim_neg))))
        
        fig, ax = create_beautiful_figure((12, 10))
        color = COLOR_PALETTES['other'][2]
        ax.fill_between(sim_pos, loss_values, alpha=0.2, color=color, zorder=1)
        ax.plot(sim_pos, loss_values, linewidth=3.5, label='InfoNCE Loss', color=color, 
                zorder=2, antialiased=True, solid_capstyle='round')
        style_axes(ax, f'{loss_name} Loss Function', 'Positive Similarity', 'Loss', color)
        legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
        legend.get_frame().set_facecolor('#FFFFFF')
        
        # Step 2: Create textbox figure separately
        textbox_fig = create_textbox_figure(infonce_row.iloc[0], width_inches=12, height_inches=2.5, accent_color=color)
        
        # Step 3 & 4: Combine and save
        filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)
    
    # Object Detection IoU losses (GIoU, DIoU, CIoU)
    iou_losses = df[df['what it is used for'].str.contains('Object detection', case=False, na=False)]
    for idx, row in iou_losses.iterrows():
        loss_name = row['name of loss function']
        if 'IoU' in loss_name:
            print(f"Visualizing: {loss_name}")
            
            # Visualize as function of IoU
            iou = np.linspace(0.01, 0.99, 1000)
            loss_values = 1 - iou  # Simplified visualization
            
            fig, ax = create_beautiful_figure((12, 10))
            color = COLOR_PALETTES['detection'][idx % len(COLOR_PALETTES['detection'])]
            ax.fill_between(iou, loss_values, alpha=0.2, color=color, zorder=1)
            ax.plot(iou, loss_values, linewidth=3.5, label=loss_name, color=color, 
                    zorder=2, antialiased=True, solid_capstyle='round')
            style_axes(ax, f'{loss_name} Loss Function', 'IoU (Intersection over Union)', 'Loss', color)
            legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, framealpha=0.95, edgecolor='#CCCCCC')
            legend.get_frame().set_facecolor('#FFFFFF')
            # Step 2: Create textbox figure separately
            textbox_fig = create_textbox_figure(row, width_inches=12, height_inches=2.5, accent_color=color)
            
            # Step 3 & 4: Combine and save
            filename = loss_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            combine_graph_and_textbox(fig, textbox_fig, output_dir / f'{filename}.png', dpi=300)

def main():
    """Main function to create all visualizations"""
    excel_file = 'loss_functions_table.xlsx'
    
    print("=" * 60)
    print("Loss Functions Visualizer")
    print("=" * 60)
    
    # Read Excel file
    print(f"\nReading Excel file: {excel_file}")
    df = pd.read_excel(excel_file)
    print(f"Found {len(df)} loss functions\n")
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}\n")
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    visualize_regression_losses(df, output_dir)
    visualize_classification_losses(df, output_dir)
    visualize_hinge_loss(df, output_dir)
    visualize_smooth_l1_loss(df, output_dir)
    visualize_segmentation_losses(df, output_dir)
    visualize_kl_divergence(df, output_dir)
    visualize_metric_learning_losses(df, output_dir)
    visualize_other_losses(df, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Visualization complete! Check '{output_dir}' directory for all plots.")
    print("=" * 60)

if __name__ == '__main__':
    main()

