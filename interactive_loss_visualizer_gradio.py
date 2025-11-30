"""
Interactive Loss Functions Visualizer using Gradio
Free hosting available on Hugging Face Spaces!
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Load data
def load_loss_functions():
    """Load loss functions data from Excel"""
    try:
        df = pd.read_excel('loss_functions_table.xlsx')
        return df
    except FileNotFoundError:
        return None

# Dictionary for "what it does" information
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

def get_info(loss_name, info_dict):
    """Get information from dictionary with fallback"""
    loss_name_lower = loss_name.lower()
    if loss_name in info_dict:
        return info_dict[loss_name]
    matches = [(key, info_dict[key]) for key in info_dict if key.lower() in loss_name_lower]
    if matches:
        matches.sort(key=lambda x: len(x[0]), reverse=True)
        return matches[0][1]
    return 'N/A'

def convert_formula_to_latex(formula):
    """Convert formula text to LaTeX format"""
    if pd.isna(formula) or formula == 'N/A':
        return 'N/A'
    
    formula = str(formula)
    import re
    
    # Fix encoding
    replacements = {
        'Î£': 'Σ', 'Î´': 'δ', 'Î±': 'α', 'Î²': 'β', 'Î³': 'γ',
        'Â²': '²', 'âˆ©': '∩', 'âˆª': '∪', 'âˆš': '√',
    }
    for old, new in replacements.items():
        formula = formula.replace(old, new)
    
    # Convert to LaTeX
    formula = formula.replace('²', '^2')
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
    
    formula = re.sub(r'sqrt\s*\(([^)]+)\)', r'\\sqrt{\1}', formula)
    formula = re.sub(r'√\s*\(([^)]+)\)', r'\\sqrt{\1}', formula)
    formula = re.sub(r'(\d+)/n\s*Σ', r'\1/n \\sum', formula)
    formula = re.sub(r'\s+Σ\s+', r' \\sum ', formula)
    formula = re.sub(r'Σ\s*\(', r'\\sum(', formula)
    formula = re.sub(r'Σ([^A-Za-z])', r'\\sum\1', formula)
    
    formula = re.sub(r'\blog\s*\(', r'\\log(', formula)
    formula = re.sub(r'\bexp\s*\(', r'\\exp(', formula)
    formula = re.sub(r'\bmax\s*\(', r'\\max(', formula)
    formula = re.sub(r'\bmin\s*\(', r'\\min(', formula)
    formula = re.sub(r'y_pred', r'y_{pred}', formula)
    formula = re.sub(r'y_true', r'y_{true}', formula)
    formula = re.sub(r'(\d+)/n', r'\\frac{\1}{n}', formula)
    
    formula = formula.replace('  ', ' ').strip()
    return formula

def create_visualization(loss_name, use_case):
    """Create interactive visualization"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    color = colors[hash(loss_name) % len(colors)]
    
    if 'Regression' in use_case:
        error = np.linspace(-5, 5, 1000)
        y_true = 0
        
        if 'MSE' in loss_name and 'RMSE' not in loss_name:
            loss_values = (error - y_true) ** 2
        elif 'MAE' in loss_name:
            loss_values = np.abs(error - y_true)
        elif 'RMSE' in loss_name:
            loss_values = np.sqrt((error - y_true) ** 2)
        elif 'Huber' in loss_name:
            delta = 1.0
            abs_error = np.abs(error - y_true)
            loss_values = np.where(abs_error <= delta, 0.5 * abs_error ** 2, delta * (abs_error - 0.5 * delta))
        elif 'Log-Cosh' in loss_name:
            loss_values = np.log(np.cosh(error - y_true))
        else:
            loss_values = (error - y_true) ** 2
        
        x_data = error
        x_label = 'Error (Predicted - True)'
        
    elif 'Classification' in use_case or 'Binary' in use_case:
        p = np.linspace(0.001, 0.999, 1000)
        
        if 'BCE' in loss_name and 'Logits' not in loss_name:
            y_true = 1
            loss_values = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            x_label = 'Predicted Probability'
            x_data = p
        elif 'BCE With Logits' in loss_name:
            logits = np.linspace(-5, 5, 1000)
            p = 1 / (1 + np.exp(-logits))
            y_true = 1
            loss_values = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            x_label = 'Logits (before sigmoid)'
            x_data = logits
        elif 'Cross-Entropy' in loss_name and 'Sparse' not in loss_name:
            loss_values = -np.log(p)
            x_label = 'Predicted Probability'
            x_data = p
        elif 'Focal' in loss_name:
            alpha = 0.25
            gamma = 2.0
            loss_values = -alpha * (1 - p) ** gamma * np.log(p)
            x_label = 'Predicted Probability'
            x_data = p
        else:
            loss_values = -np.log(p)
            x_label = 'Predicted Probability'
            x_data = p
            
    elif 'Segmentation' in use_case:
        overlap = np.linspace(0.01, 0.99, 1000)
        
        if 'Dice' in loss_name:
            loss_values = 1 - overlap
        elif 'IoU' in loss_name:
            loss_values = 1 - overlap
        elif 'Tversky' in loss_name:
            alpha = 0.7
            beta = 0.3
            tp = overlap
            fp = 1 - overlap
            fn = 1 - overlap
            tversky_coeff = tp / (tp + alpha * fp + beta * fn)
            loss_values = 1 - tversky_coeff
        else:
            loss_values = 1 - overlap
        
        x_label = 'Overlap Ratio'
        x_data = overlap
        
    elif 'Metric learning' in use_case:
        if 'Triplet' in loss_name:
            margin = 1.0
            distance_diff = np.linspace(-2, 3, 1000)
            loss_values = np.maximum(0, distance_diff + margin)
            x_label = 'Distance Difference (d(a,p) - d(a,n))'
            x_data = distance_diff
        elif 'Contrastive' in loss_name:
            distance = np.linspace(0, 5, 1000)
            margin = 1.0
            loss_values = np.where(distance < margin, distance ** 2, margin ** 2)
            x_label = 'Distance'
            x_data = distance
        else:
            distance_diff = np.linspace(-2, 3, 1000)
            loss_values = np.maximum(0, distance_diff + 1.0)
            x_label = 'Distance Difference'
            x_data = distance_diff
    else:
        x = np.linspace(-3, 3, 1000)
        loss_values = x ** 2
        x_label = 'Input'
        x_data = x
    
    def hex_to_rgba(hex_color, alpha=0.2):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=loss_values,
        fill='tozeroy',
        mode='lines',
        name=loss_name,
        line=dict(color=color, width=3),
        fillcolor=hex_to_rgba(color, 0.2),
        hovertemplate=f'<b>{loss_name}</b><br>{x_label}: %{{x:.4f}}<br>Loss: %{{y:.4f}}<extra></extra>',
    ))
    
    if 'Triplet' in loss_name:
        margin = 1.0
        fig.add_vline(x=-margin, line_dash="dash", line_color="red", annotation_text="Margin")
    
    fig.update_layout(
        title=f'{loss_name} Loss Function',
        xaxis_title=x_label,
        yaxis_title='Loss',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='#FAFAFA'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def update_visualization(loss_name):
    """Update visualization and info when loss function changes"""
    df = load_loss_functions()
    if df is None:
        return None, "Error loading data", "N/A", "N/A", "N/A"
    
    selected_row = df[df['name of loss function'] == loss_name].iloc[0]
    formula = str(selected_row['formula of loss function']) if pd.notna(selected_row['formula of loss function']) else 'N/A'
    use_case = str(selected_row['what it is used for']) if pd.notna(selected_row['what it is used for']) else 'Unknown'
    
    what_it_does = get_info(loss_name, WHAT_IT_DOES)
    when_to_use = get_info(loss_name, WHEN_TO_USE)
    
    formula_latex = convert_formula_to_latex(formula)
    
    fig = create_visualization(loss_name, use_case)
    
    return fig, formula_latex, what_it_does, use_case, when_to_use

# Load data
df = load_loss_functions()
if df is None:
    print("Error: Could not load loss_functions_table.xlsx")
    exit(1)

loss_names = df['name of loss function'].tolist()

# Create Gradio interface
with gr.Blocks(title="Loss Functions Visualizer") as demo:
    gr.Markdown("# 📊 Interactive Loss Functions Visualizer")
    gr.Markdown("Explore different loss functions used in machine learning with interactive visualizations!")
    
    with gr.Row():
        with gr.Column(scale=2):
            loss_dropdown = gr.Dropdown(
                choices=loss_names,
                value=loss_names[0],
                label="Select Loss Function",
                interactive=True
            )
            
            plot_output = gr.Plot(label="Loss Function Visualization")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Information")
            
            formula_output = gr.Markdown(label="📐 Formula")
            what_it_does_output = gr.Textbox(
                label="⚙️ What it does",
                lines=4,
                interactive=False
            )
            use_case_output = gr.Textbox(
                label="🎯 Used for",
                lines=2,
                interactive=False
            )
            when_to_use_output = gr.Textbox(
                label="💡 When to use",
                lines=3,
                interactive=False
            )
    
    # Update on change
    loss_dropdown.change(
        fn=update_visualization,
        inputs=[loss_dropdown],
        outputs=[plot_output, formula_output, what_it_does_output, use_case_output, when_to_use_output]
    )
    
    # Initial load
    demo.load(
        fn=update_visualization,
        inputs=[loss_dropdown],
        outputs=[plot_output, formula_output, what_it_does_output, use_case_output, when_to_use_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("### 💡 Tips")
    gr.Markdown("""
    - **Hover** over the graph to see exact values
    - **Zoom** and **pan** to explore different regions
    - **Download** the graph using the toolbar
    - Share this tool with your network! 🚀
    """)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates a public link

