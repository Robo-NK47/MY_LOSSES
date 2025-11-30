"""
Interactive Loss Functions Visualizer using Dash
Free and open-source framework by Plotly!
"""

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
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
    """Convert formula text to LaTeX format with proper math delimiters"""
    if pd.isna(formula) or formula == 'N/A':
        return 'N/A'
    
    formula = str(formula)
    import re
    
    # Fix encoding issues
    replacements = {
        'Î£': 'Σ', 'Î´': 'δ', 'Î±': 'α', 'Î²': 'β', 'Î³': 'γ',
        'Â²': '²', 'âˆ©': '∩', 'âˆª': '∪', 'âˆš': '√',
    }
    for old, new in replacements.items():
        formula = formula.replace(old, new)
    
    # Convert Greek letters to LaTeX
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
    formula = re.sub(r'Σ', r'\\sum', formula)
    
    # Convert superscripts (², ³, etc.)
    formula = re.sub(r'²', r'^2', formula)
    formula = re.sub(r'³', r'^3', formula)
    
    # Convert square roots
    formula = re.sub(r'sqrt\s*\(([^)]+)\)', r'\\sqrt{\1}', formula)
    formula = re.sub(r'√\s*\(([^)]+)\)', r'\\sqrt{\1}', formula)
    formula = re.sub(r'√([^\s\(\)]+)', r'\\sqrt{\1}', formula)
    
    # Convert fractions (handle both a/b and (a)/(b) formats)
    formula = re.sub(r'\(([^)]+)\)\s*/\s*\(([^)]+)\)', r'\\frac{\1}{\2}', formula)
    formula = re.sub(r'(\d+|[a-zA-Z_]+)\s*/\s*(\d+|[a-zA-Z_]+)', r'\\frac{\1}{\2}', formula)
    
    # Convert common functions
    formula = re.sub(r'\blog\s*\(', r'\\log(', formula)
    formula = re.sub(r'\bexp\s*\(', r'\\exp(', formula)
    formula = re.sub(r'\bmax\s*\(', r'\\max(', formula)
    formula = re.sub(r'\bmin\s*\(', r'\\min(', formula)
    formula = re.sub(r'\bsin\s*\(', r'\\sin(', formula)
    formula = re.sub(r'\bcos\s*\(', r'\\cos(', formula)
    formula = re.sub(r'\btan\s*\(', r'\\tan(', formula)
    formula = re.sub(r'\bln\s*\(', r'\\ln(', formula)
    
    # Convert subscripts and superscripts for variables
    formula = re.sub(r'y_pred', r'y_{pred}', formula)
    formula = re.sub(r'y_true', r'y_{true}', formula)
    formula = re.sub(r'y_(\w+)', r'y_{\1}', formula)
    
    # Convert sum notation with proper limits
    formula = re.sub(r'\\sum\s*\(([^)]+)\)', r'\\sum \1', formula)
    formula = re.sub(r'\\sum\s+([a-z])\s*=\s*1\s+to\s+n', r'\\sum_{\1=1}^{n}', formula, flags=re.IGNORECASE)
    
    # Convert multiplication signs
    formula = formula.replace('×', r' \times ')
    formula = formula.replace('·', r' \cdot ')
    
    # Convert set operations
    formula = formula.replace('∩', r' \cap ')
    formula = formula.replace('∪', r' \cup ')
    
    # Clean up spacing
    formula = re.sub(r'\s+', ' ', formula)
    formula = formula.strip()
    
    # Wrap in display math delimiters for proper rendering
    return f'$${formula}$$'

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
        autosize=False,
        showlegend=True,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='#FAFAFA'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# Load data
df = load_loss_functions()
if df is None:
    print("Error: Could not load loss_functions_table.xlsx")
    exit(1)

loss_names = df['name of loss function'].tolist()

# Initialize Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Loss Functions Visualizer"

# Add MathJax for LaTeX rendering
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>
            window.MathJax = {
                tex: {
                    inlineMath: [['$','$'], ['\\\\(','\\\\)']],
                    displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
                    processEscapes: true,
                    processEnvironments: true
                },
                options: {
                    skipHtmlTags: ['script', 'style', 'noscript', 'textarea', 'pre']
                }
            };
        </script>
        <script type="text/javascript" id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📊 Interactive Loss Functions Visualizer", className="text-center mb-4"),
            html.P("Explore different loss functions used in machine learning with interactive visualizations!", 
                   className="text-center text-muted mb-4"),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Loss Function", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='loss-dropdown',
                options=[{'label': name, 'value': name} for name in loss_names],
                value=loss_names[0],
                clearable=False,
                className="mb-3"
            ),
            html.Div(
                dcc.Graph(id='loss-plot', config={'displayModeBar': True}, responsive=False),
                style={'height': '500px', 'overflow': 'hidden'}
            )
        ], width=8),
        
        dbc.Col([
            html.H3("📋 Information", className="mb-3"),
            
            html.H5("📐 Formula", className="mt-3"),
            html.Div(id='formula-display', className="mb-3 p-3 bg-light rounded", style={'textAlign': 'center'}),
            
            html.H5("⚙️ What it does", className="mt-3"),
            dbc.Alert(id='what-it-does', color="info", className="mb-3"),
            
            html.H5("🎯 Used for", className="mt-3"),
            dbc.Alert(id='use-case', color="info", className="mb-3"),
            
            html.H5("💡 When to use", className="mt-3"),
            dbc.Alert(id='when-to-use', color="info", className="mb-3"),
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.H4("💡 Tips", className="mt-3"),
            html.Ul([
                html.Li("Hover over the graph to see exact values"),
                html.Li("Zoom and pan to explore different regions"),
                html.Li("Download the graph using the toolbar"),
                html.Li("Share this tool with your network! 🚀")
            ])
        ], width=12)
    ])
], fluid=True, className="mt-4")

# Callback to update all components when loss function changes
@app.callback(
    [Output('loss-plot', 'figure'),
     Output('formula-display', 'children'),
     Output('what-it-does', 'children'),
     Output('use-case', 'children'),
     Output('when-to-use', 'children')],
    [Input('loss-dropdown', 'value')]
)
def update_visualization(loss_name):
    """Update visualization and info when loss function changes"""
    if loss_name is None:
        loss_name = loss_names[0]
    
    selected_row = df[df['name of loss function'] == loss_name].iloc[0]
    formula = str(selected_row['formula of loss function']) if pd.notna(selected_row['formula of loss function']) else 'N/A'
    use_case = str(selected_row['what it is used for']) if pd.notna(selected_row['what it is used for']) else 'Unknown'
    
    what_it_does = get_info(loss_name, WHAT_IT_DOES)
    when_to_use = get_info(loss_name, WHEN_TO_USE)
    
    formula_latex = convert_formula_to_latex(formula)
    
    fig = create_visualization(loss_name, use_case)
    
    # Format formula for display using Markdown which supports LaTeX
    if formula_latex != 'N/A':
        # dcc.Markdown supports LaTeX math rendering
        formula_html = dcc.Markdown(
            formula_latex,
            mathjax=True,
            style={'textAlign': 'center'}
        )
    else:
        formula_html = html.Div('N/A', className="text-muted")
    
    return fig, formula_html, what_it_does, use_case, when_to_use

if __name__ == '__main__':
    import os
    # Use debug mode only in development
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))

