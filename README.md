# 📊 Interactive Loss Functions Visualizer

An interactive web application to explore machine learning loss functions with visualizations, mathematical formulas, and usage guidelines.

## Features

- 🎯 **30+ Loss Functions** - Comprehensive collection of regression, classification, and specialized loss functions
- 📈 **Interactive Visualizations** - Plotly-powered graphs showing how each loss function behaves
- 📐 **Mathematical Formulas** - Properly rendered LaTeX formulas for each loss function
- 💡 **Usage Guidelines** - When and why to use each loss function
- 🎨 **Modern UI** - Clean, responsive interface built with Dash and Bootstrap

## Quick Start

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Loss_functions
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   python my_losses.py
   ```

5. **Open in browser**: http://localhost:8050

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on deploying to:
- Render (Recommended)
- Railway
- PythonAnywhere
- Fly.io

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Files

- `my_losses.py` - Main application file
- `loss_functions_table.xlsx` - Data source with loss function information
- `requirements.txt` - Python dependencies
- `Procfile` - For Render deployment
- `render.yaml` - Render configuration

## License

Free to use and modify.

## Author

Created for educational purposes - perfect for sharing on LinkedIn! 🚀

