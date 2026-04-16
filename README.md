# Machine Learning Project Environment

This is a Python environment configured for machine learning projects.

## ⚠️ Python 3.14 Compatibility Notice

**Current Python Version:** 3.14.3

Due to the recent release of Python 3.14, some machine learning libraries may not have compatible versions yet. Here's the current status:

### ✅ Successfully Installed
- **numpy** (2.4.4) - Core numerical computing
- **pandas** (3.0.2) - Data manipulation and analysis  
- **scipy** (1.17.1) - Scientific computing
- **matplotlib** (3.10.8) - Basic plotting
- **seaborn** (0.13.2) - Statistical data visualization
- **plotly** (6.7.0) - Interactive visualizations
- **jupyter** (1.1.1) - Jupyter notebooks
- **ipython** (9.12.0) - Enhanced Python shell
- **notebook** (7.5.5) - Jupyter notebook interface
- **joblib** (1.5.3) - Lightweight pipelining

### ⚠️ Not Currently Compatible with Python 3.14
- **scikit-learn** - Requires Python < 3.13
- **tensorflow** - No compatible version available
- **pytorch** - May not have 3.14 wheels yet
- **xgboost** - May require compilation
- **lightgbm** - May require compilation

## Setup Instructions

### 1. Activate Virtual Environment

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 2. Install Available Dependencies

```bash
pip install -r requirements.txt
```

### Alternative Solutions

1. **Use Python 3.12 or 3.11** for full compatibility with all ML libraries
2. **Install from source** for libraries that don't have wheels:
   ```bash
   pip install --no-binary :all: scikit-learn
   ```
3. **Use conda/mamba** which may have better compatibility:
   ```bash
   conda install scikit-learn tensorflow
   ```

## Project Structure
```
project/
├── venv/                 # Virtual environment (ignored by git)
├── requirements.txt      # Dependencies
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── data/                # Data directory (create as needed)
├── notebooks/           # Jupyter notebooks
├── src/                 # Source code
└── models/              # Trained models
```

## Usage Examples

### Basic Data Analysis (Works with current setup)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/your_data.csv')
print(df.head())

# Basic visualization
df.plot(kind='hist')
plt.show()
```

## Best Practices

1. Always activate the virtual environment before working
2. Consider using Python 3.12 for full ML library compatibility
3. Install new packages with `pip install` and update requirements.txt
4. Use `pip freeze > requirements.txt` to update dependencies
5. Keep data in the `data/` directory