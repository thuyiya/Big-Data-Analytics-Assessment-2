# Big Data Assessment

## Setup Instructions

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
.\venv\Scripts\activate  # On Windows
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python bigdata_assessment.py
```

## Requirements
- Python 3.8+
- Java 8+ (required for PySpark)

## Dependencies
- PySpark
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NumPy

## Running the Project

1. Make sure your virtual environment is activated (see step 2 above)

2. Run the analysis script:
```bash
python bigdata_assessment.py
```

The script will:
- Process the customer purchase data
- Generate various statistics and visualizations
- Create several output files including:
  - purchase_amount_histogram.png
  - total_purchases_boxplot.png
  - purchase_vs_spending.png
  - top_customers.png

## Results
After running the script, you can find the generated visualization files in the same directory.
