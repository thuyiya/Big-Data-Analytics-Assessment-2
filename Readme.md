# Big Data Assessment

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install required packages:
```bash
pip install pyspark pandas numpy matplotlib seaborn
```

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