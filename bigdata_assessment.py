print("Big Data Analytics Assessment 2 - PySpark Implementation")

# Import necessary modules
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix 

# Create directory for saving visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# =============================================
# 1. Initial Setup and Data Loading
# =============================================
print("\nTask 1: Initial Setup and Data Loading")

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BigDataAnalyticsAssessment2") \
    .getOrCreate()

# Load the dataset
file_path = "customer_purchases.csv"
df1 = spark.read.csv(file_path, header=True, inferSchema=True)

# Describe the structure
print("DataFrame Schema:")
df1.printSchema()

print("\nFirst 5 rows:")
df1.show(5, truncate=False)

print("\nSummary Statistics:")
df1.describe().show()

spark.conf.set("spark.sql.debug.maxToStringFields", 100) 

# =============================================
# Enhanced Data Visualization
# =============================================
print("\nEnhanced Data Visualization Section")

# Convert to Pandas for visualization
df_pd = df1.limit(1000).toPandas()

# i. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_pd['Age'], bins=30, kde=True)
plt.title('Distribution of Customer Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('visualizations/age_distribution.png')
plt.close()
print("Saved age_distribution.png")

# ii. Gender Distribution
plt.figure(figsize=(8, 5))
df_pd['Gender'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('visualizations/gender_distribution.png')
plt.close()
print("Saved gender_distribution.png")

# iii. Income vs Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='Gender', data=df_pd)
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (£)')
plt.ylabel('Spending Score')
plt.savefig('visualizations/income_vs_spending.png')
plt.close()
print("Saved income_vs_spending.png")

# iv. Purchase Categories
plt.figure(figsize=(12, 6))
df_pd['PurchaseCategory'].value_counts().plot(kind='bar')
plt.title('Purchase Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('visualizations/purchase_categories.png')
plt.close()
print("Saved purchase_categories.png")

# v. Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df_pd.select_dtypes(include=[np.number]).columns
corr_matrix = df_pd[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()
print("Saved correlation_heatmap.png")

# =============================================
# 2.  Handling Missing Values
# =============================================

print("\nTask 2: Handling Missing Values")

# First check for missing/null/zero values
print("\nMissing/Null/Zero Value Analysis:")
print(f"Total rows: {df1.count()}")

# Check for nulls using proper column references
null_counts = {col_name: df1.filter(df1[col_name].isNull()).count() 
              for col_name in df1.columns}

print("\nNull counts per column:")
for col_name, count in null_counts.items():
    print(f"{col_name}: {count}")

# Check for zeros in numerical columns
numerical_cols = ['SpendingScore', 'TotalPurchases', 'PurchaseAmount', 'AnnualIncome']
zero_counts = {col_name: df1.filter(df1[col_name] == 0).count() 
              for col_name in numerical_cols}

print("\nZero counts in numerical columns:")
for col_name, count in zero_counts.items():
    print(f"{col_name}: {count}")

# Only proceed with replacement if needed
if any(null_counts.values()) or any(zero_counts.values()):
    print("\nFound missing/null/zero values - replacing with medians...")
    
    # Calculate medians using direct column references
    medians = df1.approxQuantile(["SpendingScore", "TotalPurchases"], [0.5], 0.01)
    spending_score_median = medians[0][0]
    total_purchases_median = medians[1][0]
    
    # Create new DataFrame with replacements
    from pyspark.sql.functions import when
    
    df2 = df1.withColumn(
        "SpendingScore", 
        when(df1["SpendingScore"].isNull() | (df1["SpendingScore"] == 0), spending_score_median)
        .otherwise(df1["SpendingScore"])
    ).withColumn(
        "TotalPurchases",
        when(df1["TotalPurchases"].isNull() | (df1["TotalPurchases"] == 0), total_purchases_median)
        .otherwise(df1["TotalPurchases"])
    )
    print("Replacement completed.")
else:
    print("\nNo missing/null/zero values found in SpendingScore or TotalPurchases.")
    df2 = df1  # Just create a copy if no replacement needed

# =============================================
# 3. Fixed Row Removal with Proper Imports
# =============================================
print("\nTask 3: Smart Row Removal")

# Make sure you have this import at the top of your script:
from pyspark.sql.functions import col, lit
from functools import reduce

# Define columns to check
critical_columns = ["Age", "AnnualIncome", "PurchaseAmount"]

# Check for problematic values first
has_issues = any(
    df2.where((col(c) == 0) | (col(c).isNull())).count() > 0
    for c in critical_columns
)

if has_issues:
    original_count = df2.count()
    
    # Build dynamic filter conditions only for problematic columns
    conditions = []
    for column in critical_columns:
        if df2.where((col(column) == 0) | (col(column).isNull())).count() > 0:
            conditions.append((col(column) != 0) & col(column).isNotNull())
    
    # Apply filters if needed
    if conditions:
        final_condition = reduce(lambda a, b: a & b, conditions)
        df3 = df2.where(final_condition)
        rows_removed = original_count - df3.count()
        print(f"Removed {rows_removed} rows with invalid values")
    else:
        df3 = df2
        print("No invalid values found in critical columns")
    
    print(f"Final count: {df3.count()} (Originally {original_count})")
else:
    df3 = df2
    print("No invalid values found - skipping row removal")

# =============================================
# 2,3 Steps - Post-Cleaning Visualizations
# =============================================
print("\nPost-Cleaning Visualizations")

df3_pd = df3.limit(1000).toPandas()

# i. Spending Score Distribution After Cleaning
plt.figure(figsize=(10, 6))
sns.histplot(df3_pd['SpendingScore'], bins=30, kde=True)
plt.title('Spending Score Distribution After Cleaning')
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.savefig('visualizations/spending_score_clean.png')
plt.close()
print("Saved spending_score_clean.png")

# ii. Purchase Amount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df3_pd['PurchaseAmount'], bins=30, kde=True)
plt.title('Purchase Amount Distribution')
plt.xlabel('Purchase Amount (£)')
plt.ylabel('Count')
plt.savefig('visualizations/purchase_amount_dist.png')
plt.close()
print("Saved purchase_amount_dist.png")

# iii. Age vs Purchase Amount
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='PurchaseAmount', hue='Gender', data=df3_pd)
plt.title('Age vs Purchase Amount')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (£)')
plt.savefig('visualizations/age_vs_purchase.png')
plt.close()
print("Saved age_vs_purchase.png")

# =============================================
# 4. Summary Statistics and Histogram
# =============================================
print("\nTask 4: Summary Statistics and Histogram")

if "BloodPressure" in df3.columns:
    stats = df3.select(
        min("BloodPressure").alias("min"),
        max("BloodPressure").alias("max"),
        mean("BloodPressure").alias("mean"),
        expr("percentile_approx(BloodPressure, 0.5)").alias("median"),
        variance("BloodPressure").alias("variance"),
        stddev("BloodPressure").alias("stddev")
    ).collect()[0]
    
    print("\nBloodPressure Statistics:")
    print(f"Min: {stats['min']}")
    print(f"Max: {stats['max']}")
    print(f"Mean: {stats['mean']}")
    print(f"Median: {stats['median']}")
    print(f"Variance: {stats['variance']}")
    print(f"Standard Deviation: {stats['stddev']}")
    
    # Save histogram as PNG
    bp_pd = df3.select("BloodPressure").toPandas()
    plt.figure(figsize=(10, 6))
    sns.histplot(bp_pd["BloodPressure"], bins=20, kde=True)
    plt.title("Distribution of BloodPressure")
    plt.xlabel("BloodPressure")
    plt.ylabel("Frequency")
    plt.savefig('visualizations/bloodpressure_hist.png')
    plt.close()
    print("Saved bloodpressure_hist.png")
else:
    print("BloodPressure column not found")

# =============================================
# 5. Quartile Information and Boxplot
# =============================================
print("\nTask 5: Quartile Information and Boxplot")

quantiles = df3.approxQuantile("TotalPurchases", [0.25, 0.5, 0.75], 0.01)
print("\nTotalPurchases Quartiles:")
print(f"Q1 (25th percentile): {quantiles[0]}")
print(f"Q2 (Median): {quantiles[1]}")
print(f"Q3 (75th percentile): {quantiles[2]}")

# Save boxplot as PNG
tp_pd = df3.select("TotalPurchases").toPandas()
plt.figure(figsize=(10, 6))
sns.boxplot(x=tp_pd["TotalPurchases"])
plt.title("Boxplot of TotalPurchases")
plt.xlabel("TotalPurchases")
plt.savefig('visualizations/total_purchases_boxplot.png')
plt.close()
print("Saved total_purchases_boxplot.png")

# =============================================
# 6. Relationship Between PurchaseAmount and SpendingScore
# =============================================
print("\nTask 6: Relationship Between PurchaseAmount and SpendingScore")

# Save scatter plot as PNG
pa_ss_pd = df3.select("PurchaseAmount", "SpendingScore").toPandas()
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PurchaseAmount", y="SpendingScore", data=pa_ss_pd)
plt.title("PurchaseAmount vs SpendingScore")
plt.xlabel("PurchaseAmount")
plt.ylabel("SpendingScore")
plt.savefig('visualizations/purchase_vs_spending.png')
plt.close()
print("Saved purchase_vs_spending.png")

corr = df3.corr("PurchaseAmount", "SpendingScore")
print(f"\nPearson correlation: {corr}")

# =============================================
# 7. Spark SQL Query
# =============================================
# First check if the conditions are possible
min_age, max_age = df3.select(min("Age"), max("Age")).first()
min_score, max_score = df3.select(min("SpendingScore"), max("SpendingScore")).first()

print(f"Age range: {min_age}-{max_age}")
print(f"SpendingScore range: {min_score}-{max_score}")

df3.createOrReplaceTempView("customers")
result = spark.sql("""
    SELECT Age, SpendingScore 
    FROM customers 
    WHERE Age < 50 AND SpendingScore > 100
""")

count = result.count()
if count > 0:
    print(f"\nFound {count} customers meeting criteria:")
    result.show()
else:
    print("\nNo customers found with Age < 50 and SpendingScore > 100")
    print("Showing customers with highest SpendingScore instead:")
    df3.orderBy(desc("SpendingScore")).select("Age", "SpendingScore").show(5)

# =============================================
# 8. Decision Tree Classifier
# =============================================
print("\nTask 8: Decision Tree Classifier")

# Create binary Outcome based on SpendingScore
# Customers with SpendingScore > median will be labeled as 1 (high spender), others as 0
spending_median = df3.approxQuantile("SpendingScore", [0.5], 0.01)[0]
df3_with_outcome = df3.withColumn("Outcome", 
    when(col("SpendingScore") > spending_median, 1.0).otherwise(0.0))

print(f"\nCreated binary Outcome based on SpendingScore median ({spending_median:.2f}):")
print("0 = Low spender, 1 = High spender")
df3_with_outcome.groupBy("Outcome").count().show()

# Prepare features and label
categorical_cols = ["Gender", "PurchaseCategory"]
indexers = [StringIndexer(inputCol=col, outputCol=col+"_indexed") for col in categorical_cols]

# Create numerical feature list (excluding CustomerID and Outcome)
numerical_cols = [col for col in df3_with_outcome.columns 
                 if col not in ["CustomerID", "Outcome", "SpendingScore"] + categorical_cols]

# Create pipeline for categorical conversion
indexing_pipeline = Pipeline(stages=indexers)
df_indexed = indexing_pipeline.fit(df3_with_outcome).transform(df3_with_outcome)

# Combine all features
feature_cols = numerical_cols + [col+"_indexed" for col in categorical_cols]

# Create feature vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)
df_assembled = assembler.transform(df_indexed)

# Split the data
train_data, test_data = df_assembled.randomSplit([0.7, 0.3], seed=42)

# Initialize and train the Decision Tree Classifier
dt = DecisionTreeClassifier(
    labelCol="Outcome",
    featuresCol="features",
    maxDepth=5,
    seed=42
)
dt_model = dt.fit(train_data)

# Make predictions
predictions = dt_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(
    labelCol="Outcome",
    metricName="areaUnderROC"
)
auc_roc = evaluator.evaluate(predictions)

# Calculate accuracy
accuracy = predictions.filter(col("prediction") == col("Outcome")).count() / test_data.count()

print("\nDecision Tree Model Performance:")
print(f"Area Under ROC: {auc_roc:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Feature importance analysis
print("\nFeature Importances:")
for feature, importance in sorted(zip(feature_cols, dt_model.featureImportances.toArray()), 
                                key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.4f}")

# Confusion Matrix
conf_matrix = predictions.groupBy("Outcome", "prediction").count().orderBy("Outcome", "prediction")
print("\nConfusion Matrix:")
conf_matrix.show()

# Visualize Decision Tree performance
predictions_pd = predictions.select("Outcome", "prediction", "probability").toPandas()
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(predictions_pd["Outcome"], predictions_pd["prediction"]), 
            annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('visualizations/decision_tree_confusion_matrix.png')
plt.close()
print("Saved decision_tree_confusion_matrix.png")

# =============================================
# 9. Logistic Regression Classifier
# =============================================
print("\nTask 9: Logistic Regression Classifier")

if "Outcome" in df3.columns:
    lr = LogisticRegression(labelCol="Outcome", featuresCol="features")
    lr_model = lr.fit(train_data)
    
    lr_predictions = lr_model.transform(test_data)
    
    lr_accuracy = evaluator.evaluate(lr_predictions)
    print(f"\nLogistic Regression AUC: {lr_accuracy}")
    
    print("\nModel Coefficients:")
    for feature, coef in zip(feature_cols, lr_model.coefficients):
        print(f"{feature}: {coef}")
else:
    print("Outcome column not found")

# =============================================
# 10. Linear Regression Model
# =============================================
print("\nTask 10: Linear Regression Model")

lr_assembler = VectorAssembler(inputCols=["AnnualIncome"], outputCol="features")
df3_lr = lr_assembler.transform(df3)

lr_train_data, lr_test_data = df3_lr.randomSplit([0.7, 0.3], seed=42)

lin_reg = LinearRegression(featuresCol="features", labelCol="PurchaseAmount")
lin_reg_model = lin_reg.fit(lr_train_data)

lr_predictions = lin_reg_model.transform(lr_test_data)

reg_evaluator = RegressionEvaluator(labelCol="PurchaseAmount", predictionCol="prediction")
rmse = reg_evaluator.evaluate(lr_predictions, {reg_evaluator.metricName: "rmse"})
r2 = reg_evaluator.evaluate(lr_predictions, {reg_evaluator.metricName: "r2"})

print(f"\nLinear Regression RMSE: {rmse}")
print(f"R-squared: {r2}")

print("\nModel Summary:")
print(f"Intercept: {lin_reg_model.intercept}")
print(f"Coefficient for AnnualIncome: {lin_reg_model.coefficients[0]}")

# =============================================
# Cleanup
# =============================================
spark.stop()
print("\nSpark session stopped.")
print("All visualizations saved to 'visualizations' folder")