print("Big Data Analytics Assessment 2 - PySpark Implementation")

# Import necessary modules
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# =============================================
# 2. Handling Missing Values
# =============================================
print("\nTask 2: Handling Missing Values")

# Calculate medians
medians = df1.approxQuantile(["SpendingScore", "TotalPurchases"], [0.5], 0.01)
spending_score_median = medians[0][0]
total_purchases_median = medians[1][0]

print(f"Median SpendingScore: {spending_score_median}")
print(f"Median TotalPurchases: {total_purchases_median}")

# Replace nulls with medians
df2 = df1.withColumn(
    "SpendingScore", 
    when((col("SpendingScore").isNull()) | (col("SpendingScore") == 0), spending_score_median)
    .otherwise(col("SpendingScore"))
).withColumn(
    "TotalPurchases",
    when((col("TotalPurchases").isNull()) | (col("TotalPurchases") == 0), total_purchases_median)
    .otherwise(col("TotalPurchases"))
)

# Verify replacement
print("\nMissing values after replacement:")
df2.select([
    sum(when(col(c).isNull() | (col(c) == 0), 1).otherwise(0)).alias(c) 
    for c in ["SpendingScore", "TotalPurchases"]
]).show()

# =============================================
# 3. Removing Rows with Missing Values
# =============================================
print("\nTask 3: Removing Rows with Missing Values")

df3 = df2.filter(
    (col("Age") != 0) & 
    (col("Age").isNotNull()) &
    (col("AnnualIncome") != 0) & 
    (col("AnnualIncome").isNotNull()) &
    (col("PurchaseAmount") != 0) & 
    (col("PurchaseAmount").isNotNull())
)

rows_removed = df2.count() - df3.count()
print(f"Number of rows removed: {rows_removed}")
print(f"Rows in cleaned DataFrame: {df3.count()}")

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
print("\nTask 7: Spark SQL Query")

df3.createOrReplaceTempView("customers")
result = spark.sql("""
    SELECT Age, SpendingScore 
    FROM customers 
    WHERE Age < 50 AND SpendingScore > 100
""")

print("\nCustomers with Age < 50 and SpendingScore > 100:")
result.show()

# =============================================
# 8. Decision Tree Classifier
# =============================================
print("\nTask 8: Decision Tree Classifier")

if "Outcome" in df3.columns:
    categorical_cols = [col for col in df3.columns if col in ["Gender", "PurchaseCategory"]]
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(df3) for col in categorical_cols]
    
    pipeline = Pipeline(stages=indexers)
    df3_indexed = pipeline.fit(df3).transform(df3)
    
    feature_cols = [col for col in df3.columns 
                   if col not in ["CustomerID", "Outcome"] + categorical_cols] + [col+"_index" for col in categorical_cols]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df3_assembled = assembler.transform(df3_indexed)
    
    train_data, test_data = df3_assembled.randomSplit([0.7, 0.3], seed=42)
    
    dt = DecisionTreeClassifier(labelCol="Outcome", featuresCol="features")
    dt_model = dt.fit(train_data)
    
    predictions = dt_model.transform(test_data)
    
    evaluator = BinaryClassificationEvaluator(labelCol="Outcome")
    accuracy = evaluator.evaluate(predictions)
    print(f"\nDecision Tree AUC: {accuracy}")
    
    print("\nFeature Importances:")
    for feature, importance in zip(feature_cols, dt_model.featureImportances):
        print(f"{feature}: {importance}")
else:
    print("Outcome column not found")

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