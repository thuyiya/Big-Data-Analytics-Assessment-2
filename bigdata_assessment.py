# Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan, lit, expr, min, max, mean, stddev, variance
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BigDataAssessment") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
    .getOrCreate()

# Task 1: Load the dataset and describe structure
df1 = spark.read.csv("/Users/tj/Documents/Education/Big Data/ass 2/bigdata_assessment/customer_purchases.csv", 
                    header=True, 
                    inferSchema=True)

print("=== Task 1 ===")
print("DataFrame Structure:")
df1.printSchema()
print(f"Number of rows: {df1.count()}")
print(f"Number of columns: {len(df1.columns)}")

# Task 2: Replace null/missing values with medians
print("\n=== Task 2 ===")
medians = df1.approxQuantile(["SpendingScore", "TotalPurchases"], [0.5], 0.25)
spending_score_median = medians[0][0]
total_purchases_median = medians[1][0]

df2 = df1.withColumn("SpendingScore", 
                    when((col("SpendingScore").isNull()) | (col("SpendingScore") == 0), 
                         spending_score_median)
                    .otherwise(col("SpendingScore"))) \
         .withColumn("TotalPurchases", 
                    when((col("TotalPurchases").isNull()) | (col("TotalPurchases") == 0), 
                         total_purchases_median)
                    .otherwise(col("TotalPurchases")))

print("Missing values after replacement:")
df2.select([count(when(col(c).isNull() | (col(c) == 0), c)).alias(c) for c in ["SpendingScore", "TotalPurchases"]]).show()

# Task 3: Remove rows with missing values in key columns
print("\n=== Task 3 ===")
df3 = df2.filter((col("Age") != 0) & (~col("Age").isNull()) & 
                (col("AnnualIncome") != 0) & (~col("AnnualIncome").isNull()) & 
                (col("PurchaseAmount") != 0) & (~col("PurchaseAmount").isNull()))

rows_removed = df2.count() - df3.count()
print(f"Number of rows removed: {rows_removed}")

# Task 4: Summary statistics and histogram
print("\n=== Task 4 ===")
stats = df3.select(
    min("PurchaseAmount").alias("min"),
    max("PurchaseAmount").alias("max"),
    mean("PurchaseAmount").alias("mean"),
    expr("percentile_approx(PurchaseAmount, 0.5)").alias("median"),
    variance("PurchaseAmount").alias("variance"),
    stddev("PurchaseAmount").alias("stddev")
).collect()[0]

print("PurchaseAmount Statistics:")
print(f"Min: {stats['min']:.2f}")
print(f"Max: {stats['max']:.2f}")
print(f"Mean: {stats['mean']:.2f}")
print(f"Median: {stats['median']:.2f}")
print(f"Variance: {stats['variance']:.2f}")
print(f"Standard Deviation: {stats['stddev']:.2f}")

# Enhanced Histogram with more details
pd_purchase = df3.select("PurchaseAmount").toPandas()
plt.figure(figsize=(12, 6))
plt.hist(pd_purchase["PurchaseAmount"], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Purchase Amounts", fontsize=14, pad=20)
plt.xlabel("Purchase Amount (£)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=2, label=f'Mean: £{stats["mean"]:.2f}')
plt.axvline(stats['median'], color='green', linestyle='dashed', linewidth=2, label=f'Median: £{stats["median"]:.2f}')
plt.legend()
plt.tight_layout()
plt.savefig("purchase_amount_histogram.png", dpi=300, bbox_inches='tight')
plt.close()
print("Enhanced histogram saved as purchase_amount_histogram.png")

# Task 5: Quartile info and boxplot
print("\n=== Task 5 ===")
quartiles = df3.approxQuantile("TotalPurchases", [0.25, 0.5, 0.75], 0.05)
print(f"Q1 (25th percentile): {quartiles[0]:.2f}")
print(f"Q2 (Median): {quartiles[1]:.2f}")
print(f"Q3 (75th percentile): {quartiles[2]:.2f}")

# Enhanced Boxplot with proper data
pd_total_purchases = df3.select("TotalPurchases").toPandas()
plt.figure(figsize=(10, 6))
sns.boxplot(y=pd_total_purchases["TotalPurchases"], color='lightgreen')
plt.title("Distribution of Total Purchases", fontsize=14)
plt.ylabel("Number of Purchases", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("total_purchases_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()
print("Enhanced boxplot saved as total_purchases_boxplot.png")

# Task 6: Relationship between PurchaseAmount and SpendingScore
print("\n=== Task 6 ===")
pd_scatter = df3.select("PurchaseAmount", "SpendingScore").toPandas()
plt.figure(figsize=(12, 6))
sns.scatterplot(data=pd_scatter, x="SpendingScore", y="PurchaseAmount", alpha=0.6, color='purple')
plt.title("Purchase Amount vs Spending Score", fontsize=14)
plt.xlabel("Spending Score (1-100)", fontsize=12)
plt.ylabel("Purchase Amount (£)", fontsize=12)
plt.grid(True, alpha=0.3)

# Add correlation text
plt.text(0.05, 0.95, f'Pearson r = {correlation:.2f}', 
         transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("purchase_vs_spending.png", dpi=300, bbox_inches='tight')
plt.close()

correlation = df3.stat.corr("PurchaseAmount", "SpendingScore", "pearson")
print(f"Pearson correlation coefficient: {correlation:.4f}")

# Task 7: Spark SQL query
print("\n=== Task 7 ===")
df3.createOrReplaceTempView("customers")
result = spark.sql("""
    SELECT Age, SpendingScore 
    FROM customers 
    WHERE Age < 50 AND SpendingScore > 100
    ORDER BY SpendingScore DESC
    LIMIT 10
""")
print("Top 10 Customers with Age < 50 and SpendingScore > 100:")
result.show(truncate=False)

# Visualization for Task 7
pd_top_customers = result.toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(data=pd_top_customers, x="Age", y="SpendingScore", palette="viridis")
plt.title("Top Customers: Age vs Spending Score", fontsize=14)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Spending Score", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("top_customers.png", dpi=300, bbox_inches='tight')
plt.close()
print("Top customers visualization saved as top_customers.png")

# Task 8: Decision Tree Classifier
print("\n=== Task 8 ===")
# Create binary outcome based on PurchaseAmount > median
median_purchase = df3.approxQuantile("PurchaseAmount", [0.5], 0.05)[0]
df_model = df3.withColumn("Outcome", when(col("PurchaseAmount") > median_purchase, 1).otherwise(0))

# Prepare features
feature_cols = ["Age", "AnnualIncome", "SpendingScore", "TotalPurchases"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df_model)

# Split data
train, test = df_assembled.randomSplit([0.8, 0.2], seed=42)

# Train model
dt = DecisionTreeClassifier(labelCol="Outcome", featuresCol="features")
dt_model = dt.fit(train)

# Make predictions
predictions = dt_model.transform(test)

# Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="Outcome", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print("Decision Tree Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Task 9: Logistic Regression Classifier
print("\n=== Task 9 ===")
lr = LogisticRegression(labelCol="Outcome", featuresCol="features")
lr_model = lr.fit(train)

lr_predictions = lr_model.transform(test)

lr_accuracy = evaluator.evaluate(lr_predictions, {evaluator.metricName: "accuracy"})
lr_precision = evaluator.evaluate(lr_predictions, {evaluator.metricName: "weightedPrecision"})
lr_recall = evaluator.evaluate(lr_predictions, {evaluator.metricName: "weightedRecall"})
lr_f1 = evaluator.evaluate(lr_predictions, {evaluator.metricName: "f1"})

print("Logistic Regression Performance:")
print(f"Accuracy: {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"F1 Score: {lr_f1:.4f}")

# Task 10: Linear Regression Model
print("\n=== Task 10 ===")
lr_assembler = VectorAssembler(inputCols=["AnnualIncome"], outputCol="features")
df_lr = lr_assembler.transform(df3)

train_lr, test_lr = df_lr.randomSplit([0.8, 0.2], seed=42)

linear_reg = LinearRegression(featuresCol="features", labelCol="PurchaseAmount")
lr_model = linear_reg.fit(train_lr)

lr_predictions = lr_model.transform(test_lr)

evaluator = RegressionEvaluator(labelCol="PurchaseAmount", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(lr_predictions)
evaluator = RegressionEvaluator(labelCol="PurchaseAmount", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(lr_predictions)

print("Linear Regression Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
print(f"Coefficient: {lr_model.coefficients[0]:.6f}")
print(f"Intercept: {lr_model.intercept:.2f}")

# Stop Spark session
spark.stop()