print("Big Data Analytics Assessment 2 - PySpark Implementation")
# 1. Initial Setup and Data Loading
print("\nTask 1: Initial Setup and Data Loading")
# Import necessary PySpark modules
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

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BigDataAnalyticsAssessment2") \
    .getOrCreate()

# Load the dataset into PySpark DataFrame (1st DataFrame)
file_path = "customer_purchases.csv"  # Update with your actual file path
df1 = spark.read.csv(file_path, header=True, inferSchema=True)

# Describe the structure of the DataFrame
print("DataFrame Schema:")
df1.printSchema()

print("\nFirst 5 rows:")
df1.show(5)

print("\nSummary Statistics:")
df1.describe().show()

# 2. Handling Missing Values
print("\nTask 2: Handling Missing Values")
# Calculate median values for SpendingScore and TotalPurchases
from pyspark.sql.functions import expr

# Calculate medians (PySpark doesn't have direct median function, so we use approxQuantile)
medians = df1.approxQuantile(["SpendingScore", "TotalPurchases"], [0.5], 0.01)
spending_score_median = medians[0][0]
total_purchases_median = medians[1][0]

print(f"Median SpendingScore: {spending_score_median}")
print(f"Median TotalPurchases: {total_purchases_median}")

# Replace null/missing values (0s) with medians and create 2nd DataFrame
df2 = df1.withColumn(
    "SpendingScore", 
    when((col("SpendingScore").isNull()) | (col("SpendingScore") == 0), spending_score_median)
    .otherwise(col("SpendingScore"))
).withColumn(
    "TotalPurchases",
    when((col("TotalPurchases").isNull()) | (col("TotalPurchases") == 0), total_purchases_median)
    .otherwise(col("TotalPurchases"))
)

# Verify the replacement
print("\nMissing values after replacement:")
df2.select([
    sum(when(col(c).isNull() | (col(c) == 0), 1).otherwise(0)).alias(c) 
    for c in ["SpendingScore", "TotalPurchases"]
]).show()

# 3. Removing Rows with Missing Values
print("\nTask 3: Removing Rows with Missing Values")
# Create 3rd DataFrame by removing rows with null/0 values in Age, AnnualIncome, or PurchaseAmount
df3 = df2.filter(
    (col("Age") != 0) & 
    (col("Age").isNotNull()) &
    (col("AnnualIncome") != 0) & 
    (col("AnnualIncome").isNotNull()) &
    (col("PurchaseAmount") != 0) & 
    (col("PurchaseAmount").isNotNull())
)

# Compute number of rows removed
rows_removed = df2.count() - df3.count()
print(f"Number of rows removed: {rows_removed}")

# Show the count of the cleaned DataFrame
print(f"Rows in cleaned DataFrame: {df3.count()}")

# 4. Summary Statistics and Histogram
print("\nTask 4: Summary Statistics and Histogram")
# Compute summary statistics for BloodPressure (assuming this column exists)
if "BloodPressure" in df3.columns:
    # Summary statistics
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
    
    # Generate histogram (convert to Pandas for visualization)
    bp_pd = df3.select("BloodPressure").toPandas()
    plt.figure(figsize=(10, 6))
    sns.histplot(bp_pd["BloodPressure"], bins=20, kde=True)
    plt.title("Distribution of BloodPressure")
    plt.xlabel("BloodPressure")
    plt.ylabel("Frequency")
    plt.show()
else:
    print("BloodPressure column not found in the DataFrame")

# 5. Quartile Information and Boxplot
print("\nTask 5: Quartile Information and Boxplot")
# Quartile info for TotalPurchases
quantiles = df3.approxQuantile("TotalPurchases", [0.25, 0.5, 0.75], 0.01)
print("\nTotalPurchases Quartiles:")
print(f"Q1 (25th percentile): {quantiles[0]}")
print(f"Q2 (Median): {quantiles[1]}")
print(f"Q3 (75th percentile): {quantiles[2]}")

# Boxplot for TotalPurchases (convert to Pandas for visualization)
tp_pd = df3.select("TotalPurchases").toPandas()
plt.figure(figsize=(10, 6))
sns.boxplot(x=tp_pd["TotalPurchases"])
plt.title("Boxplot of TotalPurchases")
plt.xlabel("TotalPurchases")
plt.show()

# 6. Relationship Between PurchaseAmount and SpendingScore
print("\nTask 6: Relationship Between PurchaseAmount and SpendingScore")
# Scatter plot of PurchaseAmount vs SpendingScore
pa_ss_pd = df3.select("PurchaseAmount", "SpendingScore").toPandas()
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PurchaseAmount", y="SpendingScore", data=pa_ss_pd)
plt.title("PurchaseAmount vs SpendingScore")
plt.xlabel("PurchaseAmount")
plt.ylabel("SpendingScore")
plt.show()

# Calculate Pearson correlation
corr = df3.corr("PurchaseAmount", "SpendingScore")
print(f"\nPearson correlation between PurchaseAmount and SpendingScore: {corr}")

# 7. Spark SQL Query
print("\nTask 7: Spark SQL Query")
# Create a temporary view for SQL queries
df3.createOrReplaceTempView("customers")

# Execute SQL query
result = spark.sql("""
    SELECT Age, SpendingScore 
    FROM customers 
    WHERE Age < 50 AND SpendingScore > 100
""")

print("\nCustomers with Age < 50 and SpendingScore > 100:")
result.show()

# 8. Decision Tree Classifier
print("\nTask 8: Decision Tree Classifier")
# Prepare data for Decision Tree
if "Outcome" in df3.columns:
    # Index categorical columns
    categorical_cols = [col for col in df3.columns if col in ["Gender", "PurchaseCategory"]]
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(df3) for col in categorical_cols]
    
    # Pipeline for indexing
    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages=indexers)
    df3_indexed = pipeline.fit(df3).transform(df3)
    
    # Select features (excluding CustomerID and Outcome)
    feature_cols = [col for col in df3.columns 
                   if col not in ["CustomerID", "Outcome"] + categorical_cols] + [col+"_index" for col in categorical_cols]
    
    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df3_assembled = assembler.transform(df3_indexed)
    
    # Split data
    train_data, test_data = df3_assembled.randomSplit([0.7, 0.3], seed=42)
    
    # Train Decision Tree
    dt = DecisionTreeClassifier(labelCol="Outcome", featuresCol="features")
    dt_model = dt.fit(train_data)
    
    # Make predictions
    predictions = dt_model.transform(test_data)
    
    # Evaluate model
    evaluator = BinaryClassificationEvaluator(labelCol="Outcome")
    accuracy = evaluator.evaluate(predictions)
    print(f"\nDecision Tree AUC: {accuracy}")
    
    # Feature importance
    print("\nFeature Importances:")
    for feature, importance in zip(feature_cols, dt_model.featureImportances):
        print(f"{feature}: {importance}")
else:
    print("Outcome column not found in the DataFrame")

# 9. Logistic Regression Classifier
print("\nTask 9: Logistic Regression Classifier")
# Prepare data for Logistic Regression
if "Outcome" in df3.columns:
    # Use the same prepared data from Decision Tree
    
    # Train Logistic Regression
    lr = LogisticRegression(labelCol="Outcome", featuresCol="features")
    lr_model = lr.fit(train_data)
    
    # Make predictions
    lr_predictions = lr_model.transform(test_data)
    
    # Evaluate model
    lr_accuracy = evaluator.evaluate(lr_predictions)
    print(f"\nLogistic Regression AUC: {lr_accuracy}")
    
    # Coefficients
    print("\nModel Coefficients:")
    for feature, coef in zip(feature_cols, lr_model.coefficients):
        print(f"{feature}: {coef}")
else:
    print("Outcome column not found in the DataFrame")

# 10. Linear Regression Model
print("\nTask 10: Linear Regression Model")
# Prepare data for Linear Regression
# Assemble features (just AnnualIncome in this case)
lr_assembler = VectorAssembler(inputCols=["AnnualIncome"], outputCol="features")
df3_lr = lr_assembler.transform(df3)

# Split data
lr_train_data, lr_test_data = df3_lr.randomSplit([0.7, 0.3], seed=42)

# Train Linear Regression
lin_reg = LinearRegression(featuresCol="features", labelCol="PurchaseAmount")
lin_reg_model = lin_reg.fit(lr_train_data)

# Make predictions
lr_predictions = lin_reg_model.transform(lr_test_data)

# Evaluate model
reg_evaluator = RegressionEvaluator(labelCol="PurchaseAmount", predictionCol="prediction")
rmse = reg_evaluator.evaluate(lr_predictions, {reg_evaluator.metricName: "rmse"})
r2 = reg_evaluator.evaluate(lr_predictions, {reg_evaluator.metricName: "r2"})

print(f"\nLinear Regression RMSE: {rmse}")
print(f"R-squared: {r2}")

# Show coefficients
print("\nModel Summary:")
print(f"Intercept: {lin_reg_model.intercept}")
print(f"Coefficient for AnnualIncome: {lin_reg_model.coefficients[0]}")
# Stop Spark session
spark.stop()
print("\nSpark session stopped.")
# End of the script