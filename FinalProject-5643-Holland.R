required_packages <- c("caret", "arules", "arulesViz", "rpart", "rpart.plot")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(paste("Install these packages before running:",
             paste(missing_packages, collapse = ", "),
             "\nUse install.packages(c("caret", "arules", "arulesViz", "rpart", "rpart.plot"))"))
}

#Step 1: Data exploration and understanding
#Load file and set working directory
#Read csv file
data_file <- "diabetes_012_health_indicators_BRFSS2015.csv"
if (!file.exists(data_file)) {
  stop("CSV not found. Put the original dataset in your working directory or update data_file.")
}
df <- read.csv(data_file)
required_columns <- c("Diabetes_012", "BMI", "HighBP", "HighChol",
                      "PhysActivity", "Smoker", "Age", "GenHlth")
if (!all(required_columns %in% names(df))) stop("Required columns are missing from the CSV.")
if (!all(vapply(df[required_columns], is.numeric, logical(1)))) {
  stop("Expected numeric columns in the original dataset.")
}
if (anyNA(df[required_columns])) stop("Missing values found; review before analysis.")
if (!all(df$Diabetes_012 %in% 0:2) || length(unique(df$Diabetes_012)) != 3) {
  stop("Expected all three outcome codes: 0, 1, 2.")
}

#Explore data
print(head(df))
str(df)
print(summary(df))

#Identify key features and target variable
#Target variable is Diabetes_012
#Key features include: BMI (body mass index)
#HighBP (high blood pressure)
#HighChol (high cholesterol)
#PhysActivity (physical activity level)
#Smoker
#Age
#GenHlth (Self-rated general health)
#These features were selected because they represent common medical
#and lifestyle factors associated with diabetes risk according to the CDC.

#While several key features were identified, I'm going to focus visualizations on BMI,
#high blood pressure, and physical activity as predictors.
#These variables provide a mix of medical and lifestyle factors to illustrate relationships with diabetes status.

#Histogram of BMI to show distribution
hist(df$BMI,
     main="BMI Distribution",
     xlab="BMI",
     col="lightblue")

#Box plot of BMI vs Diabetes to show relationship to the target variable
boxplot(BMI ~ Diabetes_012, data=df,
        main="BMI vs Diabetes",
        xlab="Diabetes Status",
        ylab="BMI",
        col="lightgreen")

#Bar plot of high BP vs diabetes. 
table_bp <- table(df$HighBP, df$Diabetes_012)
barplot(table_bp,
        main="High Blood Pressure vs Diabetes",
        xlab="Diabetes Status",
        ylab="Count",
        col=c("pink","lightblue"))
legend("topright",
       legend=c("No High BP","High BP"),
       fill=c("pink","lightblue"),
       bty="n")

#Bar plot of physical activity vs diabetes
table_pa <- table(df$PhysActivity, df$Diabetes_012)
barplot(table_pa,
        main="Physical Activity vs Diabetes",
        xlab="Diabetes Status",
        ylab="Count",
        col=c("orange","lightblue"))
legend("topright",
       legend=c("No Physical Activity","Physically Active"),
       fill=c("orange","lightblue"),
       bty="n")


#Step 2: Data Preprocessing

#Check for missing values
sum(is.na(df))

#This dataset contains no missing values, so no imputation or deletion is needed.

#Convert target variable to factor for classification
df$Diabetes_012 <- as.factor(df$Diabetes_012)

#Split the data into training and testing sets using createDataPartition
#This method preserves the proportion of diabetic vs non-diabetic cases

#Load caret
library(caret)

#Set seed for reproducibility
set.seed(123)

index <- createDataPartition(df$Diabetes_012, p = 0.7, list = FALSE)

train <- df[index, ]
test <- df[-index, ]

# Retain the original BMI scaling exercise, using training data only.
# Decision trees do not require scaling to prevent large values dominating.
bmi_min <- min(train$BMI)
bmi_range <- max(train$BMI) - bmi_min
if (bmi_range == 0) stop("Training BMI has no variation.")
train$BMI_scaled <- (train$BMI - bmi_min) / bmi_range
test$BMI_scaled <- (test$BMI - bmi_min) / bmi_range

#Using createDataPartition to split the data ensures
#that the training and testing sets maintain similar proportions of the
#target variable, which is important for classification problems with imbalanced datasets like this one.

#Step 3: Data mining and exploration
#Load association-rule packages
library(arules)
library(arulesViz)

#Corrleation analysis

#Convert Diabetes_012 to numeric var for correlation purposes
diabetes_numeric <- as.numeric(as.character(train$Diabetes_012))
# These correlations describe numeric category codes only; they do not
# establish causation or equal clinical distances between the categories.

#Compute simple correlations between numeric predictors and diabetes status
cor_BMI      <- cor(train$BMI, diabetes_numeric)
cor_Age      <- cor(train$Age, diabetes_numeric)
cor_HighBP   <- cor(train$HighBP, diabetes_numeric)
cor_HighChol <- cor(train$HighChol, diabetes_numeric)

#Store correlation values in a table
correlations <- data.frame(
  Variable = c("BMI", "Age", "HighBP", "HighChol"),
  Correlation = c(cor_BMI, cor_Age, cor_HighBP, cor_HighChol)
)

print(correlations)

#Visualize correlations in a simple bar plot
barplot(correlations$Correlation,
        names.arg = correlations$Variable,
        main = "Correlation with Diabetes Status",
        ylab = "Correlation Coefficient",
        col = "lightblue",
        las = 2)


#Association rule mining
#Prepare categorical data for Apriori
assoc_df <- train[, c("HighBP", "HighChol", "Smoker",
                      "PhysActivity", "GenHlth", "Diabetes_012")]

assoc_df <- as.data.frame(lapply(assoc_df, as.factor))

#Apply Apriori 
rules <- apriori(
  assoc_df,
  parameter = list(support = 0.0005, confidence = 0.5, minlen = 2)
)

# Count generated rules before inspecting or plotting them.
num_rules <- length(rules)
print(paste("Association rules generated:", num_rules))
if (num_rules == 0) {
  print("No association rules met the specified thresholds.")
} else {
  inspect(head(rules, 10))
  tryCatch(
    plot(rules[seq_len(min(20, num_rules))], method = "graph"),
    error = function(e) message("Rule plot unavailable: ", conditionMessage(e))
  )
}
# Predictor selection is exploratory; correlations alone do not establish
# which variables are most important for prediction.

#Step 4: Model selection and application

#Problem Type: Multiclass Classification
#Our goal is to build a model that predicts an individual's diabetes status
#(0 = no diabetes, 1 = prediabetes, 2 = diabetes) based on health and
#lifestyle factors. Because the target variable is categorical with three
#classes, this is a classification problem.

#We selected a decision tree because:
#It handles both numeric and categorical predictors
#It works naturally with multiclass classification
#It is easy to interpret

#Load decision-tree packages
library(rpart)
library(rpart.plot)

#Train the initial decision tree

tree_model <- rpart(
  Diabetes_012 ~ BMI_scaled + HighBP + HighChol +
    PhysActivity + Smoker + Age + GenHlth,
  data = train,
  method = "class",
  control = rpart.control(cp = 0.001)
)

#Prune the model
#Review complexity parameter table
printcp(tree_model)

#Select the cp value with the lowest cross-validated error
best_cp <- tree_model$cptable[
  which.min(tree_model$cptable[,"xerror"]), "CP"
]

#Prune the tree using the selected cp
pruned_tree <- prune(tree_model, cp = best_cp)

#Visualize both trees

rpart.plot(tree_model,
           main = "Initial Decision Tree",
           type = 2, extra = 104)

rpart.plot(pruned_tree,
           main = "Pruned Decision Tree (Optimized)",
           type = 2, extra = 104)

#Step 5: Model evaluation and diagnostics

#Generate predictions
#Predictions on the test set
test_predictions <- predict(pruned_tree, newdata = test, type = "class")

#Predictions on the training set
train_predictions <- predict(pruned_tree, newdata = train, type = "class")

# Calculate and print accuracy even when the whole file is sourced.
train_accuracy <- mean(train_predictions == train$Diabetes_012)
test_accuracy <- mean(test_predictions == test$Diabetes_012)

# Choose the baseline class from training data, then evaluate on test data.
majority_class <- names(which.max(table(train$Diabetes_012)))
baseline_accuracy <- mean(as.character(test$Diabetes_012) == majority_class)
print(data.frame(
  Measure = c("Training accuracy", "Testing accuracy", "Majority-class test baseline"),
  Accuracy = c(train_accuracy, test_accuracy, baseline_accuracy)
))

# Additional diagnostics: inspect each class instead of relying
# only on overall accuracy, which can obscure minority-class performance.
print(table(Predicted = test_predictions, Actual = test$Diabetes_012))
print(caret::confusionMatrix(test_predictions, test$Diabetes_012))
