
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

file_path = r"C:\Users\ajit kumar\Downloads\accuracy-main\accuracy-main\heart.csv"
df = pd.read_csv(file_path)

irrelevant_columns = ["id", "dataset"]#basically goes to every column and se data is missing or not
df = df.drop(columns=[col for col in irrelevant_columns if col in df.columns], errors="ignore")

df.fillna(df.median(numeric_only=True), inplace=True)#replace the missing values with the median of the column

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:#selects every column 
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])#fit_transform is used to convert the data into numerical form
    label_encoders[col] = le#stores the label encoder in the dictionary

target_col = "target"#deburging the target column
if target_col not in df.columns:
    raise ValueError(f"Error: Column '{target_col}' not found in the dataset. Check the CSV file.")

X = df.drop(columns=[target_col])#spliting the data into X and y x contains all the columns except the target column
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

accuracy_scores = {}
classification_reports = {}
confusion_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy
    
    class_report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports[name] = class_report

    confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrices[name])

accuracy_df = pd.DataFrame(accuracy_scores.items(), columns=["Model", "Accuracy"])

plt.figure(figsize=(10, 5))
sns.barplot(x="Accuracy", y="Model", data=accuracy_df, palette="viridis")
plt.xlabel("Accuracy Score")
plt.ylabel("Model")
plt.title("Model Performance Comparison")
plt.xlim(0, 1)
plt.show()