import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def generate_synthetic_data(filename="customer_dataset.csv"):
    np.random.seed(42)
    n_samples = 1500
    
    countries = ['USA', 'UK', 'Germany', 'France', 'Australia', 'Canada', 'Japan', 'Brazil', 'India', 'Italy', 'Spain', 'Mexico']
    
    data = {
        'Quantity': np.random.randint(1, 100, n_samples),
        'UnitPrice': np.random.uniform(5.0, 500.0, n_samples),
        'Country': np.random.choice(countries, n_samples),
        'Missing_Feature': np.random.choice([np.nan, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.3, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Total Revenue
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Target Class logic: trying to predict if they are high value based on combinations
    total_spend = df['Revenue']
    noise = np.random.normal(0, 5000, n_samples)
    df['Target_Class'] = ((total_spend + noise) > np.median(total_spend)).astype(int)
    
    # Intentionally add some NaNs
    df.loc[df.sample(frac=0.05).index, 'Quantity'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'UnitPrice'] = np.nan
    
    df.to_csv(filename, index=False)

def main():
    print("="*60)
    print("ADVANCED MACHINE LEARNING PROJECT RUNNER")
    print("="*60)

    dataset_file = "customer_dataset.csv"
    
    if not os.path.exists(dataset_file):
        print("[*] No existing dataset found. Generating synthetic dataset...")
        generate_synthetic_data(dataset_file)
        
    os.makedirs("plots", exist_ok=True)

    print("\n[1/5] Loading and Preprocessing Data...")
    df = pd.read_csv(dataset_file)
    
    # Data Cleaning
    df['Quantity'] = df['Quantity'].fillna(df['Quantity'].median())
    df['UnitPrice'] = df['UnitPrice'].fillna(df['UnitPrice'].median())
    df['Missing_Feature'] = df['Missing_Feature'].fillna(df['Missing_Feature'].mode()[0])
    
    # Feature engineering for plots
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Encode categorical
    le = LabelEncoder()
    df['Country_Encoded'] = le.fit_transform(df['Country'])
    print(f"      -> Data successfully loaded and preprocessed. Shape: {df.shape}")

    print("\n[2/5] Generating Exploratory Data Analysis (EDA) Plots...")
    
    print("      -> Generating Customer Value Distribution... (1/7)")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Revenue'], bins=50, kde=True, color='blue')
    plt.title('Customer Value (Revenue) Distribution')
    plt.xlabel('Revenue')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('plots/01_customer_value_distribution.png')
    plt.close()

    print("      -> Generating Top 10 Countries by Revenue... (2/7)")
    plt.figure(figsize=(10, 6))
    country_revenue = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=country_revenue.values, y=country_revenue.index, hue=country_revenue.index, palette='viridis', legend=False)
    plt.title('Top 10 Countries by Revenue')
    plt.xlabel('Total Revenue')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig('plots/02_top10_countries_revenue.png')
    plt.close()

    print("      -> Generating Feature Correlation Heatmap... (3/7)")
    plt.figure(figsize=(8, 6))
    numeric_df = df[['Quantity', 'UnitPrice', 'Revenue', 'Missing_Feature', 'Country_Encoded', 'Target_Class']]
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/03_feature_correlation_heatmap.png')
    plt.close()

    print("\n[3/5] Preparing Data & Calculating Hyperparameters...")
    X = df[['Quantity', 'UnitPrice', 'Country_Encoded', 'Revenue', 'Missing_Feature']]
    y = df['Target_Class']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("      -> Calculating KNN Elbow Curve... (4/7)")
    k_values = range(1, 30)
    knn_accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        knn_accuracies.append(accuracy_score(y_test, knn.predict(X_test)))
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, knn_accuracies, marker='o', linestyle='dashed')
    plt.title('KNN Elbow Curve')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Test Accuracy')
    plt.axvline(x=19, color='r', linestyle='--', label='Chosen K=19')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/04_knn_elbow_curve.png')
    plt.close()

    print("\n[4/5] Evaluating Models and Cross-Validation...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN (k=19)": KNeighborsClassifier(n_neighbors=19)
    }

    results = []
    confusion_matrices = {}

    for name, model in models.items():
        print(f"      -> Training and Evaluating {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_acc = accuracy_score(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_acc = cv_scores.mean()
        
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Test Accuracy": test_acc,
            "CV Accuracy": cv_acc
        })

    print("\n[5/5] Generating Final Evaluation Plots...")
    print("      -> Saving Confusion Matrices... (5/7)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/05_confusion_matrices.png')
    plt.close()

    print("      -> Saving Model Comparison Graph... (6/7)")
    df_results = pd.DataFrame(results)
    df_melted = df_results.melt(id_vars='Model', value_vars=['Test Accuracy', 'CV Accuracy'], var_name='Metric', value_name='Accuracy')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Model', y='Accuracy', hue='Metric', palette='Set2')
    plt.title('Model Comparison: Test vs CV Accuracy')
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('plots/06_model_comparison.png')
    plt.close()

    print("      -> Saving Decision Tree Feature Importance... (7/7)")
    dt_model = models["Decision Tree"]
    importances = dt_model.feature_importances_
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=X.columns, hue=X.columns, palette='magma', legend=False)
    plt.title('Decision Tree Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/07_decision_tree_feature_importance.png')
    plt.close()

    print("\n" + "="*60)
    print("EXECUTION COMPLETE! All 7 plots successfully saved in 'plots/'.")
    print("="*60)
    
    print("\nMarkdown Table for README:")
    print("| Model | Test Accuracy | CV Accuracy |")
    print("|---|---|---|")
    for res in results:
        name = res['Model']
        if name == "KNN (k=19)":
            name = "KNN (k=19)"
        print(f"| {name} | {res['Test Accuracy']:.4f} | {res['CV Accuracy']:.4f} |")
    print("\n")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
