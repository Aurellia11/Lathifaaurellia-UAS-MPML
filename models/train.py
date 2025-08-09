import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Create directories
os.makedirs('app', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Custom transformer to handle sparse matrix for Naive Bayes
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray()

def perform_eda(df):
    """Exploratory Data Analysis with visualizations"""
    # Numeric features distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], bins=20, kde=True)
    plt.title('Price Distribution')
    plt.savefig('app/price_distribution.png')
    plt.close()
    
    # Profitability distribution
    plt.figure(figsize=(8, 5))
    df['Profitability'].value_counts().plot(kind='bar')
    plt.title('Profitability Distribution')
    plt.savefig('app/profitability_distribution.png')
    plt.close()
    
    # Price vs Profitability
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Profitability', y='Price', data=df)
    plt.title('Price vs Profitability')
    plt.savefig('app/price_vs_profitability.png')
    plt.close()

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Define features and target
    X = df.drop(['Profitability', 'RestaurantID', 'Ingredients'], axis=1)
    y = df['Profitability']
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define preprocessing pipeline
    numeric_features = ['Price']
    categorical_features = ['MenuCategory', 'MenuItem']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate multiple classifiers"""
    models = {
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'))
        ]),
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'))
        ]),
        'Naive Bayes': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('to_dense', DenseTransformer()),  # Convert sparse to dense
            ('classifier', GaussianNB())
        ])
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Handle class imbalance for Naive Bayes
        if name == 'Naive Bayes':
            sample_weights = compute_sample_weight('balanced', y_train)
            model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.2f}")
        
        # Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"\n{name} Performance:")
        print(f"Accuracy: {accuracy:.2f}")
        print(report)
        
        # Save results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'cv_scores': cv_scores
        }
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Profitable', 'Profitable'],
                    yticklabels=['Not Profitable', 'Profitable'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'app/confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    return results

def save_best_model(results):
    """Save the best performing model"""
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.2f}")
    joblib.dump(best_model, 'models/best_model.pkl')

def compare_models(results):
    """Create model comparison visualization"""
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    cv_means = [results[name]['cv_scores'].mean() for name in model_names]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Test Accuracy')
    plt.bar(x + width/2, cv_means, width, label='CV Mean Accuracy')
    
    plt.ylabel('Accuracy')
    plt.title('Model Comparison by Accuracy')
    plt.xticks(x, model_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig('app/model_comparison.png')
    plt.close()

def main():
    # Load data
    df = pd.read_csv("data/restaurant_data.csv")
    
    # EDA
    perform_eda(df)
    
    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Compare models
    compare_models(results)
    
    # Save best model
    save_best_model(results)

if __name__ == "__main__":
    main()