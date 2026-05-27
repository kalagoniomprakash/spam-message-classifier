import numpy as np
import pandas as pd
import spacy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def load_and_clean_data(file_path="spam.csv"):
    """Loads dataset, fixes columns, and drops duplicates/nulls."""
    print("Loading and cleaning data...")
    # Read CSV (Handles common encoding issues with text datasets)
    data = pd.read_csv(file_path, encoding='latin-1')
    
    # 1. Drop unnecessary columns if they exist
    cols_to_drop = [col for col in ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'] if col in data.columns]
    if cols_to_drop:
        data.drop(columns=cols_to_drop, inplace=True)
        
    # 2. Rename columns to production standards
    data.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    
    # 3. Drop duplicate rows using hashing for accuracy
    data['row_hash'] = pd.util.hash_pandas_object(data, index=False)
    data.drop_duplicates(subset=['row_hash'], keep='first', inplace=True)
    data.drop(columns=['row_hash'], inplace=True)
    
    # 4. Handle missing data safely
    data.dropna(subset=['text', 'label'], inplace=True)
    data['text'] = data['text'].astype(str)
    
    return data

def main():
    # Load SpaCy medium model for 300-dimensional word vectors
    print("Loading NLP model (en_core_web_md)...")
    nlp = spacy.load("en_core_web_md")
    
    # Process data (Ensure you have your dataset saved as 'spam.csv' in the same folder)
    try:
        data = load_and_clean_data("spam.csv")
    except FileNotFoundError:
        print("Error: 'spam.csv' not found. Please place your dataset in this directory.")
        return

    # Split data into train and test sets
    X = data['text']
    Y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # FAST VECTORIZATION: Uses multi-threaded nlp.pipe with unnecessary components disabled
    print("Vectorizing text data into 2D matrices (Optimized Batch Mode)...")
    X_Vec = np.array([doc.vector for doc in nlp.pipe(x_train, disable=["parser", "tagger", "ner"])]).astype(np.float32)
    x_test_vec = np.array([doc.vector for doc in nlp.pipe(x_test, disable=["parser", "tagger", "ner"])]).astype(np.float32)

    # Encode categorical target text labels into integers (0 and 1)
    print("Encoding target labels...")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Train Linear Support Vector Classifier
    print("Training the LinearSVC model...")
    model = LinearSVC(dual="auto") 
    model.fit(X_Vec, y_train_encoded)

    # Evaluate the trained model
    print("\n Evaluating model performance...")
    predictions = model.predict(x_test_vec)
    
    print("\n" + "="*30)
    print("       MODEL PERFORMANCE       ")
    print("="*30)
    print(f"Accuracy Score: {accuracy_score(y_test_encoded, predictions):.2%}\n")
    print(classification_report(y_test_encoded, predictions, target_names=le.classes_))
    print("="*30)

    # Save the trained model configurations for deployment
    print("\n Saving model components to disk...")
    joblib.dump(model, 'spam_classifier_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("Done! 'spam_classifier_model.pkl' and 'label_encoder.pkl' are ready for your Streamlit app.")

if __name__ == "__main__":
    main()
