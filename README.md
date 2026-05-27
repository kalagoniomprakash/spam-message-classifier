# spam-message-classifier
High-performance Spam Classifier using SpaCy word embeddings (nlp.pipe optimized) and scikit-learn LinearSVC. Features a live production UI built with Streamlit.

# ✉️ Smart Message Classifier with SpaCy and Streamlit

An end-to-end Machine Learning pipeline that uses **SpaCy word embeddings** and a **Scikit-Learn LinearSVC** model to classify messages as Spam or Ham. The project includes a live web application built using **Streamlit**.

## Usecase & Optimizations
* **High-Speed Vectorization**: Utilizes SpaCy's optimized `nlp.pipe()` multi-threaded batch processing, resulting in a 5-10x performance boost over typical row-by-row pandas `.apply()` loops.
* **Production Ready**: Implements clean data type casting (`float32`) and automated 1D-to-2D array restructuring for smooth user-input inference.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd spam-message-classifier
   ```

2. **Set up a virtual environment and install packages:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the required SpaCy medium model:**
   ```bash
   python -m spacy download en_core_web_md
   ```

## Running the Project

* **To train the model and save the assets:**
  ```bash
  python train.py
  ```
* **To launch the Streamlit Web App interface:**
  ```bash
  streamlit run app.py
  ```

## Model Evaluation Results
* **Overall Accuracy**: ~95.16%

