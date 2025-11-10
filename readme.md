# AI Tools Assignment: Mastering the AI Toolkit

**Student:** Adeyemi Ayorinde  
**Date:** 10-Nov-2025  

**Live Demo:** [MNIST Streamlit App](https://classical-ml-wjoqpdcrhuc7ufsn8dge3o.streamlit.app/)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Part 1: Theoretical Understanding](#part-1-theoretical-understanding)
3. [Part 2: Practical Implementation](#part-2-practical-implementation)
   - [Task 1: Classical ML (Iris Dataset)](#task-1-classical-ml-iris-dataset)
   - [Task 2: Deep Learning (MNIST CNN)](#task-2-deep-learning-mnist-cnn)
   - [Task 3: NLP with spaCy](#task-3-nlp-with-spacy)
4. [Part 3: Ethics & Optimization](#part-3-ethics--optimization)
5. [Setup Instructions](#setup-instructions)
6. [Dependencies](#dependencies)
7. [Screenshots](#screenshots)
8. [Acknowledgements](#acknowledgements)

---

## Project Overview
This project demonstrates mastery of **AI tools and frameworks** through theoretical and practical applications:
- Classical Machine Learning with **Scikit-learn**
- Deep Learning with **TensorFlow** (MNIST CNN classifier)
- Natural Language Processing (NLP) with **spaCy**
- Deployment of an interactive **Streamlit web app** for MNIST digit classification

The project emphasizes teamwork, ethical AI considerations, and deployment best practices.

---

## Part 1: Theoretical Understanding

**Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?**  
**Answer:**  
- **TensorFlow:** production-focused, static/dynamic computation graphs, verbose syntax, strong deployment tools.  
- **PyTorch:** dynamic computation graphs, Pythonic syntax, widely used in research and prototyping.  
- **Use cases:** TensorFlow for production/deployment, PyTorch for research/prototyping.

**Q2: Describe two use cases for Jupyter Notebooks in AI development.**  
**Answer:**  
1. Exploratory Data Analysis (EDA)  
2. Model Prototyping and interactive experimentation

**Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?**  
**Answer:**  
- Built-in tokenization, POS tagging, dependency parsing  
- Named Entity Recognition (NER)  
- Faster and more accurate than raw Python parsing  
- Supports custom pipelines and rule-based extensions

**Comparative Analysis: Scikit-learn vs TensorFlow**

| Feature | Scikit-learn | TensorFlow |
|---------|--------------|------------|
| Target Applications | Classical ML | Deep Learning |
| Ease of Use | Beginner-friendly | Requires DL knowledge |
| Community Support | Large, mature | Large, growing |

---

## Part 2: Practical Implementation

### Task 1: Classical ML (Iris Dataset)
- Goal: Predict iris species using a Decision Tree classifier  
- Preprocessing: Missing value handling, label encoding  
- Evaluation: Accuracy, Precision, Recall  
- Code: `iris_decision_tree.ipynb`  
- Screenshot placeholder: ![Confusion Matrix](screenshots/iris_confusion_matrix.png)

### Task 2: Deep Learning (MNIST CNN)
- Goal: Classify handwritten digits (>95% test accuracy)  
- Model: Convolutional Neural Network (CNN)  
- Code: `mnist_cnn_model.ipynb`  
- Deployment: Streamlit web app ([Live Demo](https://classical-ml-wjoqpdcrhuc7ufsn8dge3o.streamlit.app/))  
- Screenshot placeholders:  
  - ![Training Graphs](screenshots/mnist_training_graphs.png)  
  - ![Sample Predictions](screenshots/mnist_sample_predictions.png)

### Task 3: NLP with spaCy
- Goal: Named Entity Recognition (NER) and rule-based sentiment analysis  
- Code: `spaCy_ner_sentiment.ipynb`  
- Screenshot placeholders:  
  - ![NER Output](screenshots/ner_entities.png)  
  - ![Sentiment Analysis](screenshots/sentiment_analysis.png)

---

## Part 3: Ethics & Optimization
- **Bias Considerations:**  
  - MNIST: Underrepresented handwriting styles  
  - Reviews: Sarcasm or ambiguous sentiment  
- **Mitigation:**  
  - TensorFlow Fairness Indicators  
  - Rule-based refinements in spaCy  

- **Troubleshooting:**  
  - Corrected CNN dimension mismatches  
  - Verified input shapes, activations, and loss functions  

- **Bonus Task (Optional):**  
  - Deploy MNIST model using Streamlit  
  - Screenshot placeholder: ![Streamlit App](screenshots/mnist_webapp.png)

---

## Setup Instructions
1. Clone the repository:  
```bash
git clone <your-repo-link>
cd <repo-folder>
