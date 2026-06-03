# 🏗️ Design Patterns for Machine Learning Pipelines

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-pink)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-red)
![Design Patterns](https://img.shields.io/badge/Software-Design%20Patterns-success)

A modular Machine Learning framework that demonstrates how classical **Software Design Patterns** can be applied to build scalable, maintainable, and extensible ML systems.

Instead of creating isolated scripts for each model, this project introduces an architecture where machine learning algorithms are treated as interchangeable components, enabling clean experimentation and easy system evolution.

---

## 🎯 Project Motivation

Traditional machine learning projects often become difficult to maintain as more models, preprocessing techniques, and evaluation methods are introduced.

Developers frequently encounter challenges such as:

* Duplicate training logic
* Tight coupling between models and workflows
* Difficult model replacement
* Poor scalability as projects grow

This project addresses these issues by applying proven software engineering principles through Design Patterns.

The goal is to demonstrate how Machine Learning systems can benefit from the same architectural practices used in large-scale software applications.

---

## 🚀 Project Objectives

* Build a reusable ML experimentation framework
* Apply Software Design Patterns in a real ML application
* Support multiple machine learning algorithms through a unified interface
* Compare model performance consistently
* Create an architecture that is easy to extend with new models
* Demonstrate clean separation of concerns

---

## 🧠 Design Patterns Implemented

### 1️⃣ Strategy Pattern

The Strategy Pattern enables interchangeable machine learning algorithms.

Each model implements the same interface:

```python
train()
predict()
evaluate()
```

Implemented strategies:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest
* Artificial Neural Network (ANN)

Because all models follow the same contract, the pipeline can switch algorithms without modifying execution logic.

Example:

```python
strategy.train(X_train, y_train)
strategy.evaluate(X_test, y_test)
```

#### Benefits

* Runtime model replacement
* Open for extension
* Reduced code duplication
* Consistent model behavior

---

### 2️⃣ Factory Pattern

Model creation is centralized using a Factory.

Instead of manually instantiating models throughout the codebase:

```python
model = ModelFactory.create_model("knn")
```

The Factory is responsible for generating the appropriate strategy object.

#### Benefits

* Decouples creation from usage
* Simplifies model management
* Easier future expansion

---

### 3️⃣ Template Method Pattern

The complete machine learning workflow is defined inside the training pipeline.

Workflow:

1. Preprocess Data
2. Train Model
3. Evaluate Model

Every algorithm follows the same process while delegating model-specific behavior to the selected strategy.

Example:

```python
pipeline.run(
    X_train,
    X_test,
    y_train,
    y_test
)
```

#### Benefits

* Standardized experimentation
* Consistent workflow
* Improved maintainability

---

### 4️⃣ Decorator Pattern

A custom decorator tracks execution time.

```python
@track_time
```

The decorator measures how long the pipeline takes to execute without modifying the business logic itself.

#### Benefits

* Separation of concerns
* Reusable monitoring logic
* Cleaner code

---

## 🏛️ System Architecture

```text
                    +----------------+
                    | ModelFactory   |
                    +--------+-------+
                             |
                             v

                +-----------------------+
                | BaseModelStrategy     |
                +-----------------------+
                      ▲    ▲    ▲    ▲
                      |    |    |    |

                 Logistic  KNN  RF   ANN
                 Strategy  Strategy  Strategy

                             |
                             v

                 +----------------------+
                 | TrainingPipeline     |
                 +----------------------+
                           |
                           |
          +----------------+----------------+
          |                |                |
          v                v                v

      Preprocess       Train Model      Evaluate
```

---

## 📁 Project Structure

```text
patternml/
│
├── decorators/
│   └── decorators.py
│
├── factory/
│   └── model_factory.py
│
├── strategies/
│   ├── base_model.py
│   ├── logistic_regression_strategy.py
│   ├── knn_strategy.py
│   ├── random_forest_strategy.py
│   └── ann_strategy.py
│
├── template/
│   └── training_pipeline.py
│
└── main.py
```

---

## 📊 Dataset

This project uses the Digits dataset from Scikit-Learn.

Dataset characteristics:

* 1,797 samples
* 64 numerical features
* 10 target classes
* Handwritten digit recognition task

Each sample represents an 8×8 grayscale image flattened into a feature vector.

Target classes:

```text
0,1,2,3,4,5,6,7,8,9
```

This dataset is commonly used as a benchmark for multi-class classification.

---

## 🤖 Models Implemented

| Model                     | Category                |
| ------------------------- | ----------------------- |
| Logistic Regression       | Linear Classification   |
| K-Nearest Neighbors       | Instance-Based Learning |
| Random Forest             | Ensemble Learning       |
| Artificial Neural Network | Deep Learning           |

---

## ⚙️ Machine Learning Workflow

### Step 1

Load dataset.

### Step 2

Perform stratified train-test split.

### Step 3

Standardize features using:

```python
StandardScaler
```

### Step 4

Create model through the Factory.

### Step 5

Execute training through the Template Pipeline.

### Step 6

Evaluate model performance.

### Step 7

Compare results visually.

---

## 📈 Evaluation Metrics

Each model is evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

### Weighted F1 Score

Balances precision and recall while accounting for class distribution.

Returned format:

```python
{
    "accuracy": accuracy,
    "f1": f1
}
```

---

## 📊 Example Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 0.97     | 0.97     |
| KNN                 | 0.98     | 0.98     |
| Random Forest       | 0.97     | 0.97     |
| ANN                 | 0.98     | 0.98     |

*Results may vary slightly due to randomness and environment settings.*

---

## 📉 Visualization

The framework automatically generates an accuracy comparison chart for all trained models.

This provides a quick overview of relative model performance and helps identify the strongest classifier for the dataset.

---

## 💻 Example Usage

```python
from patternml.factory.model_factory import ModelFactory
from patternml.template.training_pipeline import TrainingPipeline

model = ModelFactory.create_model("random_forest")

pipeline = TrainingPipeline(model)

metrics = pipeline.run(
    X_train,
    X_test,
    y_train,
    y_test
)

print(metrics)
```

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/Maryam-Skaik/design-patterns-ml-pipeline.git
```

Move into the project directory:

```bash
cd design-patterns-ml-pipeline
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

---

## ▶️ Run the Project

```bash
python patternml/main.py
```

---

## 🌟 Engineering Insights

This project demonstrates that Machine Learning systems can benefit significantly from software engineering best practices.

By combining Strategy, Factory, Template Method, and Decorator patterns, the framework achieves:

* Extensibility
* Reusability
* Maintainability
* Modularity
* Separation of Concerns
* Cleaner Experimentation Workflows

Adding a new model requires minimal changes to the existing codebase while preserving the overall workflow.

---

## 📚 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Object-Oriented Programming
* Software Design Patterns
* Machine Learning Engineering
* Model Abstraction
* Pipeline Design
* Neural Networks
* Scikit-Learn Workflows
* TensorFlow/Keras Integration

---

## 👩‍💻 Author

**Maryam Skaik** <br>
Computer Science Graduate<br>
Backend Developer | Machine Learning Engineer | Data Science Enthusiast

---

## 📄 License

This project is intended for educational, learning, and portfolio purposes.
