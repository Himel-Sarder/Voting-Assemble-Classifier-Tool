# Voting Ensemble Classifier Project

### Introduction
This project is a **web-based application** built using Flask that allows users to upload datasets, select machine learning models, and perform **ensemble voting** to improve prediction accuracy. The application leverages the power of **Scikit-learn** to implement multiple classifiers and combines their predictions using either **hard voting** or **soft voting**.

The goal of this project is to provide an intuitive interface for users to experiment with ensemble learning techniques without needing to write code. It is particularly useful for educational purposes, quick prototyping, or comparing the performance of different machine learning models.

## 🟢 Live : https://voting-assemble-classifier-tool.onrender.com
### Screenshots
![image](https://github.com/user-attachments/assets/f25f94a9-a462-4d19-ac36-bd6fb73a231a)
![image](https://github.com/user-attachments/assets/c82ceff3-ef67-42d3-9e9d-e04ca5076884)
![image](https://github.com/user-attachments/assets/df909412-6045-4d4c-a378-57bc65cb2b23)
![image](https://github.com/user-attachments/assets/b81ad9d0-9d4e-4659-bed3-cea3c0105587)
![image](https://github.com/user-attachments/assets/07f8ac24-7323-40c4-95fc-22484d2435b7)
![image](https://github.com/user-attachments/assets/ecf2ad80-393f-4e6d-8974-013e477c8f1e)

### Key Features

#### 1. **Dataset Upload**
   - Users can upload a CSV file containing their dataset.
   - The dataset must have **numerical features**, and the **target column** (the variable to predict) must be the last column.
   - Uploaded datasets are stored temporarily for processing.

#### 2. **Model Selection**
   - Users can select from the following machine learning models:
     - **Logistic Regression**
     - **Random Forest**
     - **Support Vector Machine (SVM)**
     - **K-Nearest Neighbors (KNN)**
     - **Decision Tree**
   - Multiple models can be selected simultaneously for ensemble learning.

#### 3. **Hyperparameter Tuning**
   - Users can customize hyperparameters for each selected model:
     - **Logistic Regression**: Maximum iterations, regularization strength (`C`).
     - **Random Forest**: Number of trees, maximum depth.
     - **SVM**: Regularization parameter (`C`), kernel type.
     - **KNN**: Number of neighbors.
     - **Decision Tree**: Maximum depth.

#### 4. **Ensemble Voting**
   - Users can choose between **hard voting** and **soft voting**:
     - **Hard Voting**: The final prediction is based on the majority vote of the individual models.
     - **Soft Voting**: The final prediction is based on the average predicted probabilities of the individual models.

#### 5. **Results Display**
   - The application displays the **accuracy** of each individual model.
   - It also shows the **accuracy of the ensemble voting**.
   - Results are presented in a clear and user-friendly format.

#### 6. **Dataset Management**
   - Users can delete uploaded datasets to free up space or remove outdated data.

---

### Technical Details

#### Backend
- **Flask**: The web framework used to handle routing, requests, and responses.
- **Scikit-learn**: Used for implementing machine learning models and ensemble voting.
- **Pandas**: Used for data preprocessing and handling CSV files.
- **NumPy**: Used for numerical computations.
- **Flask-WTF**: Used for form handling and CSRF protection.

#### Frontend
- **HTML/CSS**: Used for structuring and styling the web pages.
- **Bootstrap**: Used for responsive design and pre-built UI components.
- **JavaScript**: Used for interactive features like clearing file inputs.

#### Deployment
- **Gunicorn**: Used as the production server for running the Flask application.
- **Render**: The platform used for deploying the application (can be replaced with other platforms like Heroku, AWS, etc.).

---

### Workflow

1. **Upload Dataset**:
   - The user uploads a CSV file through the **Upload Dataset** page.
   - The file is saved to the server, and its metadata is stored for future use.

2. **Select Models**:
   - The user navigates to the **Select Models** page.
   - They choose the models they want to include in the ensemble and customize hyperparameters (optional).

3. **Perform Voting**:
   - The user selects the voting type (hard or soft) and clicks **Run Voting Ensemble**.
   - The application preprocesses the data, trains the selected models, and performs ensemble voting.

4. **View Results**:
   - The results are displayed on the **Results** page, showing the accuracy of each model and the ensemble.

5. **Delete Dataset**:
   - The user can delete a dataset from the server using the **Delete** button.

---

### Data Preprocessing
The application performs the following preprocessing steps on the uploaded dataset:
1. **Imputation**: Missing values are filled using the median strategy.
2. **Scaling**: Features are standardized using `StandardScaler`.
3. **Encoding**: The target column is encoded using `LabelEncoder` if it contains categorical values.

---

### Machine Learning Models
The following models are implemented in the application:

#### 1. **Logistic Regression**
   - A linear model used for binary or multi-class classification.
   - Hyperparameters:
     - `max_iter`: Maximum number of iterations.
     - `C`: Inverse of regularization strength.

#### 2. **Random Forest**
   - An ensemble of decision trees for classification.
   - Hyperparameters:
     - `n_estimators`: Number of trees in the forest.
     - `max_depth`: Maximum depth of each tree.

#### 3. **Support Vector Machine (SVM)**
   - A powerful model for classification tasks.
   - Hyperparameters:
     - `C`: Regularization parameter.
     - `kernel`: Type of kernel (e.g., linear, RBF).

#### 4. **K-Nearest Neighbors (KNN)**
   - A non-parametric model that classifies data points based on their neighbors.
   - Hyperparameters:
     - `n_neighbors`: Number of neighbors to consider.

#### 5. **Decision Tree**
   - A tree-based model for classification.
   - Hyperparameters:
     - `max_depth`: Maximum depth of the tree.

---

### Ensemble Voting
The application combines the predictions of the selected models using one of the following techniques:
1. **Hard Voting**:
   - The final prediction is the mode of the predictions made by individual models.
2. **Soft Voting**:
   - The final prediction is based on the average predicted probabilities of the individual models.

---

### Deployment
The application is deployed using **Gunicorn** as the production server and hosted on **Render**. The deployment process involves:
1. Installing dependencies from `requirements.txt`.
2. Running the application using the command:
   ```bash
   gunicorn app:app
   ```

---

### Future Enhancements
- **Support for Regression**: Extend the application to support regression tasks.
- **Advanced Preprocessing**: Add more preprocessing options (e.g., feature selection, outlier removal).
- **Model Persistence**: Save trained models for future use.
- **Visualizations**: Add charts and graphs to visualize model performance and predictions.
