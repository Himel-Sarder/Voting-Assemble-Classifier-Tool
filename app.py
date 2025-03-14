from flask import Flask, render_template, request, redirect, url_for, flash
import os
from flask_wtf.csrf import generate_csrf
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Required for CSRF protection
csrf = CSRFProtect(app)
app.config['UPLOAD_FOLDER'] = 'datasets'

print("NumPy Version:", np.__version__)
print("Pandas Version:", pd.__version__)
print("Scikit-Learn Version:", sklearn.__version__)

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory storage for datasets (replace with a database in production)
datasets = []

@app.route('/')
def index():
    return redirect(url_for('select_models'))

@app.route('/select_models', methods=['GET', 'POST'])
def select_models():
    return render_template('select_models.html', datasets=datasets)

@app.route('/upload_dataset', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        name = request.form.get('name')
        file = request.files.get('file')

        if file and file.filename.endswith('.csv'):
            # Save the file to the upload folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Store dataset information (replace with a database in production)
            datasets.append({'name': name, 'path': file_path})

            flash('Dataset uploaded successfully!', 'success')
            return redirect(url_for('select_models'))
        else:
            flash('Invalid file. Please upload a CSV file.', 'error')

    return render_template('upload_dataset.html')

@app.route('/perform_voting', methods=['POST'])
def perform_voting():
    dataset_id = int(request.form.get('dataset'))
    selected_models = request.form.getlist('models')
    voting_type = request.form.get('voting_type')

    if not selected_models:
        return render_template('results.html', error='No models selected. Please select at least one model.')

    try:
        dataset = datasets[dataset_id]
        df = pd.read_csv(dataset['path'])
    except (IndexError, FileNotFoundError):
        return render_template('results.html', error='Dataset not found!')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_map = {
        "logistic_regression": ('Logistic Regression', LogisticRegression(
            max_iter=int(request.form.get('lr_max_iter', 100)),
            C=float(request.form.get('lr_c', 1.0))
        )),
        "random_forest": ('Random Forest', RandomForestClassifier(
            n_estimators=int(request.form.get('rf_n_estimators', 100)),
            max_depth=int(request.form.get('rf_max_depth', 5)),
            random_state=42
        )),
        "svm": ('SVM', SVC(
            C=float(request.form.get('svm_c', 1.0)),
            kernel=request.form.get('svm_kernel', 'rbf'),
            probability=True
        )),
        "k_nearest_neighbors": ('KNN', KNeighborsClassifier(
            n_neighbors=int(request.form.get('knn_n_neighbors', 5))
        )),
        "decision_tree": ('Decision Tree', DecisionTreeClassifier(
            max_depth=int(request.form.get('dt_max_depth', 3)),
            random_state=42
        )),
    }
    models = [model_map[m] for m in selected_models if m in model_map]

    model_accuracies = {}
    for name, model in models:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_accuracies[name] = accuracy_score(y_test, y_pred) * 100
        except Exception as e:
            print(f"Error fitting {name}: {e}")
            model_accuracies[name] = f"Error: {e}"

    if voting_type in ['hard', 'soft']:
        try:
            voting_clf = VotingClassifier(estimators=models, voting=voting_type)
            voting_clf.fit(X_train, y_train)
            y_pred = voting_clf.predict(X_test)
            voting_accuracy = accuracy_score(y_test, y_pred) * 100
        except Exception as e:
            print(f"Error creating Voting Classifier: {e}")
            voting_accuracy = f"Error: {e}"
    else:
        voting_accuracy = None

    print("Model Accuracies:", model_accuracies)
    print("Voting Accuracy:", voting_accuracy)

    return render_template('results.html', model_accuracies=model_accuracies, voting_accuracy=voting_accuracy)

@app.route('/delete_dataset/<int:dataset_id>', methods=['POST', 'DELETE'])
def delete_dataset(dataset_id):
    try:
        dataset = datasets.pop(dataset_id)
        os.remove(dataset['path'])
        return '', 204  # No content (success)
    except (IndexError, FileNotFoundError):
        return '', 404  # Not found

if __name__ == '__main__':
    app.run(debug=True)