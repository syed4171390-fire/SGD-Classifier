# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights and bias

2.Choose a loss function

3.Set hyperparameters (learning rate, regularization, etc.)

4.Shuffle training data

5.For each epoch:

6.For each training sample:

7.Compute gradient of loss

8.Update weights using gradient descent

9.Apply regularization

10.Check for convergence or stop after max iterations

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Syed Shamsheer Ali
RegisterNumber:  212225220114
*/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```


## Output:
<img width="498" height="328" alt="image" src="https://github.com/user-attachments/assets/873aea11-254a-4087-b0c5-5c6e4147f472" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
