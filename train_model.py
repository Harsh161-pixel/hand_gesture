import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

if not os.path.exists("gestures.csv"):
    print("No gestures.csv found")
    exit()

print("Loading Data...")
df = pd.read_csv("gestures.csv")

X = df.drop(columns = ['label'])
y = df['label']

print(f"Total samples: {len(df)}")
print(f"gesture found: {y.unique()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"model training complete")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\n Detailed information: ")
print(classification_report(y_test, y_pred))

with open("gestures_model.pkl", "wb" ) as f:
    pickle.dump(model, f)

print("\n model saved")
print("\n you can now use this model in your main app")
