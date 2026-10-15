"""Train and evaluate the KNN demo model."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"synthetic_poultry_house_data.csv")
features=["Temperature_C","Humidity_pct","Light_Level_lux"]
X,y=df[features],df["Condition"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
pipe=Pipeline([("scaler",StandardScaler()),("knn",KNeighborsClassifier())])
search=GridSearchCV(pipe,{"knn__n_neighbors":[3,5,7,9,11,13],"knn__weights":["uniform","distance"]},cv=5,scoring="accuracy",n_jobs=-1)
search.fit(X_train,y_train)
model=search.best_estimator_
pred=model.predict(X_test)
metrics={"dataset_type":"Synthetic demonstration data","rows":len(df),"train_rows":len(X_train),"test_rows":len(X_test),"class_counts":{str(k):int(v) for k,v in df.Condition.value_counts().items()},"best_parameters":search.best_params_,"test_accuracy":accuracy_score(y_test,pred),"test_precision_macro":precision_score(y_test,pred,average="macro",zero_division=0),"test_recall_macro":recall_score(y_test,pred,average="macro",zero_division=0),"test_f1_macro":f1_score(y_test,pred,average="macro",zero_division=0),"confusion_matrix_labels":list(model.classes_),"confusion_matrix":confusion_matrix(y_test,pred,labels=model.classes_).tolist(),"classification_report":classification_report(y_test,pred,zero_division=0,output_dict=True)}
joblib.dump(model,ROOT/"poultry_knn_model.joblib")
(ROOT/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print(json.dumps(metrics,indent=2))
