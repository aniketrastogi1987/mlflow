import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Parameters
N_ESTIMATORS = 120
MAX_DEPTH = 8
MAX_FEATURES = 2

# Run name
RUN_NAME = "run-02"

# Set an experiment name, unique and case-sensitive
# It will create a new experiment if the experiment with given doesn't exist
exp = mlflow.set_experiment(experiment_name="Diabetes Experiments")

# Start Run
mlflow.start_run(
    run_name=RUN_NAME,                      # specify name of the run, else it will assign a random name
    experiment_id=exp.experiment_id         # experiment id under which to create the current run
)

# Log parameters
mlflow.log_param("n_estimators", N_ESTIMATORS)
mlflow.log_param("max_depth", MAX_DEPTH)
mlflow.log_param("max_features", MAX_FEATURES)


db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(
    db.data, db.target
)

# Create and train model
rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, max_features=MAX_FEATURES)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test set
predictions = rf.predict(X_test)

# Log performance metrics
mlflow.log_metric("training_r2_score", r2_score(rf.predict(X_train), y_train))
mlflow.log_metric("testing_r2_score", r2_score(predictions, y_test))
mlflow.log_metric("training_mse", mean_squared_error(rf.predict(X_train), y_train))
mlflow.log_metric("testing_mse", mean_squared_error(predictions, y_test))

# Log a scikit-learn model as an MLflow artifact for the current run

# In MLflow, a signature is a way to define the input and output schema of a machine learning model.
# It provides a clear specification of what inputs the model expects and what outputs it produces.
# Using `infer_signature`, you can automatically infer the signature from the training data and the model's predictions.
signature = infer_signature(X_train, rf.predict(X_train))

mlflow.sklearn.log_model(sk_model=rf, artifact_path="trained_model", signature = signature)

mlflow.end_run()