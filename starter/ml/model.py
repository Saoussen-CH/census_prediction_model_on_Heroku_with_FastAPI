from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=20)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_model_metrics_slice(
    model, data, encoder, lb, cat_features, sliced_feature, label
):

    dict_result = {}

    for i in data[sliced_feature].unique():
        data_slice = data[data[sliced_feature] == i]

        X, y, _, _ = process_data(
            data_slice,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, X[:, :103])
        precision, recall, fbeta = compute_model_metrics(y, preds)

        dict_result[i] = {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "sample": len(y),
        }

    return {sliced_feature: dict_result}