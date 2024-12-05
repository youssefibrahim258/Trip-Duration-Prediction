import os
import pickle
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import Ridge


def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred, squared=False)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")

def Get_model(option):
    if option == 1:
        return Ridge(alpha=50)


def save_the_model(model,kmeans):
    root_dir = r"Y:\01 ML\Projects\02 Trip Duration Prediction\Last_finish"
    with open(os.path.join(root_dir, 'kmeans_model.pkl'), 'wb') as file:
        pickle.dump(kmeans, file)

    # Save the trained Ridge model into a pickle file
    with open(os.path.join(root_dir, 'model.pkl'), 'wb') as file:
        pickle.dump(model, file)
