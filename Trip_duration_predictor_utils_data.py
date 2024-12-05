import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.cluster import MiniBatchKMeans


def feature_creation(train):
    def haversine_array(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    train['distance_haversine'] = haversine_array(train['pickup_latitude'].values,
                                                  train['pickup_longitude'].values,
                                                  train['dropoff_latitude'].values,
                                                  train['dropoff_longitude'].values)

    # Function to calculate the bearing between two points
    def bearing_array(lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371  # in km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    train['bearing'] = bearing_array(train['pickup_latitude'].values,
                                     train['pickup_longitude'].values,
                                     train['dropoff_latitude'].values,
                                     train['dropoff_longitude'].values)

    # Function to calculate a dummy Manhattan distance
    def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
        a = haversine_array(lat1, lng1, lat1, lng2)
        b = haversine_array(lat1, lng1, lat2, lng1)
        return a + b

    train['distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values,
                                                                 train['pickup_longitude'].values,
                                                                 train['dropoff_latitude'].values,
                                                                 train['dropoff_longitude'].values)

    train['distance_haversine'] = np.log1p(train.distance_haversine)
    train['distance_dummy_manhattan'] = np.log1p(train.distance_dummy_manhattan)
    train['log_trip_duration'] = np.log1p(train.trip_duration)

    train.drop(columns=['trip_duration', 'pickup_datetime'], inplace=True)


    return train

def feature_extraction(train):
    train.drop(columns=['id'], inplace=True)

    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['DayofMonth'] = train['pickup_datetime'].dt.day
    train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
    train['month'] = train['pickup_datetime'].dt.month
    train['hour'] = train['pickup_datetime'].dt.hour
    train['dayofyear'] = train['pickup_datetime'].dt.dayofyear
    return train
def CleanData(train):
    m = np.mean(train['trip_duration'])
    s = np.std(train['trip_duration'])
    train = train[train['trip_duration'] <= m + 2 * s]
    train = train[train['trip_duration'] >= m - 2 * s]

    train = train[train['pickup_longitude'] <= -73.75]
    train = train[train['pickup_longitude'] >= -74.03]
    train = train[train['pickup_latitude'] <= 40.85]
    train = train[train['pickup_latitude'] >= 40.63]
    train = train[train['dropoff_longitude'] <= -73.75]
    train = train[train['dropoff_longitude'] >= -74.03]
    train = train[train['dropoff_latitude'] <= 40.85]
    train = train[train['dropoff_latitude'] >= 40.63]

    train[train['passenger_count'] == 0] = np.nan
    train.dropna(axis=0, inplace=True)
    return train

def cluster_features(train, n=10, random_state=42):
    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                        train[['dropoff_latitude', 'dropoff_longitude']].values))

    np.random.seed(random_state)
    sample_ind = np.random.permutation(len(coords))[:500000]

    kmeans = MiniBatchKMeans(n_clusters=n, batch_size=10000, random_state=random_state).fit(coords[sample_ind])

    train['pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
    train['dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])

    return train, kmeans


def load_data(train_path,val_path):
    train=pd.read_csv(train_path)
    val=pd.read_csv(val_path)

    train = CleanData(train)

    train, kmeans = cluster_features(train, n=100, random_state=42)
    val['pickup_cluster'] = kmeans.predict(val[['pickup_latitude', 'pickup_longitude']])
    val['dropoff_cluster'] = kmeans.predict(val[['dropoff_latitude', 'dropoff_longitude']])

    train = feature_extraction(train)
    train = feature_creation(train)
    val = feature_extraction(val)
    val = feature_creation(val)

    return train,val,kmeans


def Choose_preprocessor(option):
    if option == 1 :
        return MinMaxScaler()
    elif option ==2 :
        return StandardScaler()

def Get_preprocessor(option):
    numeric_features = ['distance_haversine','distance_dummy_manhattan','bearing']
    categorical_features = ['passenger_count','store_and_fwd_flag','vendor_id','pickup_cluster', 'dropoff_cluster',
                            'DayofMonth','dayofweek','month','hour','dayofyear']
    train_features = categorical_features + numeric_features



    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', Choose_preprocessor(option), numeric_features)
        ]
        , remainder = 'passthrough'
    )
    return column_transformer

def Get_feature():
    numeric_features = ['distance_haversine', 'distance_dummy_manhattan', 'bearing']
    categorical_features = ['passenger_count', 'store_and_fwd_flag', 'vendor_id', 'pickup_cluster', 'dropoff_cluster',
                            'DayofMonth', 'dayofweek', 'month', 'hour', 'dayofyear']
    train_features = categorical_features + numeric_features

    return train_features


