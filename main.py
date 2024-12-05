import argparse
from Trip_duration_predictor_utils_data import load_data ,Get_preprocessor,Get_feature
from Trip_duration_predictor_utils_model import Get_model ,predict_eval,save_the_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trip Duration')

    parser.add_argument('--train_path',type=str,default=r'Y:\01 ML\Projects\02 Trip Duration Prediction\Data\train.csv',help='train_path')
    parser.add_argument('--val_path',type=str,default=r"Y:\01 ML\Projects\02 Trip Duration Prediction\Data\val.csv",help='val_path')

    parser.add_argument('--preprocessor',type=int,default=2,help='1 for MinMaxScaler & 2 for standardScaler')
    parser.add_argument('--poly_degree',type=int,default=2,help='Degree of Polynomial feature')

    parser.add_argument('--model',type=int,default=1,help='1 for Ridge')

    args=parser.parse_args()

    # Load_data and prepare_data
    train,val,kmeans=load_data(args.train_path,args.val_path)

    # Preprocessing
    Preprocessing =Get_preprocessor(args.preprocessor)

    # train the model
    feature=Get_feature()
    get_model=Get_model(args.model)
    pipeline = Pipeline(steps=[
        ('ohe', Preprocessing),
        ('poly', PolynomialFeatures(degree=args.poly_degree)),
        ('regression', get_model)
    ])
    model=pipeline.fit(train[feature],train.log_trip_duration)

    # Evaluation
    predict_eval(model,train,feature,'train')
    predict_eval(model,val,feature,'val')

    # save the model
    save_the_model(model,kmeans)













