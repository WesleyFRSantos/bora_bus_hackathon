import pandas as pd
import pickle
# from sklearn.linear_model import LogisticRegression TROCAR POR ALGORITMO DE CLASSIFICAÇÃO / MLFLOW (?)
from sklearn.engineering_pipeline import FeatureEngineering

def main():

    # SPLIT DATASET

    df = pd.read_csv('../data/raw/BoraBusTratado.csv')
    df_teste_final = df.sample(frac=0.1)
    df_modelagem = df.drop(df_teste_final.index)

    X_train = df_modelagem.drop(['SatisfacaoGeral'], axis=1)
    y_train = df_modelagem['SatisfacaoGeral']

    X_test = pd.read_csv('data/test.csv')

    features = ['Pclass', 'Age', 'Sex', 'Fare']
    X_train = X_train[features]
    X_test = X_test[features]

    feature_engineering_pipeline = FeatureEngineering().get_pipeline()
    X_train = feature_engineering_pipeline.fit_transform(X_train)

    X_test = feature_engineering_pipeline.transform(X_test)

    X_train.to_csv('data/train_after_feature_engineering.csv', index=False)
    X_test.to_csv('data/test_after_feature_engineering.csv', index=False)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    with open('models/model.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


if __name__ == '__main__':
    main()