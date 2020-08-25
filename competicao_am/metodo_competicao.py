from base_am.metodo import MetodoAprendizadoDeMaquina
import pandas as pd
from .preprocessamento_atributos_competicao import DataframePreprocessing
from base_am.resultado import Resultado
from typing import Union, List, Tuple
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from IPython.display import display

# TODO:
# [] Implement multi ml_method support;
# [] Verify unused properties;


class MetodoCompeticao(MetodoAprendizadoDeMaquina):
    def __init__(self, ml_method: Union[ClassifierMixin, RegressorMixin], preprocessor: DataframePreprocessing = None, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None):
        self.ml_method = ml_method
        self.preprocessor = preprocessor

        if df_treino is not None and df_data_to_predict is not None and col_classe is not None and preprocessor is None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

    # Properties

    @property
    def df_treino(self):
        return self.preprocessor.df_treino

    @df_treino.setter
    def df_treino(self, x):
        self.preprocessor.df_treino = x

    @property
    def df_data_to_predict(self):
        return self.preprocessor.df_data_to_predict

    @df_data_to_predict.setter
    def df_data_to_predict(self, x):
        self.preprocessor.df_data_to_predict = x

    @property
    def col_classe(self):
        return self.preprocessor.col_classe

    @property
    def x(self):
        return self.preprocessor.x

    @property
    def y(self):
        return self.preprocessor.y

    # Combiner

    def combine_predictions(self, *predictions: Tuple[List[int]]) -> List[int]:
        y_final_predictions = []

        if len(predictions) < 1:
            raise ValueError("Predictins can't be empty")

        for i in range(len(predictions[0])):
            comedy = 0
            action = 0

            for prediction in predictions:
                if self.preprocessor.nom_classe(prediction[i]) == 'Comedy':
                    comedy += 1
                else:
                    action += 1

            if comedy > action:
                y_final_predictions.append(
                    self.preprocessor.int_classe('Comedy'))
            elif comedy == action:
                y_final_predictions.append(predictions[0][i])
            else:
                y_final_predictions.append(
                    self.preprocessor.int_classe('Action'))

        return y_final_predictions

    # Evals

    def _eval_boi(self, items: List[str], min_occur: int = 3, seed: int = 1):
        y_treino, y_to_predict = self.y

        df_treino_boi, df_to_predict_boi = self.preprocessor.bag_of_items(
            items, min_occur)
        df_treino_boi, df_to_predict_boi = DataframePreprocessing.remove_id(
            df_treino_boi, df_to_predict_boi)

        # ? Should the seed go here?
        self.ml_method.fit(df_treino_boi, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_boi)

        return y_to_predict, arr_predict

    def _eval_bow(self, column: str, seed: int = 1):
        y_treino, y_to_predict = self.y

        df_treino_bow, df_to_predict_bow = self.preprocessor.bag_of_words(
            column)

        # ? Should the seed go here?
        self.ml_method.fit(df_treino_bow, y_treino)
        arr_predict = self.ml_method.predict(df_to_predict_bow)

        return y_to_predict, arr_predict

    def eval(self, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1) -> Resultado:
        if df_treino is not None and df_data_to_predict is not None and col_classe is not None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

        # Dataframes

        dataframes = self.preprocessor.generate_dataframes()

        y_treino, y_to_predict = self.y

        def fit_and_predict(_x_treino, _x_to_predict):
            # _x_treino.to_csv('bruh.csv')
            self.ml_method.fit(_x_treino, y_treino)
            predicted = self.ml_method.predict(_x_to_predict)

            return predicted

        final_predictions = self.combine_predictions(fit_and_predict(
            *dataframes['bow_cru']), fit_and_predict(*dataframes['boi_cru']))  # Ordem indica prioridade

        # final_predictions = fit_and_predict(*dataframes['completo'])

        return Resultado(y_to_predict, final_predictions)


class MetodoCompeticaoValidacao(MetodoCompeticao):
    def base_eval(self, dfs: List[str], df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1):
        if df_treino is not None and df_data_to_predict is not None and col_classe is not None:
            self.preprocessor = DataframePreprocessing(
                df_treino, df_data_to_predict, col_classe)

        # Dataframes

        dataframes = self.preprocessor.generate_dataframes()

        y_treino, y_to_predict = self.y

        def fit_and_predict(_x_treino, _x_to_predict):
            # _x_treino.to_csv('bruh.csv')
            self.ml_method.fit(_x_treino, y_treino)
            predicted = self.ml_method.predict(_x_to_predict)

            return predicted

        predictions = [fit_and_predict(*dataframes[df_name])
                       for df_name in dfs]

        final_predictions = self.combine_predictions(
            *predictions)  # Ordem indica prioridade

        # final_predictions = fit_and_predict(*dataframes['completo'])

        return Resultado(y_to_predict, final_predictions)


class MetodoCompeticaoValidacao_A(MetodoCompeticaoValidacao):
    def eval(self, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1) -> Resultado:
        return self.base_eval(['bow_cru', 'boi_cru'], df_treino, df_data_to_predict, col_classe, seed)


class MetodoCompeticaoValidacao_B(MetodoCompeticaoValidacao):
    def eval(self, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1) -> Resultado:
        return self.base_eval(['bow', 'boi'], df_treino, df_data_to_predict, col_classe, seed)


class MetodoCompeticaoValidacao_C(MetodoCompeticaoValidacao):
    def eval(self, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1) -> Resultado:
        return self.base_eval(['completo'], df_treino, df_data_to_predict, col_classe, seed)


class MetodoCompeticaoValidacao_D(MetodoCompeticaoValidacao):
    def eval(self, df_treino: pd.DataFrame = None, df_data_to_predict: pd.DataFrame = None, col_classe: str = None, seed: int = 1) -> Resultado:
        return self.base_eval(['bow', 'limpo', 'boi'], df_treino, df_data_to_predict, col_classe, seed)
