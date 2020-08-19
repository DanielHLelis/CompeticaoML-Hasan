from competicao_am.metodo_competicao import MetodoCompeticao
from sklearn.svm import LinearSVC
import pandas as pd


def gerar_saida_teste(df_data_to_predict, col_classe, num_grupo):
    """
    Treina o modelo com os dados de treino fornecidos e computa a saída de teste por meio do modelo criado 
    """

    # Método usado
    scikit_method = LinearSVC(C=7, random_state=2)

    # DataFrame de treino
    df_treino = pd.read_csv("datasets/movies_amostra.csv")

    # Instanciação do método
    ml_method = MetodoCompeticao(
        scikit_method, df_treino=df_treino, df_data_to_predict=df_data_to_predict, col_classe=col_classe)

    # Treinamento e geração dos resultados
    resultado = ml_method.eval()
    arr_final_predictions = resultado.predict_y

    with open(f"predict_grupo_{num_grupo}.txt", "w") as file_predict:
        for predict in arr_final_predictions:
            file_predict.write(ml_method.preprocessor.nom_classe(predict)+"\n")


def resultado_teste(df_data_to_predict, col_classe):
    scikit_method = LinearSVC(C=7, random_state=2)

    df_treino = pd.read_csv("datasets/movies_amostra.csv")

    ml_method = MetodoCompeticao(
        scikit_method, df_treino=df_treino, df_data_to_predict=df_data_to_predict, col_classe=col_classe)

    resultado = ml_method.eval()
    return resultado
