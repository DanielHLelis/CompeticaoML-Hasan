from competicao_am.metodo_competicao import MetodoCompeticaoFinal
from competicao_am.preprocessamento_atributos_competicao import DataframePreprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def gerar_saida_teste(df_data_to_predict, col_classe, num_grupo):
    """
    Treina o modelo com os dados de treino fornecidos e computa a saída de teste por meio do modelo criado 
    """

    # Params
    exp_cost = 1.8103213437733223
    gamma = 0.29314287596361527
    min_samples = 0.04411375470350195
    max_features = 0.30427723689993286
    num_arvores = 1

    # Métodos usado
    svm_method = SVC(C=2**exp_cost, gamma=gamma, kernel='rbf', random_state=2)
    rf_method = RandomForestClassifier(
        min_samples_split=min_samples, max_features=max_features, n_estimators=num_arvores, random_state=2)

    # DataFrame de treino
    df_treino = pd.read_csv("datasets/movies_amostra.csv")

    dfp = DataframePreprocessing(df_treino, df_data_to_predict, col_classe)

    # Instanciação do método
    ml_method = MetodoCompeticaoFinal(svm_method, rf_method, dfp)

    # Treinamento e geração dos resultados
    resultado = ml_method.eval()
    arr_final_predictions = resultado.predict_y

    with open(f"predict_grupo_{num_grupo}.txt", "w") as file_predict:
        for predict in arr_final_predictions:
            file_predict.write(ml_method.preprocessor.nom_classe(predict)+"\n")
