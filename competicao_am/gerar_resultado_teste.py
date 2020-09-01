from competicao_am.metodo_competicao import MetodoCompeticaoFinal
from competicao_am.preprocessamento_atributos_competicao import DataframePreprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Parametros dos modelos
EXP_COST = 1.8103213437733223
GAMMA = 0.29314287596361527
MIN_SAMPLES_SPLIT = 0.04466339369901853
MAX_FEATURES = 0.0010292429905777445
NUM_ARVORES = 1


def gerar_saida_teste(df_data_to_predict, col_classe, num_grupo):
    """
    Treina o modelo com os dados de treino fornecidos e computa a saída de teste por meio do modelo criado 
    """

    # Métodos usado
    svm_method = SVC(C=2**EXP_COST, gamma=GAMMA, kernel='rbf', random_state=2)
    rf_method = RandomForestClassifier(
        min_samples_split=MIN_SAMPLES_SPLIT, max_features=MAX_FEATURES, n_estimators=NUM_ARVORES, random_state=2)

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
