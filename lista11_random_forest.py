# Importar bibliotecas necessárias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

# Nomes dos arquivos
arquivo_treino = 'train.csv'
arquivo_teste = 'test.csv'
arquivo_respostas_teste = 'gender_submission.csv' # Contém PassengerId e Survived para o teste

# Variáveis globais para armazenar objetos do treinamento
modelo_rf_final = None
imputer_geral = None # Alterado para imputer_geral, pois pode tratar mais que 'Age'
colunas_treinamento_final = None

# ----- PARTE 1: TREINAMENTO DO MODELO (repetindo para garantir consistência) -----
print("--- INICIANDO PARTE 1: Treinamento do Modelo ---")
try:
    df_treino_original = pd.read_csv(arquivo_treino)
    print("DataFrame de treino original carregado.")

    # 1.1 Deletar colunas
    colunas_para_deletar = ['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    df_treino_processado = df_treino_original.drop(columns=[col for col in colunas_para_deletar if col in df_treino_original.columns])
    print(f"Colunas deletadas do treino: {', '.join([col for col in colunas_para_deletar if col in df_treino_original.columns])}")

    # 1.2 Converter 'Sex' para binário
    if 'Sex' in df_treino_processado.columns:
        mapeamento_sexo = {'male': 1, 'female': 0}
        df_treino_processado['Sex'] = df_treino_processado['Sex'].map(mapeamento_sexo)
        print("Coluna 'Sex' do treino convertida para binário.")

    # 1.3 Separar features (X) e alvo (y)
    if 'Survived' not in df_treino_processado.columns:
        raise ValueError("Coluna 'Survived' não encontrada no arquivo de treino.")
    
    y_treino = df_treino_processado['Survived']
    X_treino = df_treino_processado.drop(columns=['Survived', 'PassengerId'], errors='ignore') # 'PassengerId' também não é feature
    
    colunas_treinamento_final = X_treino.columns.tolist() # Guardar nomes das colunas ANTES da imputação

    # 1.4 Tratar dados faltantes com SimpleImputer (para colunas numéricas)
    colunas_numericas_treino = X_treino.select_dtypes(include=np.number).columns.tolist()
    colunas_nao_numericas_treino = X_treino.select_dtypes(exclude=np.number).columns.tolist()

    if colunas_numericas_treino:
        imputer_geral = SimpleImputer(strategy='median')
        X_treino_numericas_imputadas = imputer_geral.fit_transform(X_treino[colunas_numericas_treino])
        X_treino_numericas_imputadas_df = pd.DataFrame(X_treino_numericas_imputadas, columns=colunas_numericas_treino, index=X_treino.index)
        
        # Reconstruir X_treino
        X_treino = pd.concat([X_treino_numericas_imputadas_df, X_treino[colunas_nao_numericas_treino]], axis=1)
        # Garantir a ordem original das colunas
        X_treino = X_treino[colunas_treinamento_final]
        print("Dados faltantes em colunas numéricas do treino preenchidos com mediana.")
    
    # Tratar faltantes em não numéricas (se houver e se necessário, com moda)
    for col in colunas_nao_numericas_treino:
        if X_treino[col].isnull().any():
            moda = X_treino[col].mode()[0]
            X_treino[col] = X_treino[col].fillna(moda)
            print(f"Dados faltantes na coluna não-numérica '{col}' do treino preenchidos com a moda ('{moda}').")


    if X_treino.isnull().any().any():
        print("\nALERTA PÓS-IMPUTAÇÃO TREINO: Ainda existem valores NaN em X_treino!")
        print(X_treino.isnull().sum())
    else:
        print("\nNenhum valor NaN restante em X_treino.")

    # 1.5 Treinar o modelo
    modelo_rf_final = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    modelo_rf_final.fit(X_treino, y_treino)
    print("Modelo Random Forest treinado com sucesso!")
    print(f"Colunas usadas para o treinamento: {colunas_treinamento_final}")

except FileNotFoundError:
    print(f"Erro: O arquivo de treino '{arquivo_treino}' não foi encontrado.")
    exit() # Sair se o treino falhar
except Exception as e:
    print(f"Ocorreu um erro inesperado na Parte 1 (Treinamento): {e}")
    exit()
print("--- FIM PARTE 1 ---")

# ----- PARTE 2: PRÉ-PROCESSAMENTO DOS DADOS DE TESTE -----
print("\n--- INICIANDO PARTE 2: Pré-processamento dos Dados de Teste ---")
if modelo_rf_final is None or imputer_geral is None or colunas_treinamento_final is None:
    print("Erro: Objetos de treinamento não foram inicializados. Saindo.")
    exit()
    
try:
    df_teste_original = pd.read_csv(arquivo_teste)
    print("DataFrame de teste original carregado.")
    
    # Guardar PassengerId para referência, se o arquivo de respostas não estiver na mesma ordem
    passenger_ids_teste = df_teste_original['PassengerId'] if 'PassengerId' in df_teste_original else None

    # 2.1 Deletar as MESMAS colunas que no treino
    df_teste_processado = df_teste_original.drop(columns=[col for col in colunas_para_deletar if col in df_teste_original.columns])
    print(f"Colunas deletadas do teste: {', '.join([col for col in colunas_para_deletar if col in df_teste_original.columns])}")

    # 2.2 Converter 'Sex' para binário
    if 'Sex' in df_teste_processado.columns:
        df_teste_processado['Sex'] = df_teste_processado['Sex'].map(mapeamento_sexo) # Usar o mesmo mapeamento
        print("Coluna 'Sex' do teste convertida para binário.")

    # Selecionar apenas as colunas que foram usadas no treinamento (e remover PassengerId se ainda presente)
    X_teste = df_teste_processado.drop(columns=['PassengerId'], errors='ignore')
    
    # Garantir que X_teste tenha as mesmas colunas que X_treino, na mesma ordem
    # Adicionar colunas faltantes (que estavam no treino mas não no teste) com valor 0 ou NaN (para imputação)
    for col in colunas_treinamento_final:
        if col not in X_teste.columns:
            X_teste[col] = 0 # ou np.nan se for para ser imputado
            print(f"Coluna '{col}' adicionada ao conjunto de teste com valor 0.")
    # Remover colunas do teste que não estavam no treino
    X_teste = X_teste[colunas_treinamento_final]
    
    # 2.3 Tratar dados faltantes USANDO O IMPUTER AJUSTADO NO TREINO
    colunas_numericas_teste = X_teste.select_dtypes(include=np.number).columns.tolist()
    colunas_nao_numericas_teste = X_teste.select_dtypes(exclude=np.number).columns.tolist()

    if colunas_numericas_teste:
        X_teste_numericas_imputadas = imputer_geral.transform(X_teste[colunas_numericas_teste]) # USAR TRANSFORM, NÃO FIT_TRANSFORM
        X_teste_numericas_imputadas_df = pd.DataFrame(X_teste_numericas_imputadas, columns=colunas_numericas_teste, index=X_teste.index)
        
        X_teste = pd.concat([X_teste_numericas_imputadas_df, X_teste[colunas_nao_numericas_teste]], axis=1)
        X_teste = X_teste[colunas_treinamento_final] # Reordenar
        print("Dados faltantes em colunas numéricas do teste preenchidos com medianas do treino.")
    
    # Tratar faltantes em não numéricas (se houver e se necessário, com moda DO TREINO)
    # Este passo é mais complexo se a moda precisar ser guardada do treino para cada coluna categórica.
    # No nosso caso, as colunas categóricas ('Embarked') foram removidas. 'Sex' foi mapeada.
    # Se houvesse outras, seria necessário um tratamento mais robusto (ex: SimpleImputer com strategy='most_frequent' ajustado no treino).
    for col in colunas_nao_numericas_teste:
        if X_teste[col].isnull().any():
            # Idealmente, usar moda calculada no TREINO.
            # Se não foi guardada, uma solução simples (mas não ideal) é usar a moda do teste ou uma constante.
            # Para este dataset específico, não devemos ter categóricas com NA após os passos anteriores.
            moda_teste = X_teste[col].mode()[0] # Menos ideal, mas preenche
            X_teste[col] = X_teste[col].fillna(moda_teste)
            print(f"Dados faltantes na coluna não-numérica '{col}' do teste preenchidos com a moda do teste ('{moda_teste}').")

    if X_teste.isnull().any().any():
        print("\nALERTA PÓS-IMPUTAÇÃO TESTE: Ainda existem valores NaN em X_teste!")
        print(X_teste.isnull().sum())
        # Se ainda houver NaNs, uma medida drástica seria preenchê-los com 0 ou outra estratégia.
        X_teste = X_teste.fillna(0) # Preencher quaisquer NaNs remanescentes com 0
        print("NaNs remanescentes em X_teste preenchidos com 0 como último recurso.")

    else:
        print("\nNenhum valor NaN restante em X_teste.")

except FileNotFoundError:
    print(f"Erro: O arquivo de teste '{arquivo_teste}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro inesperado na Parte 2 (Pré-processamento do Teste): {e}")
    exit()
print("--- FIM PARTE 2 ---")

# ----- PARTE 3: CARREGAR RESPOSTAS VERDADEIRAS E FAZER PREVISÕES -----
print("\n--- INICIANDO PARTE 3: Carregar Respostas, Prever e Avaliar ---")
try:
    df_respostas = pd.read_csv(arquivo_respostas_teste)
    print("Arquivo de respostas do teste carregado.")

    # Verificar se as respostas estão alinhadas com os dados de teste.
    # gender_submission.csv geralmente tem 'PassengerId' e 'Survived'.
    # Vamos assumir que a ordem é a mesma ou que podemos usar PassengerId para alinhar.
    # Por simplicidade, se o número de linhas for o mesmo, vamos assumir que estão alinhadas.
    if len(df_respostas) != len(X_teste):
        print("Alerta: O número de respostas não corresponde ao número de amostras de teste. Tentando alinhar por PassengerId se possível.")
        # Se passenger_ids_teste foi salvo e df_respostas tem PassengerId
        if passenger_ids_teste is not None and 'PassengerId' in df_respostas.columns:
            df_respostas_alinhado = pd.DataFrame({'PassengerId': passenger_ids_teste}).merge(df_respostas, on='PassengerId', how='left')
            if df_respostas_alinhado['Survived'].isnull().any():
                raise ValueError("Não foi possível alinhar todas as respostas do teste usando PassengerId. Verifique os arquivos.")
            y_verdadeiro = df_respostas_alinhado['Survived']
            print("Respostas alinhadas com o conjunto de teste usando PassengerId.")
        else:
            raise ValueError("Não é possível alinhar respostas e dados de teste. Verifique os arquivos.")
    else:
        y_verdadeiro = df_respostas['Survived'] # Assumindo que a ordem está correta

    # 3.1 Fazer previsões
    previsoes_rf = modelo_rf_final.predict(X_teste)
    print("Previsões feitas no conjunto de teste.")

    # 3.2 Calcular métricas
    precisao = accuracy_score(y_verdadeiro, previsoes_rf)
    recall = recall_score(y_verdadeiro, previsoes_rf)
    f1 = f1_score(y_verdadeiro, previsoes_rf)

    print("\n--- Métricas de Avaliação do Modelo Random Forest ---")
    print(f"Acurácia (Precisão Global): {precisao:.4f}")
    print(f"Recall (Sensibilidade):     {recall:.4f}")
    print(f"F1-Score:                   {f1:.4f}")

except FileNotFoundError:
    print(f"Erro: O arquivo de respostas '{arquivo_respostas_teste}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado na Parte 3 (Avaliação): {e}")
print("--- FIM PARTE 3 ---")