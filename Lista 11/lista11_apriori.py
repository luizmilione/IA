# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# Nome do arquivo
arquivo_treino = 'train.csv'

print(f"--- Iniciando Análise Apriori no arquivo: {arquivo_treino} ---")

try:
    # 1. Carregar o DataFrame
    df = pd.read_csv(arquivo_treino)
    print("\nDataFrame original carregado.")

    # 2. Pré-processamento dos Dados para Apriori
    
    # Selecionar colunas relevantes para a análise de padrões
    # Excluindo 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked' conforme suas solicitações anteriores
    # e 'PassengerId' que é um identificador.
    df_apriori = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].copy()
    print(f"\nColunas selecionadas para Apriori: {df_apriori.columns.tolist()}")

    # 2.1 Tratar valores ausentes na coluna 'Age' com a mediana
    if df_apriori['Age'].isnull().any():
        median_age = df_apriori['Age'].median()
        df_apriori['Age'].fillna(median_age, inplace=True)
        print(f"Valores ausentes em 'Age' preenchidos com a mediana ({median_age}).")

    # 2.2 Discretizar colunas
    # Discretizar 'Age'
    age_bins = [0, 12, 18, 35, 60, 100]
    age_labels = ['Age_Crianca', 'Age_Adolescente', 'Age_JovemAdulto', 'Age_AdultoMeiaIdade', 'Age_Idoso']
    df_apriori['Age_Cat'] = pd.cut(df_apriori['Age'], bins=age_bins, labels=age_labels, right=False)
    print("'Age' discretizada em categorias.")

    # Discretizar 'SibSp' (Número de irmãos/cônjuges)
    sibsp_bins = [-1, 0, 1, 10] # -1 para incluir 0 corretamente
    sibsp_labels = ['SibSp_0', 'SibSp_1', 'SibSp_2+']
    df_apriori['SibSp_Cat'] = pd.cut(df_apriori['SibSp'], bins=sibsp_bins, labels=sibsp_labels, right=True)
    print("'SibSp' discretizada em categorias.")

    # Discretizar 'Parch' (Número de pais/filhos)
    parch_bins = [-1, 0, 1, 10] # -1 para incluir 0 corretamente
    parch_labels = ['Parch_0', 'Parch_1', 'Parch_2+']
    df_apriori['Parch_Cat'] = pd.cut(df_apriori['Parch'], bins=parch_bins, labels=parch_labels, right=True)
    print("'Parch' discretizada em categorias.")

    # Converter colunas categóricas existentes e as novas para o formato de string para o one-hot encoding
    df_apriori['Survived'] = df_apriori['Survived'].apply(lambda x: f"Survived_{x}")
    df_apriori['Pclass'] = df_apriori['Pclass'].apply(lambda x: f"Pclass_{x}")
    df_apriori['Sex'] = df_apriori['Sex'].apply(lambda x: f"Sex_{x}")
    
    # Selecionar apenas as colunas que serão transformadas em itens
    df_to_encode = df_apriori[['Survived', 'Pclass', 'Sex', 'Age_Cat', 'SibSp_Cat', 'Parch_Cat']]
    
    # 2.3 Transformar para formato transacional (one-hot encoding)
    df_transacional = pd.get_dummies(df_to_encode, prefix_sep='=')
    # Garantir que os valores sejam booleanos (True/False) para mlxtend
    df_transacional = df_transacional.astype(bool) 

    print("\nDataFrame transformado para formato transacional (one-hot encoded):")
    print(df_transacional.head())
    print(f"Número de itens (colunas) no formato transacional: {df_transacional.shape[1]}")


    # 3. Aplicar o algoritmo Apriori para encontrar conjuntos de itens frequentes
    # Um min_support baixo pode gerar muitos itemsets, um alto pode gerar poucos.
    # Vamos tentar com 0.02 (2% de ocorrência) inicialmente.
    min_support_val = 0.02 
    frequent_itemsets = apriori(df_transacional, min_support=min_support_val, use_colnames=True)
    
    if frequent_itemsets.empty:
        print(f"\nNenhum conjunto de itens frequente encontrado com min_support = {min_support_val}.")
        print("Tente um valor menor para min_support se desejar explorar mais.")
    else:
        print(f"\nEncontrados {len(frequent_itemsets)} conjuntos de itens frequentes com min_support >= {min_support_val}.")
        print(frequent_itemsets.sort_values(by='support', ascending=False).head())

        # 4. Gerar regras de associação
        # Usaremos confiança como métrica principal, com um limiar mínimo.
        min_confidence_val = 0.5 
        regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_val)

        if regras.empty:
            print(f"\nNenhuma regra de associação encontrada com min_confidence >= {min_confidence_val}.")
        else:
            print(f"\nEncontradas {len(regras)} regras de associação com min_confidence >= {min_confidence_val}.")
            
            # Remover colunas 'leverage' e 'conviction' para melhor visualização se desejar
            regras_filtradas = regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

            # 5. Exibir as 3 principais regras por diferentes métricas
            print("\n--- Top 3 Regras por SUPORTE ---")
            top_regras_suporte = regras_filtradas.sort_values(by='support', ascending=False).head(3)
            print(top_regras_suporte)

            print("\n--- Top 3 Regras por CONFIANÇA ---")
            top_regras_confianca = regras_filtradas.sort_values(by='confidence', ascending=False).head(3)
            print(top_regras_confianca)

            print("\n--- Top 3 Regras por LIFT ---")
            top_regras_lift = regras_filtradas.sort_values(by='lift', ascending=False).head(3)
            print(top_regras_lift)
            
            # Explicação das colunas das regras:
            # - antecedents: o item (ou conjunto de itens) à esquerda da regra (Se...).
            # - consequents: o item (ou conjunto de itens) à direita da regra (Então...).
            # - support: a frequência com que os itens aparecem juntos no dataset.
            # - confidence: a probabilidade de o 'consequent' ocorrer dado que o 'antecedent' ocorreu.
            # - lift: indica o quão mais provável é o 'consequent' ocorrer quando o 'antecedent' ocorre,
            #         comparado com a probabilidade normal do 'consequent'.
            #         Lift > 1: associação positiva.
            #         Lift < 1: associação negativa.
            #         Lift = 1: sem associação.

except FileNotFoundError:
    print(f"Erro: O arquivo '{arquivo_treino}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado durante a análise Apriori: {e}")

print("\n--- Análise Apriori Concluída ---")