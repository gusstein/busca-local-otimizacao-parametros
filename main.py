# Importação das bibliotecas necessárias
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import time

# Função para criar um estado com um determinado valor e uma avaliação
def criar_estado(estado):
    return {'estado': estado, 'valor': None}

# Função para gerar os estados vizinhos de um estado atual
def gerar_estados_vizinhos(no_atual):
    # Lista para armazenar os estados vizinhos
    estados_vizinhos = []

    # Gera vizinhos com pequenas alterações em C
    for delta_c in [-0.05, 0.05]:
        c = no_atual['estado']['C'] + delta_c  # Calcula o novo valor de C
        if 0.5 <= c <= 250:
            # Cria um novo estado vizinho com o valor alterado de C e o mesmo valor de degree do estado atual
            vizinho = criar_estado({
                'C': c,
                'degree': no_atual['estado']['degree']
            })
            # Adiciona o estado vizinho à lista de vizinhos
            estados_vizinhos.append(vizinho)

    # Gera vizinhos com pequenas alterações em degree
    for delta_d in [-1, 1]:
        # Calcula o novo valor de degree
        d = no_atual['estado']['degree'] + delta_d
        # Verifica se o novo valor de degree está dentro do intervalo permitido
        if 1 <= d <= 15:
            # Cria um novo estado vizinho com o valor alterado de degree e o mesmo valor de C do estado atual
            vizinho = criar_estado({
                'C': no_atual['estado']['C'],
                'degree': d
            })
            # Adiciona o estado vizinho à lista de vizinhos
            estados_vizinhos.append(vizinho)
    # Retorna a lista de estados vizinhos gerados
    return estados_vizinhos

# Função para avaliar um estado, treinando um SVM com os valores do estado e calculando a acurácia
def avaliar_estado(no, X_train, y_train, X_test, y_test):
    # Cria um objeto SVM com kernel polinomial e os valores de C e degree do estado atual
    svm = SVC(kernel='poly', C=no['estado']['C'], degree=no['estado']['degree'])
    # Treina o modelo SVM com os dados de treinamento
    svm.fit(X_train, y_train)
    # Realiza previsões nos dados de teste
    y_pred = svm.predict(X_test)
    # Calcula a acurácia comparando as previsões com os rótulos reais
    acc = accuracy_score(y_test, y_pred)
    # Atribui a acurácia ao campo 'valor' do estado atual
    no['valor'] = acc

# Função de Random Restart Hill Climbing para otimizar os parâmetros C e degree
def random_restart_hill_climbing(num_reinicios):
    melhor_acuracia = 0.0
    melhor_C = None
    melhor_degree = None

    for _ in range(num_reinicios):
        # Inicializa o nó atual com valores aleatórios dentro dos intervalos permitidos
        no_atual = criar_estado({
            # Valor aleatório para C
            'C': np.random.uniform(0.5, 250),
            # Valor aleatório para degree
            'degree': np.random.randint(1, 15)
        })

        while True:
            # Avalia o estado atual
            avaliar_estado(no_atual, X_train, y_train, X_test, y_test)

            # Verifica se o estado atual é o melhor encontrado até agora
            if no_atual['valor'] > melhor_acuracia:
                # Atualiza a melhor acurácia
                melhor_acuracia = no_atual['valor']
                # Atualiza o melhor valor de C
                melhor_C = no_atual['estado']['C']
                melhor_degree = no_atual['estado']['degree']  # Atualiza o melhor valor de degree

            # Gera os vizinhos do estado atual
            vizinhos = gerar_estados_vizinhos(no_atual)
            estado_vizinho_melhor = False

            # Avalia cada vizinho e atualiza o estado atual se encontrar um vizinho com melhor valor
            for estado_vizinho in vizinhos:
                avaliar_estado(estado_vizinho, X_train, y_train, X_test, y_test)
                if estado_vizinho['valor'] > no_atual['valor']:
                    # Atualiza o estado atual para o vizinho melhor
                    no_atual = estado_vizinho
                    estado_vizinho_melhor = True
                    break

            # Sai do loop se não encontrar um vizinho melhor
            if not estado_vizinho_melhor:
                break

    # Retorna os melhores valores encontrados
    return melhor_C, melhor_degree, melhor_acuracia

if __name__ == '__main__':
    # Carrega os dados do câncer de mama
    cancer = datasets.load_breast_cancer()

    # Separa os atributos descritivos do atributo que descreve as classes do problema
    # atributos descritivos
    X = cancer.data
    # classes - maligna (1) ou benigna (0)
    y = cancer.target

    # Divisão dos dados em treino (70%) e teste (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    while True:
        # Solicita o número de reinícios ao usuário
        numero_reinicios = int(input("Digite o número de reinícios: "))

        # Obtém o tempo de início da busca
        tempo_inicio = time.time()

        # Executa o algoritmo Random Restart Hill Climbing
        melhor_valor_C, melhor_valor_degree, melhor_acuracia = random_restart_hill_climbing(numero_reinicios)
        porcentagem_acuracia = melhor_acuracia * 100

        # Obtém o tempo de término da busca
        tempo_final = time.time()

        # Calcula o tempo total de execução
        tempo_execucao = tempo_final - tempo_inicio

        # Exibe os resultados
        print("--------------------")
        print("Melhor valor de C: ")
        print(melhor_valor_C)
        print("--------------------")
        print("Melhor valor de degree: ")
        print(melhor_valor_degree)
        print("--------------------")
        print("Acurácia obtida: ")
        print(f"{porcentagem_acuracia:.2f}%")
        print("--------------------")
        print(f"Tempo de execução: {tempo_execucao:.2f} segundos")
        print("--------------------")

        # Pergunta se o usuário deseja reiniciar
        reiniciar = input("Deseja reiniciar? (S/N): ")
        if reiniciar.upper() == "N":
            break

