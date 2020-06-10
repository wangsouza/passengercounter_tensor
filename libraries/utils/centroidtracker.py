from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        #Inicializa o próximo único ID do objeto juntamente com 2 dicionários
        # ordenados usados para acompanhar o mapeamento de um determinado ID de
        # objeto para o seu centróide e o número de quadros consecutivos em que
        # ele foi marcado como "desaparecido", respectivamente.
        self.nextObjectID = 0 #Salva o próximo ID único
        self.objects = OrderedDict() #Salva o ID do objeto em relação ao centróide
        self.disappeared = OrderedDict() #Número de frames consecutivos que deixou
                                        # de detectar.

        #Armazena o número máximo de frames consecutivos que um determinado objeto
        # pode ser marcado como "desaparecido" até que seja necessário cancelar o
        # registro do objeto do rastreamento.
        self.maxDisappeared = maxDisappeared

        #Armazena a distância máxima entre centróides para associar a um objeto -
        # se a distância for maior que essa distância máxima, começaremos a marcar
        # o objeto como "desaparecido".
        self.maxDistance = maxDistance

    def register(self, centroid):
        #Ao registrar um objeto, usamos o próximo ID de objeto disponível para
        # armazenar o centróide.
        self.objects[self.nextObjectID] = centroid #Salva a informação do centróide
                                                   # na lista objetos, vinculado ao
                                                   # próximo ID do objeto.
        self.disappeared[self.nextObjectID] = 0 #Lista onde é armazenada a princípio
                                                # com zero, a informação no número de
                                                # frames consecutivos que deixou de
                                                # detectar e vinculá-lo ao próximo ID
                                                #  do objeto.
        self.nextObjectID += 1 #Incrementa o próximo ID do objeto.

    def deregister(self, objectID):
        #Para desregistrar o ID do objeto, os mesmos são deletados de ambos os
        # dicionários.
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        #Verifica se a lista de retângulos da caixa delimitadora (bounding box) de
        # entrada está vazia.
        if len(rects) == 0:
            #Loop sobre qualquer objeto rastreado existente e o marca como desaparecido.
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                #Se nós atingirmos o número máximo de frames consecutivos em que um
                # determinado objeto foi marcado como ausente, cancele o registro.
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            #Retorna mais cedo, pois não há centróides ou informações de rastreamento
            # para atualizar.
            return self.objects

        #Inicializa um array de centróides de entrada para o frame atual.
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        #Loop sobre os retângulos das caixas delimitadoras
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            #Usa as coordenadas da caixa delimitadora para derivar o centróide.
            #Ou seja, define a posição central do objeto (somando a posição inicial
            # e final de cada eixo e dividindo por 2).
            cX = int((startX + endX) / 2.0) #Coordenada do centróide no eixo X
            cY = int((startY + endY) / 2.0) #Coordenada do centróide no eixo Y
            inputCentroids[i] = (cX, cY) #Adiciona na lista inputCentroids as
            # entradas cX e cY.

        #Se atualmente não estamos rastreando nenhum objeto, pega os centróides
        # de entrada e registra cada um deles.
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        #Caso contrário, estão rastreando objetos no momento, precisamos tentar
        # corresponder os centróides de entrada aos centróides de dados existentes.
        else:
            #Pega o conjunto de IDs dos objetos e centróides correspondentes.
            objectIDs = list(self.objects.keys()) #Parâmetro key, está relacionado com o
                                                 # índice único do objeto, ou seja, a
                                                 # posição na lista.
            objectCentroids = list(self.objects.values()) #Parâmetro value, está relacionado
                                                          # com o valor dos centróides em
                                                          # cada objeto na lista.

            #Calcula a distância entre cada par de objeto centróides e os
            # centróides de entrada, respectivamente - nosso objetivo será
            # corresponder um centróide de entrada a um centróide de objeto
            # existente.
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            #Para realizar essa correspondência, devemos (1) encontrar o menor valor em cada
            # linha e (2) classificar os índices de linha com base em seus valores mínimos, de
            # modo que a linha com menor valor esteja na *frente* do índice da lista.
            rows = D.min(axis=1).argsort()

            #Em seguida, realizamos o processo similar nas colunas, localizando o menor valor
            # em cada coluna e, em seguida, classificando usando a lista de índices de linha
            # calculada anteriormente.
            cols = D.argmin(axis=1)[rows]

            #Para determinar se precisamos atualizar, registrar ou cancelar o registro de um
            # objeto, precisamos acomparnhar quais linhas e índices de coluna já examinamos.
            usedRows = set()
            usedCols = set()

            #Loop sobre a combinação das tuplas de índices (linhas, colunas)
            for (row, col) in zip(rows, cols):
                #Se já tivemos examinado a linha ou coluna antes, ignore-a.
                if row in usedRows or col in usedCols:
                    continue

                #Se a distância entre os centróides for maior que a distância
                # máxima, não associe a 2 centróides do mesmo objeto.
                if D[row, col] > self.maxDistance:
                    continue

                #Caso contrário, pega o ID do objeto da linha atual, defina seu novo centróide
                # e redefina o contador de desaparecido.
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                #Indicam que examinamos cada um dos índices de linha e coluna, respectivamente.
                usedRows.add(row)
                usedCols.add(col)

            #Calcula o índice de ambas as linhas e colunas que ainda não examinamos.
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            #Se o número de centróides do objeto seja igual ou superior ao número de centróides
            # de entrada, precisamos verificar e ver se alguns desses objetos potencialmente
            # desapareceram.
            if D.shape[0] >= D.shape[1]:
                #Loop sobre os índices de linha não utilizados
                for row in unusedRows:
                    #Pega o ID do objeto para o índice correspondente e increnta o contador de
                    # desaparecido.
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    #Verifica se o número de frames consecutivos em que o objeto foi marcado como
                    # "desapareceu" para garantir o cancelamento do registro do objeto.
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            #Caso contrário, se o número de centróide entrada for maior que o número de objetos
            # centróides existentes, precisamos registrar cada nova entrada de centróide como
            # objeto rastreável.
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        #Retorna o conjunto de objetos rastreáveis
        return self.objects