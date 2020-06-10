class TrackableObject:
    def __init__(self, objectID, centroid):
        #Armazena o ID do objeto, então inicializa a lista de
        # centróides usando o centróide atual.
        self.objectID = objectID
        self.centroids = [centroid]

        #inicializa um booleano usado para indicar se o objeto já
        # foi contado ou não
        self.counted = False