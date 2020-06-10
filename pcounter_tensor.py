from libraries.utils.centroidtracker import CentroidTracker
from libraries.utils.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import json
import time
import dlib
import cv2

# from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import tensorflow as tf

ap = argparse.ArgumentParser()
'''ap.add_argument("-m", "--model", required=True, help="base path for frozen "
                                                "checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True, help="labels file")
ap.add_argument("-i", "--input", required=True, help="path to input output")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-n", "--num-classes", type=int, required=True, help="# of
                                                        # class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                    help="minimum probability used to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")'''
ap.add_argument("-e", "--conf", required=True,
                help="Caminho do arquivo de configuracao JSON")
args = vars(ap.parse_args())

configure = json.load(open(args["conf"]))

# Inicia aleatoriamente um conjunto de cores RGB para cada caixa delimitadora.
# A inilialização aleatória é feita por conveniência - Podemos modificar esse
# script para usar cores fixas por rótulo
COLORS = np.random.uniform(0, 255, size=(configure["num_classes"], 3))

# Inicializa o modelo que carregamos no disco.
model = tf.Graph()

with model.as_default():
    # Inicia a definição do gráfico
    graphDef = tf.GraphDef()

    # Carrega o gráfico a partir do disco
    with tf.gfile.GFile(configure["model"], "rb") as f:
        serializedGraph = f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef, name="")

# Carrega a classe de rótulos (classes.pbtxt) a partir do disco.
labelMap = label_map_util.load_labelmap(configure["labels"])

# Cria um conjunto de categorias a partir da função
# convert_label_map_to_categories com a opção --num-classes.
categories = label_map_util.convert_label_map_to_categories(
    labelMap, max_num_classes=configure["num_classes"],
    use_display_name=True)

# cria um mapeamento a partir do ID inteiro do rótulo da classe (ou seja,
# o que o TensorFlow retornará ao prever) para o rótulo da classe seja legível
# por humanos.
categoryIdx = label_map_util.create_category_index(categories)

# uma sessão para executar a inferência. Para prever caixas delimitadoras
# para nossa imagem de entrada, primeiro precisamos criar uma sessão
# TensorFlow e obter referências para cada tensor de imagem, caixa
# delimitadora, probabilidade e classes dentro da rede.
with model.as_default():
    with tf.Session(graph=model) as sess:
        # Inicializa os pontos nos arquivos de vídeo.
        print("[INFO] loading model...")

        # Se o caminho do vídeo não for fornecido, pega a referência da WebCam
        if not configure.get("input", False):
            print("[INFO] starting video stream...")
            stream = VideoStream(src=0).start()
            time.sleep(2.0)

        # Caso contrário, pega arquivo de vídeo de referência
        else:
            print("[INFO] opening video file...")
            stream = cv2.VideoCapture(configure["input"])

        # Inicializa a gravação do vídeo.
        writer = None

        # Instancia nosso rastreador centróide e, em seguida, inicializa uma
        # lista para armazenar cada um de nosso rastreadores de correlação
        # dlib, seguidos por um dicionário para mapear cada ID de objeto
        # exclusivo para um TrackableObject.
        ct = CentroidTracker(
                            configure["maxDisappeared"],
                            configure["maxDistance"])
        trackers = []
        trackableObjects = {}

        # Inicializa o número total de frames processados até o momento,
        # juntamente com o número total de objetos que foram movidos para
        # cima ou para baixo.
        totalFrames = 0
        totalDown = 0
        totalUp = 0
        countlogU = 0
        countlogD = 0
        outFrames = totalFrames
        fracao = 4

        # Inicia os frames por segundo durante a estimativa.
        fps = FPS().start()

        # Loop sobre os frames do fluxo de arquivos de vídeo.
        while True:

            # Pega o próximo frame e manipula se estamos lendo o VideoCapture
            # ou VideoStream
            frame = stream.read()
            frame = frame[1] if configure.get("input", False) else frame

            # Se o frame não for pego, então nós temos alcançado o final do
            # fluxo.
            if configure["input"] is not None and frame is None:
                break

            # Pega a referência para o tensor da imagem de entrada e o tensor
            # de caixas (boxes)
            # OBS: Essas referências nos permitirão acessar seus valores
            # associados depois de passar a imagem pela rede.
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # Para cada caixa delimitadora nós gostaríamos de saber a pontuação
            # (score), isto é, a probabilidade dos rótulos de classe.
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")

            # Pega as dimensões da imagem (W = comprimento e H = altura).
            (H, W) = frame.shape[:2]

            # Verifica se nós deveríamos redimensionar junto com o comprimento
            if W > H and W > 1000:
                frame = imutils.resize(frame, width=1000)

            # Verifica se nós deveríamos redimensionar junto a altura.
            elif H > W and H > 1000:
                frame = imutils.resize(frame, height=1000)

            # Verifica se nós deveríamos redimensionar junto a altura.
            (H, W) = frame.shape[:2]
            output = frame.copy()
            frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            frame_dims = np.expand_dims(frame, axis=0)

            # Se o gravador de vídeo for None, inicia a gravação
            if configure["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(configure["output"], fourcc, 30,
                                         (W, H), True)

            # Executar a inferência e calcular as caixas delimitadoras,
            # probabilidades e rótulos de classe.
            # Aqui passamos nossa lista de caixa delimitadora, scores
            # (probabilidades), rótulos de classe e o número de tensores de
            # classe e o número de tensores para o método sess.run.
            # O feed_dict instrui o TensorFlow a definir o imageTensor como a
            # nossa imagem e executar um forward-pass, produzindo nossas caixas
            # delimitadoras, scores e os rótulos de classe.

            (boxes, scores, labels, N) = sess.run(
                    [boxesTensor, scoresTensor, classesTensor, numDetections],
                    feed_dict={imageTensor: frame_dims})

            # Achata as listas em uma única dimensão.
            # As caixas, pontuações e etiquetas são todas matrizes
            # multidimensionais, então as comprimimos em uma matriz 1D,
            # permitindo que passemos facilmente sobre elas.
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)

            status = "Waiting"
            rects = []

            # verifique se devemos executar um método de detecção de objetos
            # mais caro em termos computacionais para ajudar nosso rastreador.
            if totalFrames % configure["skip_frames"] == 0:
                # Ajusta o status e inicializa o nosso novo conjunto de objeto
                # rastreados.
                status = "Detecting"
                trackers = []

                # Loop sobre as predições das caixas delimitadoras
                for (box, score, label) in zip(boxes, scores, labels):
                    # Se a probabilidade das predições for menor que mínimo de
                    # confiança, ignore-a.
                    if score < configure["min_confidence"]:
                        continue
                    # Escalona a caixa delimitadora a partir do intervalo
                    # [0, 1] para [W, H], ou seja,
                    # caminho inverso de quando foi feito o treinamento.
                    (startY, startX, endY, endX) = box
                    startX = int(startX * W)
                    startY = int(startY * H)
                    endX = int(endX * W)
                    endY = int(endY * H)
                    '''print("startY: ", startY)
                    print("startX: ", startX)
                    print("endY: ", endY)
                    print("endX: ", endX)'''
                    '''
                    #Desenha a previsão nas imagens de saída.
                    #Executa o rótulo legível por humanos no dicionário
                    #categoryIdx e desenham o rótulo e a probabilidade
                    # associada a imagem.
                    label = categoryIdx[label]
                    idx = int(label["id"]) - 1
                    label = "{}: {:.2f}".format(label["name"], score)
                    cv2.rectangle(output, (startX, startY), (endX, endY),
                                        COLORS[idx], 2)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(output, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)
                    '''
                    # constrói um objeto de retângulo dlib a partir das
                    # coordenadas da caixa delimitadora e inicie o rastreador
                    # de correlação dlib.
                    tracker = dlib.correlation_tracker()
                    # Salva em rect a lista de coordenadas apenas no momento
                    # que detecta. Esses dados servirão para o rastreador
                    # acompanhar a posição atual e seguir em frente.
                    rect = dlib.rectangle(startX, startY, endX, endY)

                    tracker.start_track(frame, rect)

                    # Adiciona o rastreador na lista de rastreadores para que
                    # possamos utilizá-los durante o salto de frames.
                    trackers.append(tracker)

            # Caso contrário, deveríamos utilizar nossos objetos *rastreadores*
            # em vez de objetos *detectores* para obter uma maior taxa de
            # transferência de processamento de frames.
            else:
                # Loop sobre a lista de objetos rastreados
                for tracker in trackers:
                    # Ajusta o status do sistema para ser 'rastreado' em vez
                    # de 'esperar' ou 'detectar'.
                    status = "Tracking"

                    # Atualiza o rastreador e pega a posição atualizada
                    tracker.update(frame)
                    pos = tracker.get_position()

                    # Desfaz a posição do objeto
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # Adiciona as coordenadas das caixas delimitadoras na
                    # lista de retângulos.
                    rects.append((startX, startY, endX, endY))
            # Desenha uma linha horizontal no centro do frame --  uma vez que
            # o objeto cruza a linha nós determinamos se eles se movem para
            # cima ou para baixo.
            cv2.line(
                    output,
                    (0, H // fracao), (W, H // fracao), (0, 255, 255), 2)

            # Usa o centróide do rastreador para associar o centróide do
            # objeto antigo com cálculo do novo objeto centróide.
            objects = ct.update(rects)

            # Loop sobre os objetos rastreados
            for (objectID, centroid) in objects.items():
                # Verifica se o objeto rastreável existe para o ID do objeto
                # atual.
                to = trackableObjects.get(objectID, None)

                # Se não há objeto rastreável, crie um
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # Caso contrário, existe um objeto rastreável então nós
                # podemos utilizá-lo para determinar a direção.
                else:
                    # A diferença entre a coordenada y do centróide *atual* e
                    # a média dos centróides *anteriores* nos dirá em que
                    # direção o objeto está se movendo (negativo 'para cima'
                    # e positivo 'para baixo')
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    # Verifica se o objeto tem sido contado ou não
                    if not to.counted:
                        # Se a direção for negativa (indica que o objeto está
                        # se movendo para cima) E o centróide está acima do
                        # centro da linha, conta o objeto.
                        # if direction < 0 and centroid[1] < H // 2:
                        if direction < 0 and centroid[1] < H // fracao \
                                and centroid[1] > ((H // fracao) - 20):
                            totalUp += 1
                            to.counted = True

                        # Se a direção for positiva (indica que o objeto está
                        # se movendo para baixo) E o centróide está abaixo da
                        # linha central, conta o objeto.
                        # elif direction > 0 and centroid[1] > H // 2:
                        elif direction > 0 and centroid[1] > H // fracao \
                                and centroid[1] < ((H // fracao) + 20):
                            totalDown += 1
                            to.counted = True

                # Armazena o objeto rastreável em nosso dicionário
                trackableObjects[objectID] = to

                # Desenha ambos o ID do objeto e o centróide do objeto no frame
                # de saída
                text = "ID {}".format(objectID)
                cv2.putText(output, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(
                        output,
                        (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # Constrói uma tupla de informações nós mostraremos no frame.
            info = [
                ("Up", totalUp),
                ("Down", totalDown),
                ("Status", status)
            ]

            # Loop sobre a informações das tuplas e desenha-os no nosso frame.
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(
                            output, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Verifica se nós deveríamos gravar o frame no disco.
            if writer is not None:
                writer.write(output)

            # Mostra o frame de saída
            cv2.imshow("Frame", output)
            key = cv2.waitKey(1) & 0xFF

            # Se o 'q' for pressionado, para o loop
            if key == ord("q"):
                break

            # incrementa o número total de frames processados até o momento
            # e atualiza o contador FPS.
            totalFrames += 1
            fps.update()
        # Para o tempo e mostra as informações FPS
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # verifique se precisamos liberar o ponteiro do gravador de vídeo
        if writer is not None:
            writer.release()

        # se não estivermos usando um arquivo de vídeo, pare o fluxo de vídeo \
        # da câmera
        if not configure.get("input", False):
            stream.stop()

        # caso contrário, solte o ponteiro do arquivo de vídeo
        else:
            stream.release()

        # feche todas as janelas abertas
        cv2.destroyAllWindows()
