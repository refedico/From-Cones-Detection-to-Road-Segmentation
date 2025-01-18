import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import os
import time
import math
import numpy as np

# Only for Windows user
#import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

# Costanti
YELLOW_INDEX = 4
BLUE_INDEX = 0
CLASS_NUM = 5


def generate_binary_mask(img, yellow_points, blue_points, mask_path):
    if len(blue_points) == 0:
        #Inserisci all'inizio nei blue points il punto x = 0 e y max
        blue_points.insert(0, (0, img.size[1]))
    elif len(blue_points) == 1:
        # Inserisci il prolugamento sul punto in basso a sinistra
        blue_points.insert(0, (0, img.size[1]))
    else:
        # Inserisci il prolugamento sul bordo della retta tra il primo e il secondo punto
        x1, y1 = blue_points[0]
        x2, y2 = blue_points[1]
        m = (y2 - y1) / (x2 - x1)
        q = y1 - m * x1
        x = 0
        y = m * x + q
        # Se l'inclinazione della retta è positiva, inserisci il punto in basso a sinistra
        if m > 0:
            blue_points.insert(0, (0, img.size[1]))
        else:
            blue_points.insert(0, (x, y))
            blue_points.insert(0, (0, img.size[1]))
    if len(yellow_points) == 0:
        #Inserisci all'inizio nei yellow points il punto x = max e y max
        yellow_points.insert(0, (img.size[0], img.size[1]))
    elif len(yellow_points) == 1:
        yellow_points.insert(0, (img.size[0], img.size[1]))
    else:
        # Inserisci il prolugamento sul bordo della retta tra il primo e il secondo punto
        x1, y1 = yellow_points[0]
        x2, y2 = yellow_points[1]
        m = (y2 - y1) / (x2 - x1)
        q = y1 - m * x1
        x = img.size[0]
        y = m * x + q
        # Se l'inclinazione della retta è negativa, inserisci il punto in basso a destra
        if m < 0:
            yellow_points.insert(0, (img.size[0], img.size[1]))
        else:
            yellow_points.insert(0, (x, y))
            yellow_points.insert(0, (img.size[0], img.size[1]))
    img_array = np.array(img)
    mask_f = np.zeros(img_array.shape[:2], dtype=np.uint8)
    # Ensure both arrays have the same number of dimensions
    yellow_points = np.array(yellow_points).reshape(-1, 2)
    blue_points = np.array(blue_points).reshape(-1, 2)

    # Inverti l'ordine dei blue points
    blue_points = blue_points[::-1]
    points = np.concatenate((yellow_points, blue_points))
    points = points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask_f, [points], 255)

    # Save the binary mask
    cv2.imwrite(mask_path, mask_f)

    return mask_f
    

def order_img(img_dir):
    imgs = []
    for img_name in os.listdir(img_dir):
        imgs.append(img_name)
    imgs.sort()

    return imgs

def max_box(boxes):
    max_height = 0
    max_box = 0
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i][:4]
        if abs(y2 - y1) > max_height:
            max_height = abs(y2 - y1)
            max_box = i
    return max_box


def lowest_point(points):
    lowest = 0
    for i in range(len(points)):
        if points[i][1] > points[lowest][1]:
            lowest = i
    return lowest

def doesIntersect(p1, q1, p2, q2):
    def onSegment(p, q, r):
        if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
               (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False

    def orientation(p, q, r):
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if (val > 0):
            return 1
        elif (val < 0):
            return 2
        else:
            return 0

    p1 = list(p1)
    q1 = list(q1)
    p2 = list(p2)
    q2 = list(q2)

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if ((o1 != o2) and (o3 != o4)):
        return True

    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True

    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True

    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True

    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True

    return False


 
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def list_graphs(boxes, points, class_ids):
    graphs = []
    boxes_classified = []
    points_classified = []
    for i in range(9):
        boxes_class = []
        mid_points_class = []
        for j in range(len(boxes)):
            if class_ids[j] == i:
                boxes_class.append(boxes[j])
                mid_points_class.append(points[j])
        graph = graph_generator(boxes_class, mid_points_class)
        graphs.append(graph)
        boxes_classified.append(boxes_class)
        points_classified.append(mid_points_class)
    return graphs, boxes_classified, points_classified


def graph_generator(boxes, points):
    graph = []
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                height1 = boxes[i][3] - boxes[i][1]
                height2 = boxes[j][3] - boxes[j][1]
                distance = (((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 3) ** 0.5 ) * (1 / height1 + 1 / height2)
                graph.append((i, j, distance))
    graph.sort(key=lambda x: x[2])
    return graph

def inference(model, imgs, i):
    # Inference
    results = model(imgs)

    # Dizionario dei colori definiti manualmente per ciascuna classe
    # Ogni classe è associata a un colore RGB
    CLASS_COLORS = {
        0: (255, 0, 0),    # Rosso
        1: (0, 255, 0),    # Verde
        2: (0, 0, 255),    # Blu
        3: (255, 255, 0),  # Giallo
        4: (255, 165, 0),  # Arancione
        5: (128, 0, 128),  # Viola
        6: (0, 255, 255),  # Azzurro
        7: (255, 192, 203),# Rosa
        8: (128, 128, 0)   # Oliva
    }

    # Funzione per disegnare bounding box con colori specifici per classe
    def draw_colored_boxes(img, boxes, class_ids):
        draw = ImageDraw.Draw(img)
        for box, cls_id in zip(boxes, class_ids):
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))  # Default: bianco se la classe non è nel dizionario
            draw.rectangle(box[:4], outline=color, width=3)
        return img
    
    def draw_colored_points(img, mid_points, class_ids):
        draw = ImageDraw.Draw(img)
        for point, cls_id in zip(mid_points, class_ids):
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))  # Default: bianco se la classe non è nel dizionario
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
        return img

    '''def isoutlier(boxes, i, j, x1, x2, y1, y2):
        box_area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        box_area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
        print(box_area_i, box_area_j)
        pixel_distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        print(pixel_distance)
        if pixel_distance > 500 and box_area_i > 3000 and box_area_j > 3000:
            return False
        elif pixel_distance < 500:
            return False
        else:
            return True'''

    def isoutlier(x1, x2, x3, y1, y2, y3):
        # Se l'angolo tra i due segmenti è maggiore di 90 gradi, ritorna True
        angle = getAngle((x1, y1), (x2, y2), (x3, y3)) - 180
        angle = abs(angle) > 160

        # Se la distanza tra i punti 2 e 3 è maggiore di 3 volte la distanza tra i punti 1 e 2, ritorna True
        distance = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
        distance = distance > 3 * (((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
        if angle or distance:
            return True
        else:
            return False
        

    def draw_colored_lines(img, points_classified, boxes_classified, graphs):
        draw = ImageDraw.Draw(img)
        
        yellow_points = []
        blue_points = []
        for class_id in range(CLASS_NUM):
            if len(boxes_classified[class_id]) == 0:
                continue
            if class_id != YELLOW_INDEX and class_id != BLUE_INDEX:
                continue
            graph = graphs[class_id]
            boxes = boxes_classified[class_id]
            mid_points = points_classified[class_id]
            # Prendi l'indice del punto più basso per la classe corrente
            starting_index = lowest_point(mid_points)
            connected = [starting_index]
            current_index = starting_index
            last_index = starting_index
            ended = False
            if class_id == YELLOW_INDEX:
                yellow_points.append(mid_points[starting_index])
            elif class_id == BLUE_INDEX:
                blue_points.append(mid_points[starting_index])
            while len(connected) < len(boxes) and not ended:
                for i, j, distance in graph:
                    if i == current_index and j not in connected:
                        x2, y2 = mid_points[i]
                        x3, y3 = mid_points[j]
                        x1, y1 = mid_points[last_index]
                        color = CLASS_COLORS.get(class_id, (255, 255, 255))
                        if i == last_index or not isoutlier(x1, x2, x3, y1, y2, y3):
                            #draw.line([mid_points[i], mid_points[j]], fill=color, width=3)
                            if class_id == YELLOW_INDEX:
                                yellow_points.append(mid_points[j])
                            elif class_id == BLUE_INDEX:
                                blue_points.append(mid_points[j])
                        else:
                            ended = True
                            break
                        connected.append(j)
                        current_index = j
                        last_index = i
                        break
        # Controlla che i segmenti delle linee non si sovrappongano tra colori diversi
        # Se si sovrappongono, rimuovi i punti più outlier
        if len(yellow_points) > 1 and len(blue_points) > 1:
            for i in range(1, len(yellow_points)):
                if i - 1 < 0 or i >= len(yellow_points):
                    break
                x1, y1 = yellow_points[i - 1]
                x2, y2 = yellow_points[i]
                for j in range(1, len(blue_points)):
                    if j - 1 < 0 or j >= len(blue_points):
                        break
                    x3, y3 = blue_points[j - 1]
                    x4, y4 = blue_points[j]
                    # Se i segmenti si incrociano, rimuovi i punti più outlier
                    if doesIntersect((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
                        distance1 = ((x1 - x2) ** 2 + (y1 - y2) ** 3) ** 0.5
                        distance2 = ((x3 - x4) ** 2 + (y3 - y4) ** 3) ** 0.5
                        if distance1 > distance2:
                            yellow_points = yellow_points[:i]
                        else:
                            blue_points = blue_points[:j]

        # Se il segmento tracciato tra il punto in basso a destra e il primo cono giallo incrocia un segmento blu, rimuovi tutti i punti gialli
        if len(yellow_points) > 1 and len(blue_points) > 1:
            # Punto in basso a destra nell'immagine
            x1, y1 = (img.size[0], img.size[1])
            x2, y2 = yellow_points[0]
            for j in range(1, len(blue_points)):
                if j - 1 < 0 or j >= len(blue_points):
                    break
                x3, y3 = blue_points[j - 1]
                x4, y4 = blue_points[j]
                if doesIntersect((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
                    yellow_points = []
                    break
        # Se il segmento tracciato tra il punto in basso a sinistra e il primo cono blu incrocia un segmento giallo, rimuovi tutti i punti blu
        if len(yellow_points) > 1 and len(blue_points) > 1:
            # Punto in basso a sinistra nell'immagine
            x1, y1 = (0, img.size[1])
            x2, y2 = blue_points[0]
            for i in range(1, len(yellow_points)):
                if i - 1 < 0 or i >= len(yellow_points):
                    break
                x3, y3 = yellow_points[i - 1]
                x4, y4 = yellow_points[i]
                if doesIntersect((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
                    blue_points = []
                    break

        return img, (blue_points, yellow_points)
    
    def draw_colored_curves(img, points_classified, boxes_classified, graphs):
        draw = ImageDraw.Draw(img)
        for class_id in range(CLASS_NUM):
            if len(boxes_classified[class_id]) == 0:
                continue
            if class_id != YELLOW_INDEX and class_id != BLUE_INDEX:
                continue
            graph = graphs[class_id]
            boxes = boxes_classified[class_id]
            mid_points = points_classified[class_id]
            # Prendi l'indice della box più alta per la classe corrente
            starting_index = lowest_point(mid_points)
            connected = [starting_index]
            ordered_points = [mid_points[starting_index]]
            current_index = starting_index
            while True:
                for i, j, distance in graph:
                    if i == current_index and j not in connected:
                        ordered_points.append(mid_points[j])
                        connected.append(j)
                        current_index = j
                        break
                if len(connected) == len(boxes):
                    break
            # Polinomial interpolation of the points to draw a curve
            #npImg = np.array(img)
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            # Use Centripetal Catmull–Rom spline
            #tck, u = splprep(np.array(ordered_points).T, u=None, s=0.0, per=1)
            #u_new = np.linspace(u.min(), u.max(), 1000)
            #x_new, y_new = splev(u_new, tck, der=0)
            #ordered_points = np.column_stack((x_new, y_new)).tolist()
            #cv2.polylines(npImg, [np.array(ordered_points).astype(int)], isClosed=False, color=color, thickness=3)
            #img = Image.fromarray(npImg)
        return img

    
    # Funzione per calcolare l'Intersection over Union (IoU) tra due bounding box
    def iou(box1, box2):
        # Calcola l'area di intersezione
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        # Calcola l'area dell'unione
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area_box1 + area_box2 - intersection_area
        # Calcola e restituisce l'IoU
        return intersection_area / union_area

    # Esegui inferenza
    results = model(imgs)

    # Ottieni bounding box e classi
    boxes = results[0].boxes.data.cpu().numpy()
    boxes = boxes[:, :4]  # Bounding box: [xmin, ymin, xmax, ymax]
    #boxes = results.xyxy[0].cpu().numpy()  # Bounding box: [xmin, ymin, xmax, ymax, conf, cls]
    class_ids = results[0].boxes.data.cpu().numpy()[:, 5].astype(int)    # Indici di classe predetti
    img_path = imgs[0]

    # Leggi fiducia e classe predetta
    confidences = results[0].boxes.conf.cpu().numpy()
    # Se due boxes si sovrappongono per più del 60%, mantieni solo quella con fiducia maggiore
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou(boxes[i], boxes[j]) > 0.6:
                if confidences[i] > confidences[j]:
                    boxes[j] = 0
                else:
                    boxes[i] = 0
    boxes = boxes[boxes[:, 0] != 0]
    #confidences = boxes[:, 4]

    # Rimuovi bounding box con fiducia minore di 0.5
    #boxes = boxes[confidences > 0.5]

    #Rimuovi boxes più piccoli del 30% della box più grande
    max_height = 0
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i][:4]
        if abs(y2 - y1) > max_height:
            max_height = abs(y2 - y1)

    boxes = [box for box in boxes if abs(box[3] - box[1]) > 0.10 * max_height]

    # Trova punti medi sulla base delle coordinate dei bounding box
    mid_points = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        mid_points.append(((x1 + x2) / 2, y2))

    # genera lista di grafi dove in ogni grafo ci sono i punti medi e le distanze tra i punti di una classe
    #Misura tempo di esecuzione
    start = time.time()
    graphs, boxes_classified, points_classified = list_graphs(boxes, mid_points, class_ids)
    end1 = time.time()

    img = Image.open(img_path)

    # Disegna bounding box colorati in base alle classi
    #img_with_colored_boxes = draw_colored_boxes(img, boxes, class_ids)

    #img_with_colored_boxes = draw_colored_points(img, mid_points, class_ids)

    img_with_colored_boxes, (blue_points, yellow_points) = draw_colored_lines(img, points_classified, boxes_classified, graphs)
    #mask = generate_binary_mask(img, yellow_points, blue_points, "mask.jpg")

    # For datasets generation
    mask_path = "/home/root/ADAS/mask.jpg"
    mask = generate_binary_mask(img, yellow_points, blue_points, mask_path)
    #img.save(img_path)

    # Somma la maschera binaria all'immagine originale con 0.5 di opacità
    img_sum = Image.blend(img_with_colored_boxes, Image.fromarray(np.stack((mask,) * 3, axis=-1)), 0.2)

    # Riarrangia i colori nell'imagine risultante, mettendo il valore del blu al posto del giallo
    img_sum = np.array(img_sum)
    img_sum = img_sum[:, :, [2, 1, 0]]
    img_sum = Image.fromarray(img_sum)

    end2 = time.time()
    print("Time for graph generation: ", end1 - start)
    print("Time for drawing lines: ", end2 - end1)

    # Salva il risultato
    img_with_colored_boxes.save('output.jpg')

    # Visualizza il risultato con opencv
    img_cv = cv2.imread('output.jpg')
    #mask_cv = cv2.imread('mask.jpg')
    resized = cv2.resize(img_cv, (800, 600), interpolation=cv2.INTER_AREA)
    #resized_mask = cv2.resize(mask_cv, (800, 600), interpolation=cv2.INTER_AREA)
    #cv2.imshow('Mask', resized_mask)
    cv2.imshow('Result', resized)
    img_sum = np.array(cv2.resize(np.array(img_sum), (800, 600), interpolation=cv2.INTER_AREA))
    cv2.imshow('Result with mask', img_sum)
    return img, mask


def seg_inference(model, imgs):
    img = Image.open(imgs[0])
    results = model(imgs)
    # Draw the masks
    if (results[0].masks is None):
        return
    for mask in results[0].masks.xy:
        points = np.int32([mask])
        segmented = cv2.fillPoly(np.array(img), points, (255, 255, 255))

    img_sum = Image.blend(img, Image.fromarray(segmented), 0.2)
    img_sum = np.array(img_sum)
    img_sum = np.array(img_sum)[:, :, [0, 1, 2]]
    img_sum = Image.fromarray(img_sum)

    img_sum.save('segmented.jpg')
    img_cv = cv2.imread('segmented.jpg')
    resized = cv2.resize(img_cv, (800, 600), interpolation=cv2.INTER_AREA)
    cv2.imshow('Segmentation', resized)


def main():
    # Model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    # Load custom model
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolom.pt')
    model = YOLO("unipr-detect.pt")
    seg_model = YOLO("yolo-segment.pt")
    # Cicla sulle immagini jpg nella cartella Dataset/amz/img
    img_dir = '/home/root/ADAS/Dataset/amz/img/'
    CROP = True
    #img_dir = '/home/root/ADAS/frames_uniprrt'
    
    imgs = order_img(img_dir)

    # Crop each image to the region of interest that is in 140 px from each side
    if CROP:
        for img_name in imgs:
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                img_path = os.path.join(img_dir, img_name)
                img = Image.open(img_path)
                img = img.crop((140, 140, img.size[0] - 140, img.size[1] - 140))
                img.save("/home/root/ADAS/test/" + img_name)
        img_dir = "/home/root/ADAS/test"
    i = 1000
    while i - 1000 < len(imgs):
        img_name = imgs[i - 1000]
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_path = os.path.join(img_dir, img_name)
            start = time.time()
            img, mask = inference(model, [img_path], i)        
            end = time.time()
            seg_inference(seg_model, [img_path])

            key = cv2.waitKey(0)
            if key == ord('s'):
                mask_path = "/home/root/ADAS/saved_images/masks/img" + str(i) + ".jpg"
                img_path = "/home/root/ADAS/saved_images/images/img" + str(i) + ".jpg"
                img.save(img_path)
                mask = Image.fromarray(mask)
                mask.save(mask_path)
                i += 1
            if key == ord('q'):
                break
            elif key == ord('n'):
                i += 1
            elif key == ord('p'):
                if i > 0:
                    i -= 1
            
        print("Time for inference: ", end - start)


main()
