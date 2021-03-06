import os
import re
import cv2
import json
import onnx
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


READ_PATH = "Files"
SAVE_PATH = "Processed"
MODEL_PATH = "Models"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


class ImageHandler(object):
    def __init__(self):
        pass

    def read_image(self, path: str) -> np.ndarray:
        return cv2.imread(path, cv2.IMREAD_COLOR)


    def save_image(self, image: np.ndarray) -> None:
        cv2.imwrite(os.path.join(SAVE_PATH, "Processed.jpg"), image)


    def show(self, image: np.ndarray, title=None):
        plt.figure()
        plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB))
        plt.axis("off")
        if title:
            plt.title(title)
        figmanager = plt.get_current_fig_manager()
        figmanager.window.state("zoomed")
        plt.show()

image_handler = ImageHandler()

#######################################################################################################

def new_color(pixel: int, num_colors: int) -> int:
    colors = [(1/num_colors)*i for i in range(num_colors)]
    distances = [abs(pixel-colors[i]) for i in range(len(colors))]
    index = distances.index(min(distances))
    return colors[index]


def find_closest_color(pixel: int, num_colors: int):
    colors = [i*(1/num_colors) for i in range(num_colors+1)]
    distances = [abs(colors[i]-pixel) for i in range(len(colors))]
    index = distances.index(min(distances))
    return colors[index]


class ImageProcessor(object):
    def __init__(self):
        pass

    def gauss_blur(self, image: np.ndarray, kernel_size, sigmaX) -> np.ndarray:
        kernel_size = int(kernel_size)

        if kernel_size == 1:
            kernel_size = 3
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if sigmaX != "": sigmaX = int(sigmaX)
        else: sigmaX = 0

        return cv2.GaussianBlur(src=image, ksize=(kernel_size, kernel_size), sigmaX=sigmaX)
    
    def average_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        return cv2.blur(src=image, ksize=(kernel_size, kernel_size))
    
    def median_blur(self, image: np.ndarray, kernel_size) -> np.ndarray:
        kernel_size = int(kernel_size)

        if kernel_size == 1:
            kernel_size = 3
        
        if kernel_size % 2 == 0:
            kernel_size += 1

        return cv2.medianBlur(src=image, ksize=kernel_size)

    def adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        image = image / 255
        image = np.clip(((image ** gamma) * 255), 0, 255).astype("uint8")
        return image
    
    def adjust_linear_contrast(self, image: np.ndarray, alpha: float) -> np.ndarray:
        return np.clip((image + alpha), 0, 255).astype("uint8")
    
    def adaptive_equalization(self, image: np.ndarray, clipLimit: float, tileGridSize) -> np.ndarray:
        if tileGridSize != "": tileGridSize = int(tileGridSize)
        else: tileGridSize = 2

        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
        if len(image.shape) == 3:
            for i in range(3):
                image[:, :, i] = clahe.apply(image[:, :, i])
        else:
            image = clahe.apply(image)
        return image
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            for i in range(3):
                image[:, :, i] = cv2.equalizeHist(image[:, :, i])
        else:
            image = cv2.equalizeHist(image)
        return image

    def adjust_hue(self, image: np.ndarray, hue: float) -> np.ndarray:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
        feature = image[:, :, 0]
        feature = np.clip((hue * feature), 0, 179).astype("uint8")
        image[:, :, 0] = feature
        return cv2.cvtColor(src=image, code=cv2.COLOR_HSV2BGR)

    def adjust_saturation(self, image: np.ndarray, saturation: float) -> np.ndarray:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
        feature = image[:, :, 1]
        feature = np.clip((saturation * feature), 0, 255).astype("uint8")
        image[:, :, 1] = feature
        return cv2.cvtColor(src=image, code=cv2.COLOR_HSV2BGR)

    def adjust_vibrance(self, image: np.ndarray, vibrance: float) -> np.ndarray:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
        feature = image[:, :, 2]
        feature = np.clip((vibrance * feature), 0, 255).astype("uint8")
        image[:, :, 2] = feature
        return cv2.cvtColor(src=image, code=cv2.COLOR_HSV2BGR)

    def resize_image(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(src=image, dsize=(width, height), interpolation=cv2.INTER_AREA)

    def sharpen(self, image: np.ndarray, kernel_size):
        kernel_size = int(kernel_size)

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(kernel_size, kernel_size)) * -1
        kernel[int(kernel_size / 2), int(kernel_size / 2)] = ((kernel_size - 1) * 2) + 1

        image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        image = np.clip(image, 0, 255).astype("uint8")
        return image

    def posterize_image(self, image: np.ndarray, num_colors: int) -> np.ndarray:
        h, w, c = image.shape
        image = image / 255
        for c in range(c):
            for i in range(h):
                for j in range(w):
                    image[i][j][c] = new_color(image[i][j][c], num_colors)
        return np.clip((image*255), 0, 255).astype("uint8")
    
    def dither_image(self, image: np.ndarray, num_colors: int) -> np.ndarray:
        image = image / 255
        h, w, c = image.shape
        for c in range(c):
            for i in range(h-1):
                for j in range(1, w-1):
                    old_pixel = image[i][j][c]
                    new_pixel = find_closest_color(old_pixel, num_colors)
                    image[i][j][c] = new_pixel
                    quant_error = old_pixel - new_pixel

                    image[i][j+1][c]   = image[i][j+1][c] + (quant_error * 7/16)
                    image[i+1][j+1][c] = image[i+1][j+1][c] + (quant_error * 1/16)
                    image[i+1][j][c]   = image[i+1][j][c] + (quant_error * 5/16)
                    image[i+1][j-1][c] = image[i+1][j-1][c] + (quant_error * 3/16)
        return np.clip((image*255), 0, 255).astype("uint8")
    
    def alpha_blend(self, image_1: np.ndarray, image_2: np.ndarray, alpha: float):
        h1, w1, _ = image_1.shape
        h2, w2, _ = image_2.shape
        if w1 != w2 or h1 != h2:
            image_2 = cv2.resize(src=image_2, dsize=(w1, h1), interpolation=cv2.INTER_AREA)

        return cv2.addWeighted(image_1, alpha, image_2, 1-alpha, 0)
    
    def combine(self, image_1: np.ndarray, image_2: np.ndarray, vertical: bool, adapt_small: bool):
        h1, w1, _ = image_1.shape
        h2, w2, _ = image_2.shape

        if vertical:
            if w1 > w2:
                if adapt_small:
                    image_2 = cv2.resize(src=image_2, dsize=(w1, h2), interpolation=cv2.INTER_AREA)
                else:
                    image_1 = cv2.resize(src=image_1, dsize=(w2, h1), interpolation=cv2.INTER_AREA)

            elif w2 > w1:
                if adapt_small:
                    image_1 = cv2.resize(src=image_1, dsize=(w2, h1), interpolation=cv2.INTER_AREA)
                else:
                    image_2 = cv2.resize(src=image_2, dsize=(w1, h2), interpolation=cv2.INTER_AREA)
                    
            return np.vstack((image_1, image_2))
        
        else:
            if h1 > h2:
                if adapt_small:
                    image_2 = cv2.resize(src=image_2, dsize=(w2, h1), interpolation=cv2.INTER_AREA)
                else:
                    image_1 = cv2.resize(src=image_1, dsize=(w1, h2), interpolation=cv2.INTER_AREA)

            elif h2 > h1:
                if adapt_small:
                    image_1 = cv2.resize(src=image_1, dsize=(w1, h2), interpolation=cv2.INTER_AREA)
                else:
                    image_2 = cv2.resize(src=image_2, dsize=(w2, h1), interpolation=cv2.INTER_AREA)
                    
            return np.hstack((image_1, image_2))

image_processor = ImageProcessor()

#######################################################################################################

def segmenter_decode(class_index_image: np.ndarray) -> np.ndarray:
    colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                       (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                       (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                       (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r, g, b = np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8), \
              np.zeros(class_index_image.shape, dtype=np.uint8)

    for i in range(21):
        indexes = (class_index_image == i)
        r[indexes] = colors[i][0]
        g[indexes] = colors[i][1]
        b[indexes] = colors[i][2]
    return np.stack([r, g, b], axis=2)


class ImageInferer(object):
    def __init__(self, infer_type: str):
        self.infer_type = infer_type
        self.ort_session = None

        if re.match(r"^classify$" ,self.infer_type, re.IGNORECASE):
            self.size = 768
            self.path = "Models/classifier.onnx"
            self.labels = json.load(open("Models/labels_cls.json", "r"))
        
        elif re.match(r"^detect$", self.infer_type, re.IGNORECASE) or re.match(r"^detect_all$", self.infer_type, re.IGNORECASE):
            self.path = "Models/detector.onnx"
            self.labels = json.load(open("Models/labels_det.json", "r"))
            
        
        elif re.match(r"^segment$", self.infer_type, re.IGNORECASE):
            self.path = "Models/segmenter.onnx"
            self.size = 520
            self.labels = json.load(open("Models/labels_seg.json", "r"))
        
    def setup(self):
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)

    def infer(self, image: np.ndarray, disp_image=None, w=None, h=None):
        if re.match(r"^classify$", self.infer_type, re.IGNORECASE):
            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            image = image / 255
            image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
            for i in range(image.shape[0]):
                image[i, :, :] = (image[i, :, :] - MEAN[i]) / STD[i]
            image = np.expand_dims(image, axis=0)
            input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}

            result = np.argmax(self.ort_session.run(None, input))
            return self.labels[str(result)].split(",")[0].title()
        
        elif re.match(r"^detect$", self.infer_type, re.IGNORECASE):
            label = ""
            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
            input = {self.ort_session.get_inputs()[0].name : image.astype("uint8")}
            
            result = self.ort_session.run(None, input)
            boxes, labels, scores, num_detections = result[0].squeeze(), \
                                                    result[1].squeeze(), \
                                                    result[2].squeeze(), \
                                                    int(result[3])

            if num_detections != 0:
                y1, x1, y2, x2 = boxes[0][0] * h, boxes[0][1] * w, boxes[0][2] * h, boxes[0][3] * w
                x1, y1, x2, y2 = int(max(0, np.floor(x1 + 0.5))), \
                                 int(max(0, np.floor(y1 + 0.5))), \
                                 int(min(w, np.floor(x2 + 0.5))), \
                                 int(min(h, np.floor(y2 + 0.5)))

                label = self.labels[str(int(labels[0]))].title()
                cv2.rectangle(disp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(disp_image, label, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            return label
        
        elif re.match(r"^detect_all$", self.infer_type, re.IGNORECASE):
            detected_labels = []

            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
            input = {self.ort_session.get_inputs()[0].name : image.astype("uint8")}
            
            result = self.ort_session.run(None, input)
            boxes, labels, scores, num_detections = result[0].squeeze(), \
                                                    result[1].squeeze(), \
                                                    result[2].squeeze(), \
                                                    int(result[3])

            if num_detections != 0:
                for detection in range(num_detections):
                    y1, x1, y2, x2 = boxes[detection][0] * h, boxes[detection][1] * w, boxes[detection][2] * h, boxes[detection][3] * w
                    x1, y1, x2, y2 = int(max(0, np.floor(x1 + 0.5))), \
                                     int(max(0, np.floor(y1 + 0.5))), \
                                     int(min(w, np.floor(x2 + 0.5))), \
                                     int(min(h, np.floor(y2 + 0.5)))

                    label = self.labels[str(int(labels[0]))].title()
                    detected_labels.append(label)
                    cv2.rectangle(disp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(disp_image, label, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            return list(set(detected_labels))

        elif re.match(r"^segment$", self.infer_type, re.IGNORECASE):
            detected_labels = []

            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            image = image / 255
            image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
            for i in range(image.shape[0]):
                image[i, :, :] = (image[i, :, :] - MEAN[i]) / STD[i]
            image = np.expand_dims(image, axis=0)

            input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}
            result = self.ort_session.run(None, input)
            class_index_image = np.argmax(result[0].squeeze(), axis=0)
            disp_image = cv2.resize(src=segmenter_decode(class_index_image), dsize=(w, h), interpolation=cv2.INTER_AREA)
            
            class_indexes = np.unique(class_index_image)
            for index in class_indexes:
                if index != 0:
                    detected_labels.append(self.labels[str(index)].title())
            return disp_image, list(set(detected_labels))

#######################################################################################################
