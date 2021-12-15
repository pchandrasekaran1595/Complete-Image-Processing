import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


READ_PATH = "Files"
SAVE_PATH = "Processed"


if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


class ImageHandler(object):
    def __init__(self):
        pass

    def read_image(path: str) -> np.ndarray:
        return cv2.imread(path, cv2.IMREAD_COLOR)


    def save_image(image) -> None:
        cv2.imwrite(os.path.join(SAVE_PATH, "Processed.jpg"), image)


    def show(image: np.ndarray, title=None):
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


def find_closest_color(pixel, num_colors):
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

image_processor = ImageProcessor()

#######################################################################################################

class ImageInferer(object):
    def __init__(self):
        pass

    def infer(self):
        pass

#######################################################################################################
