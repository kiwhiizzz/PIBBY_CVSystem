import os 
import cv2

IMG_SIZE = 224

EMOTION_LABELS = {
    "happy" : 0,
    "sad" : 1,
    "neutral" : 2,
    "angry" :  3,
    "fear" : 4,
}

def list_images_class(base_path):
    classes = os.listdir(base_path)

    images_data = []
    labels_data = []

    for class_name in classes:
        class_folder = os.path.join(base_path, class_name)
        image_names = os.listdir(class_folder)

        if class_name not in EMOTION_LABELS:
            continue;

        label  = EMOTION_LABELS[class_name]

        for image_name in image_names:
            image_path = os.path.join(class_folder, image_name)
            image = load_resize_image(image_path)

            images_data.append(image)
            labels_data.append(label)

    return images_data, labels_data

def load_resize_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image