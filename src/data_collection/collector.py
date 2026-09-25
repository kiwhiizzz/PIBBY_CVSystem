import os
import shutil
import random

from src.utils.config import EMOTION_LABELS

TRAIN_RATIO = 0.8

def split_train_test(source_class_folder, image_names, train_ratio=TRAIN_RATIO):
    random.shuffle(image_names)

    split_index = int(len(image_names) * train_ratio)

    train_images = image_names[:split_index]
    test_images = image_names[split_index:]

    return train_images, test_images


def copy_images(image_names, source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    for image_name in image_names:
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copy2(source_path, destination_path)


def organize_dataset(raw_dataset_path, processed_path):
    class_names = os.listdir(raw_dataset_path)

    for class_name in class_names:
        if class_name not in EMOTION_LABELS:
            print(f"Ignorando carpeta desconocida: {class_name}")
            continue

        source_class_folder = os.path.join(raw_dataset_path, class_name)
        image_names = os.listdir(source_class_folder)

        train_images, test_images = split_train_test(source_class_folder, image_names)

        train_destination = os.path.join(processed_path, "train", class_name)
        test_destination = os.path.join(processed_path, "test", class_name)

        copy_images(train_images, source_class_folder, train_destination)
        copy_images(test_images, source_class_folder, test_destination)

        print(f"{class_name}: {len(train_images)} a train, {len(test_images)} a test")


if __name__ == "__main__":
    organize_dataset("data/raw/dataset1", "data/processed")