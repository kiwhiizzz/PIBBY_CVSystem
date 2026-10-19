import torch

from src.training.train_fer import build_model, set_dataloader
from src.training.dataset_loader import list_images_class
from src.utils.config import EMOTION_LABELS, MODEL_PATH
from datetime import datetime

def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes = len(EMOTION_LABELS))
    model.load_state_dict(torch.load(model_path, map_location = device, weights_only = True))
    model = model.to(device)
    model.eval()

    return model, device

def get_predictions(model, device, data_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim = 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    return all_labels, all_preds

def main():
    print(f"Modelo evaluado: {MODEL_PATH}")
    images, labels = list_images_class("data/processed/test")
    test_loader = set_dataloader(images, labels, batch_size = 32, shuffle = False)

    model, device = load_trained_model(MODEL_PATH)
    preds, true_labels = get_predictions(model, device, test_loader)

    total = len(true_labels);
    correct = sum(p == t for p, t in zip(preds, true_labels))
    accuracy = correct / total
    
    print(f"Images de test: {len(true_labels)}")
    print(f"Aciertos: {correct} ({correct / len(true_labels):.1%})")

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open("models/registro.txt", "a", encoding = "utf-8") as f:
        f.write(
            f"Evaluación | {MODEL_PATH} | test: {total} |"
            f"Aciertos: {correct} | exactitud: {accuracy:.1%} | date: {fecha}\n"
        )
    print("Resultados agregado al registro")

if __name__ == "__main__":
    main()
