import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.title("Application de reconnaissance d'images avec YOLO v8")

uploaded_file = st.file_uploader("Choisissez une image à analyser", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    model = YOLO("yolov8n.pt")
    results = model.predict(source=image)

    # Convertir l'image en un tableau numpy pour le traitement avec OpenCV
    image = np.array(image)

    # Parcourir les résultats de la prédiction
    for result in results:
        # Parcourir les boîtes englobantes des détections
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            # Récupérer les coordonnées de la boîte englobante
            x1, y1, x2, y2 = box.int().tolist()

            # Dessiner un rectangle sur l'image avec OpenCV
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Récupérer le nom de la classe et le score de confiance
            class_name = model.names[int(cls)]
            confidence = f"{conf:.2f}"

            # Dessiner une étiquette avec le nom de la classe et le score de confiance
            label = f"{class_name} {confidence}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(y1, label_size[1])
            cv2.rectangle(image, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Convertir l'image traitée en PIL.Image pour l'affichage avec Streamlit
    image = Image.fromarray(image)

    # Afficher l'image avec les rectangles et les étiquettes
    st.image(image, caption="Image avec rectangles et étiquettes", use_column_width=True)
