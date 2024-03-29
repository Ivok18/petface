import numpy as np
from src.models.vision_pet_emotion import model
from src.models.vision_pet_classifier import model as model_pet_classifier
from src.params import PET_EMOTIONS_LABELS_MAP, PET_LABELS
from src.services.img_process import process_img_for_prediction
from PIL import Image

PDT=0.1
async def get_emotion_from_image(image: Image):
    processed_img = await process_img_for_prediction(image)
    prediction = await predict_img_emotion(processed_img)
    res = infer_prediction(prediction)
    return res


async def predict_img_emotion(img: np.ndarray[float]):
    prediction = model.predict(img)
    return prediction[0, :]


def infer_prediction(prediction: np.ndarray):
    labels_within_PDT_pythonic = np.where(prediction >= prediction.max() - PDT)[0]
    print(labels_within_PDT_pythonic)
    print(prediction)
    return [PET_EMOTIONS_LABELS_MAP.get(label, 0) for label in labels_within_PDT_pythonic]


async def predict_img_classification(image: Image):
    inputs = model_pet_classifier.process_img_classifications(image=image,
                                                              labels=PET_LABELS)
    label_idx = model_pet_classifier.predict_label(inputs)
    return label_idx
