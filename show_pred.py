import cv2
import numpy as np
import matplotlib.pyplot as plt
def show_prediction(image, model):
    img = cv2.resize(image, (256, 256))
    img = np.expand_dims(img, axis=0)
    gender_prob, age_prob = model.predict(img)
    gender = "male" if gender_prob < 0.5 else "female"
    age = [
        "(0, 2)",
        "(4, 6)",
        "(8, 12)",
        "(15, 20)",
        "(25, 32)",
        "(38, 43)",
        "(48, 53)",
        "(60, 100)",
    ][np.argmax(age_prob)]
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{gender}, {age}")
