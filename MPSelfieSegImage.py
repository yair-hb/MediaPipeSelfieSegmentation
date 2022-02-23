import cv2
import mediapipe as mp

mp_selfie_segmentation = mp.solutions.selfie_segmentation

with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
    
    imagen = cv2.imread('imagen.jpg')
    imagenRGB = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultado = selfie_segmentation.process(imagenRGB)

    cv2.imshow('imagen', imagen)
    cv2.imshow('mask', resultado.segmentation_mask)
    cv2.waitKey(0)
cv2.destroyAllWindows()
