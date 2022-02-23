import cv2
import mediapipe as mp

mp_selfie_segmentation = mp.solutions.selfie_segmentation
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:

    while True:
        ret, frame = captura.read()
        if ret == False:
            break

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        resultados = selfie_segmentation.process(frameRGB)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', resultados.segmentation_mask)
        if cv2.waitKey(1) & 0xFF == 27:
            break
captura.release()
cv2.destroyAllWindows()
