import cv2
from detection import FaceDetection


if __name__ == '__main__':
    img = cv2.imread("detection/test.jpg")
    fd = FaceDetection()
    bboxs, landmarks = fd.infer(img)
    # print(f'bboxs:{bboxs}')
    # print(f'landmarks:{landmarks}')