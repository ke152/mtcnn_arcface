import cv2
from detection import FaceDetection



if __name__ == '__main__':
    img = cv2.imread("detection/test.jpg")
    fd = FaceDetection()
    bboxs, landmarks = fd.infer(img)
    # print(f'bboxs:{bboxs}')
    # print(f'landmarks:{landmarks}')
    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)

    for bbox, landmark in zip(bboxs, landmarks):
        x1, y1, x2, y2, score = bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        for x, y in landmark.reshape((5, 2)):
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.imshow('img', img)
    cv2.waitKey(0)

