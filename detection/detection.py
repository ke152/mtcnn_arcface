import cv2
from .dface.core.detect import create_mtcnn_net, MtcnnDetector


class FaceDetection:

    def __init__(self):
        pnet, rnet, onet = create_mtcnn_net(p_model_path="detection/model_store/pnet_epoch.pt", r_model_path="detection/model_store/rnet_epoch.pt", o_model_path="detection/model_store/onet_epoch.pt", use_cuda=False)
        self.mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    def infer(self, img):
        new_img, ratio = self.resize(img)
        bboxs, landmarks = self.mtcnn_detector.detect_face(new_img)
        bboxs = bboxs / ratio
        landmarks = landmarks / ratio
        return bboxs, landmarks

    def resize(self, img, dst_size=1080):
        h, w = img.shape[:2]

        if w > h:
            ratio = dst_size / w
            new_w = dst_size
            new_h = int(h * ratio)
        else:
            ratio = dst_size / h
            new_h = dst_size
            new_w = int(w * ratio)

        new_img = cv2.resize(img, (new_w, new_h))
        return new_img, ratio

