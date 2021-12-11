import cv2
from .dface.core.detect import create_mtcnn_net, MtcnnDetector


class FaceDetection:

    def __init__(self):
        pnet, rnet, onet = create_mtcnn_net(p_model_path="detection/model_store/pnet_epoch.pt", r_model_path="detection/model_store/rnet_epoch.pt", o_model_path="detection/model_store/onet_epoch.pt", use_cuda=False)
        self.mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    def infer(self, img):
        img = cv2.resize(img, (512,512))
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #b, g, r = cv2.split(img)
        #img2 = cv2.merge([r, g, b])

        bboxs, landmarks = self.mtcnn_detector.detect_face(img)
        return bboxs, landmarks

