import cv2
from models import *
import torch
import numpy as np
import time
from torch.nn import DataParallel
from tqdm import tqdm


class FaceRecognition:

    def __init__(self):
        self.opt = {
            'use_se': False,
            'test_model_path': 'checkpoints/resnet18_110.pth',
        }

        self.model = self.load_model()

    def get_feature(self, img, landmark_5pts=None):
        if landmark_5pts is not None:
            pass
        images = None
        features = None
        cnt = 0
        for i, img_path in enumerate(tqdm(test_list)):

            image = self.load_image(img_path)
            if image is None:
                print('read {} error'.format(img_path))

            if images is None:
                images = image
            else:
                images = np.concatenate((images, image), axis=0)

            if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
                cnt += 1

                # images = images[:1,:,:,:]
                data = torch.from_numpy(images)
                data = data.to(torch.device("cpu"))
                output = model(data)
                # print(images.shape)
                # for _ in range(10):
                #     ts = time.time()
                #     output = model(data)
                #     te = time.time()
                #     print(f'cost time:{te-ts}')
                # exit()
                output = output.data.cpu().numpy()

                fe_1 = output[::2]
                fe_2 = output[1::2]
                feature = np.hstack((fe_1, fe_2))
                # print(feature.shape)

                if features is None:
                    features = feature
                else:
                    features = np.vstack((features, feature))

                images = None

        return features, cnt

    def infer(self):
        identity_list = ['Abel_Pacheco/Abel_Pacheco_0001.jpg', 'Abel_Pacheco/Abel_Pacheco_0004.jpg', 'Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg', 'Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg']
        img_paths = ['D:/data/lfw-align-128\\Abel_Pacheco/Abel_Pacheco_0001.jpg', 'D:/data/lfw-align-128\\Abel_Pacheco/Abel_Pacheco_0004.jpg', 'D:/data/lfw-align-128\\Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg', 'D:/data/lfw-align-128\\Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg']
        self.lfw_test(self.model, img_paths, identity_list, 'lfw_test_pair.txt')

    def load_model(self, backbone='resnet18'):
        if backbone == 'resnet18':
            model = resnet_face18(self.opt['use_se'])
        elif backbone == 'resnet34':
            model = resnet34()
        elif backbone == 'resnet50':
            model = resnet50()
        else:
            model = resnet_face18(self.opt['use_se'])

        model = DataParallel(model)
        model.load_state_dict(torch.load(self.opt['test_model_path'], map_location='cpu'))
        model.eval()
        return model





    def load_image(self, img_path):
        image = cv2.imread(img_path, 0)
        if image is None:
            return None
        image = np.dstack((image, np.fliplr(image)))
        image = image.transpose((2, 0, 1))
        image = image[:, np.newaxis, :, :]
        image = image.astype(np.float32, copy=False)
        image -= 127.5
        image /= 127.5
        return image

    def get_featurs(self, model, test_list, batch_size=10):
        images = None
        features = None
        cnt = 0
        for i, img_path in enumerate(tqdm(test_list)):

            image = self.load_image(img_path)
            if image is None:
                print('read {} error'.format(img_path))

            if images is None:
                images = image
            else:
                images = np.concatenate((images, image), axis=0)

            if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
                cnt += 1

                # images = images[:1,:,:,:]
                data = torch.from_numpy(images)
                data = data.to(torch.device("cpu"))
                output = model(data)
                # print(images.shape)
                # for _ in range(10):
                #     ts = time.time()
                #     output = model(data)
                #     te = time.time()
                #     print(f'cost time:{te-ts}')
                # exit()
                output = output.data.cpu().numpy()

                fe_1 = output[::2]
                fe_2 = output[1::2]
                feature = np.hstack((fe_1, fe_2))
                # print(feature.shape)

                if features is None:
                    features = feature
                else:
                    features = np.vstack((features, feature))

                images = None

        return features, cnt

    def get_feature_dict(self, test_list, features):
        fe_dict = {}
        for i, each in enumerate(test_list):
            # key = each.split('/')[1]
            fe_dict[each] = features[i]
        return fe_dict

    def cosin_metric(self, x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def cal_accuracy(self, y_score, y_true):
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        best_acc = 0
        best_th = 0
        for i in range(len(y_score)):
            th = y_score[i]
            y_test = (y_score >= th)
            acc = np.mean((y_test == y_true).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = th

        return (best_acc, best_th)

    def test_performance(self, fe_dict, pair_list):
        with open(pair_list, 'r') as fd:
            pairs = fd.readlines()

        sims = []
        labels = []
        for pair in pairs:
            splits = pair.split()
            fe_1 = fe_dict[splits[0]]
            fe_2 = fe_dict[splits[1]]
            label = int(splits[2])
            sim = self.cosin_metric(fe_1, fe_2)

            sims.append(sim)
            labels.append(label)

        acc, th = self.cal_accuracy(sims, labels)
        return acc, th

    def lfw_test(self, model, img_paths, identity_list, compair_list):
        s = time.time()
        features, cnt = self.get_featurs(model, img_paths, batch_size=1)
        print(features.shape)
        t = time.time() - s
        print('total time is {}, average time is {}'.format(t, t / cnt))
        fe_dict = self.get_feature_dict(identity_list, features)
        acc, th = self.test_performance(fe_dict, compair_list)
        print('lfw face verification accuracy: ', acc, 'threshold: ', th)
        return acc


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.infer()





