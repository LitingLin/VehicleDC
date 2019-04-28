# coding: utf-8

import argparse

from darknet_util import *
from darknet import Darknet
from preprocess import process_img
from dataset import color_attrs, direction_attrs, type_attrs

import torch
import torchvision
import cv2
import PIL
import os
from PIL import Image

use_cuda = True  # True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(
    'cuda: 0' if torch.cuda.is_available() and use_cuda else 'cpu')

if use_cuda:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
# print('=> device: ', device)


class Cls_Net(torch.nn.Module):
    """
    vehicle multilabel classification model
    """

    def __init__(self, num_cls, input_size):
        """
        network definition
        :param is_freeze:
        """
        torch.nn.Module.__init__(self)

        # output channels
        self._num_cls = num_cls

        # input image size
        self.input_size = input_size

        # delete original FC and add custom FC
        self.features = torchvision.models.resnet18(pretrained=True)
        del self.features.fc
        # print('feature extractor:\n', self.features)

        self.features = torch.nn.Sequential(
            *list(self.features.children()))

        self.fc = torch.nn.Linear(512 ** 2, num_cls)  # 输出类别数
        # print('=> fc layer:\n', self.fc)

    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]

        X = self.features(X)  # extract features

        X = X.view(N, 512, 1 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1 ** 2)  # Bi-linear CNN

        X = X.view(N, 512 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, self._num_cls)
        return X


# ------------------------------------- vehicle detection model
class VehicleClassifier(object):
    """
    vehicle detection model mabager
    """

    def __init__(self,
                 num_cls,
                 model_path):
        """
        load model and initialize
        """

        # define model and load weights
        self.net = Cls_Net(num_cls=num_cls, input_size=224).to(device)
        # self.net = torch.nn.DataParallel(Net(num_cls=20, input_size=224),
        #                                  device_ids=[0]).to(device)
        if use_cuda:
            self.net.load_state_dict(torch.load(model_path))
        else:
            self.net.load_state_dict(torch.load(model_path, map_location='cpu'))

        #print('=> vehicle classifier loaded from %s' % model_path)

        # set model to eval mode
        self.net.eval()

        # test data transforms
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        # split each label
        self.color_attrs = color_attrs
        #print('=> color_attrs:\n', self.color_attrs)

        self.direction_attrs = direction_attrs
        #print('=> direction attrs:\n', self.direction_attrs)

        self.type_attrs = type_attrs
        #print('=> type_attrs:\n', self.type_attrs)

    def get_predict(self, output):
        """
        get prediction from output
        """
        # get each label's prediction from output
        output = output.cpu()  # fetch data from gpu
        pred_color = output[:, :9]
        pred_direction = output[:, 9:11]
        pred_type = output[:, 11:]

        color_idx = pred_color.max(1, keepdim=True)[1]
        direction_idx = pred_direction.max(1, keepdim=True)[1]
        type_idx = pred_type.max(1, keepdim=True)[1]
        pred = torch.cat((color_idx, direction_idx, type_idx), dim=1)
        return pred

    def pre_process(self, image):
        """
        image formatting
        :rtype: PIL.JpegImagePlugin.JpegImageFile
        """
        # image data formatting
        if type(image) == np.ndarray:
            if image.shape[2] == 3:  # turn all 3 channels to RGB format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 1:  # turn 1 channel to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # turn numpy.ndarray into PIL.Image
            image = Image.fromarray(image)
        elif type(image) == PIL.JpegImagePlugin.JpegImageFile:
            if image.mode == 'L' or image.mode == 'I':  # turn 8bits or 32bits into 3 channels RGB
                image = image.convert('RGB')

        return image

    def predict(self, img):
        """
        predict vehicle attributes by classifying
        :return: vehicle color, direction and type
        """
        # image pre-processing
        img = self.transforms(img)
        img = img.view(1, 3, 224, 224)

        # put image data into device
        img = img.to(device)

        # calculating inference
        output = self.net.forward(img)

        # get result
        # self.get_predict_ce, return pred to host side(cpu)
        pred = self.get_predict(output)
        # color_name = self.color_attrs[pred[0][0]]
        # direction_name = self.direction_attrs[pred[0][1]]
        # type_name = self.type_attrs[pred[0][2]]

        return pred[0][0], pred[0][1], pred[0][2]

    def get_color_name(self, index):
        return self.color_attrs[index]

    def get_direction_name(self, index):
        return self.direction_attrs[index]

    def get_type_name(self, index):
        return self.type_attrs[index]


class VehicleDC:
    classifier_model_name = 'classifier.pth'
    detector_cfg_name = 'detector.cfg'
    detector_weight_name = 'detector.weights'

    def __init__(self,
                 model_path_prefix='.',
                 inp_dim=768,
                 prob_th=0.2,
                 nms_th=0.4,
                 num_classes=1):
        detector_cfg_path = os.path.join(model_path_prefix, self.detector_cfg_name)
        detector_weight_path = os.path.join(model_path_prefix, self.detector_weight_name)
        classifier_model_path = os.path.join(model_path_prefix, self.classifier_model_name)
        """
        model initialization
        """
        # super parameters
        self.inp_dim = inp_dim
        self.prob_th = prob_th
        self.nms_th = nms_th
        self.num_classes = num_classes

        # initialize vehicle detection model
        self.detector = Darknet(detector_cfg_path)
        self.detector.load_weights(detector_weight_path)
        # set input dimension of image
        self.detector.net_info['height'] = self.inp_dim
        self.detector.to(device)
        self.detector.eval()  # evaluation mode
        #print('=> car detection model initiated.')

        # initiate multilabel classifier
        self.classifier = VehicleClassifier(num_cls=19,
                                            model_path=classifier_model_path)

    def process_predict(self,
                        prediction,
                        prob_th,
                        num_cls,
                        nms_th,
                        inp_dim,
                        orig_img_size):
        """
        processing detections
        """
        scaling_factor = min([inp_dim / float(x)
                              for x in orig_img_size])  # W, H scaling factor
        output = post_process(prediction,
                              prob_th,
                              num_cls,
                              nms=True,
                              nms_conf=nms_th,
                              CUDA=use_cuda)  # post-process such as nms

        if type(output) != int:
            output[:, [1, 3]] -= (inp_dim - scaling_factor *
                                  orig_img_size[0]) / 2.0  # x, w
            output[:, [2, 4]] -= (inp_dim - scaling_factor *
                                  orig_img_size[1]) / 2.0  # y, h
            output[:, 1:5] /= scaling_factor
            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(
                    output[i, [1, 3]], 0.0, orig_img_size[0])
                output[i, [2, 4]] = torch.clamp(
                    output[i, [2, 4]], 0.0, orig_img_size[1])
        return output

    def detect_classify(self, img):
        """
        detect and classify
        """
        img2det = process_img(img, self.inp_dim)
        img2det = img2det.to(device)  # put image data to device

        # vehicle detection
        prediction = self.detector.forward(img2det, CUDA=use_cuda)

        # calculating scaling factor
        orig_img_size = list(img.size)
        output = self.process_predict(prediction,
                                          self.prob_th,
                                          self.num_classes,
                                          self.nms_th,
                                          self.inp_dim,
                                          orig_img_size)
        return self.classify(img, output)

    def result_stringify(self, car_colors, car_directions, car_types):
        return [self.classifier.get_color_name(car_color) for car_color in car_colors],\
               [self.classifier.get_direction_name(car_direction) for car_direction in car_directions],\
               [self.classifier.get_type_name(car_type) for car_type in car_types]

    def classify(self, img, output):
        """
        1. predict vehicle's attributes based on bbox of vehicle
        2. draw bbox to orig_img
        """
        car_colors = []
        car_directions = []
        car_types = []
        bounding_boxes = []

        # 1
        for det in output:
            # rectangle boudingBoxes
            x1, y1 = tuple(det[1:3].int())  # the left-up point
            x1 = int(x1)
            y1 = int(y1)
            x2, y2 = tuple(det[3:5].int())  # the right down point
            x2 = int(x2)
            y2 = int(y2)

            x = x1
            y = y1
            w = x2 - x
            h = y2 - y

            if w == 0 or h == 0:
                continue

            # print('x1: ', x1, ", y1: ", y1, ", x2: ", x2, ", y2: ", y2)

            ROI = img.crop((x1, y1, x2, y2))

            # call classifier to predict
            car_color, car_direction, car_type = self.classifier.predict(ROI)

            bounding_boxes.append((x, y, w, h))
            car_colors.append(car_color)
            car_directions.append(car_direction)
            car_types.append(car_type)

        return bounding_boxes, car_colors, car_directions, car_types

    def draw_result_to_file(self, img, points, car_colors, car_directions, car_types, dst_path):
        orig_img = cv2.cvtColor(np.asarray(
            img), cv2.COLOR_RGB2BGR)  # RGB => BGR

        color = (0, 215, 255)

        for point, car_color, car_direction, car_type in zip(points, car_colors, car_directions, car_types):
            # draw bounding box
            pt_1 = (point[0], point[1])
            pt_2 = (point[0] + point[2], point[1] + point[3])
            cv2.rectangle(orig_img, pt_1, pt_2, color, thickness=2)

            label = str(car_color + ' ' + car_direction + ' ' + car_type)

            # get str text size
            txt_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            # pt_2 = pt_1[0] + txt_size[0] + 3, pt_1[1] + txt_size[1] + 5
            pt_2 = pt_1[0] + txt_size[0] + 3, pt_1[1] - txt_size[1] - 5

            # draw text background rect
            cv2.rectangle(orig_img, pt_1, pt_2, color, thickness=-1)  # text

            # draw text
            cv2.putText(orig_img, label, (pt_1[0], pt_1[1]),  # pt_1[1] + txt_size[1] + 4
                        cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 2)

        cv2.imwrite(dst_path, orig_img)


if __name__ == '__main__':
    car_dc = VehicleDC()
    img = Image.open("D:\\git\\Vehicle-Car-detection-and-multilabel-classification\\test_imgs\\test_0.jpg")
    points, car_colors, car_directions, car_types = car_dc.detect_classify(img)
    car_dc.draw_result_to_file(img, points, car_colors, car_directions, car_types, "D:\\1.jpg")
