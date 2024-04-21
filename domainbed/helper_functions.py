import cv2
import numpy as np
import torch


def scale_np_from_tensor(cam, target_size=None):
    if target_size is None:
        result = np.zeros(cam.shape)
    else:
        result = np.zeros((target_size[0], target_size[1], cam.shape[-1]))
    for i in range(cam.shape[-1]):
        img = cam[:, :, i]
        min_ = np.min(img)
        max_ = np.max(img)
        img = img - min_
        img = img / (max_ - min_)

        if target_size is not None:
            img = cv2.resize(img, target_size)
        result[:, :, i] = img
    result = np.uint8(255*result)

    return result


def scale_feature_image(image, target_size=None):
    result = []
    for img in image:
        img = img - np.min(img)
        img = img / (np.max(img)-np.min(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


def get_acc(predicted, real):
    correct = 0
    total = predicted.size(0)
    if predicted.size(1) == 1:
        correct += (predicted.gt(0).eq(real).float()).sum().item()
    else:
        correct += (predicted.argmax(1).eq(real).float()).sum().item()
    return correct / total


def conv1_feature_before_pool(x, network):
    x = network.network.conv1(x)
    x = network.network.bn1(x)
    x = network.network.relu(x)
    return x

def conv1_feature_after_pool(x, network):
    x = conv1_feature_before_pool(x)
    x = network.network.maxpool(x)
    return x

def block1_feature(x, network):
    x = conv1_feature_after_pool(x)
    x = network.network.layer1(x)
    return x

def block2_feature(x, network):
    x = block1_feature(x)
    x = network.network.layer2(x)
    return x

def block3_feature(x, network):
    x = block2_feature(x)
    x = network.network.layer3(x)
    return x

def block4_feature(x, network):
    x = block3_feature(x)
    x = network.network.layer4(x)
    return x


def get_sample_by_class(all_y, num_class):
    indices_list = []
    for i in range(num_class):
        indices = np.argwhere(all_y == i)
        indices_list.append(indices.squeeze())
    return indices_list

def assign_domain_labels(num_domain, batch_size):
    domain = torch.LongTensor(num_domain*batch_size)
    for i in range(num_domain):
        domain[i*batch_size:(i+1)*batch_size] = i
    return domain 
