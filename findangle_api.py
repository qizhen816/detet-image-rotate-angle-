# -*- coding: utf-8 -*-
"""
 @Time       : 2019/12/9 9:52
 @Author     : Zhen Qi
 @Email      : qizhen816@163.com
 @File       : rotate_api.py
 @Description: API Script 2 find rotate angle of a text-contained image
"""

import cv2
import numpy as np

def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    img = cv2.warpAffine(image, M, (w, h))
    return img

def rotate_points(points, angle, cX, cY):
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0).astype(np.float16)
    a = M[:, :2]
    b = M[:, 2:]
    b = np.reshape(b, newshape=(1, 2))
    a = np.transpose(a)
    points = np.dot(points, a) + b
    points = points.astype(np.int)
    return points


def findangle(_image):
    # 用来寻找当前图片文本的旋转角度 在±90度之间
    # toWidth: 特征图大小：越小越快 但是效果会变差
    # minCenterDistance：每个连通区域坐上右下点的索引坐标与其质心的距离阈值 大于该阈值的区域被置0
    # angleThres：遍历角度 [-angleThres~angleThres]

    toWidth = _image.shape[1]//2 #500
    minCenterDistance = toWidth/20 #10
    angleThres = 45

    image = _image.copy()
    h, w = image.shape[0:2]
    if w > h:
        maskW = toWidth
        maskH = int(toWidth / w * h)
    else:
        maskH = toWidth
        maskW = int(toWidth / h * w)
    # 使用黑色填充图片区域
    swapImage = cv2.resize(image, (maskW, maskH))
    grayImage = cv2.cvtColor(swapImage, cv2.COLOR_BGR2GRAY)
    gaussianBlurImage = cv2.GaussianBlur(grayImage, (3, 3), 0, 0)
    histImage = cv2.equalizeHist(~gaussianBlurImage)
    binaryImage = cv2.adaptiveThreshold(histImage, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    # pointsNum: 遍历角度时计算的关键点数量 越多越慢 建议[5000,50000]之中
    pointsNum = np.sum(binaryImage!=0)//2

    # # 使用最小外接矩形返回的角度作为旋转角度
    # # >>一步到位 不用遍历
    # # >>如果输入的图像切割不好 很容易受干扰返回0度
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # dilated = cv2.dilate(binaryImage*255, element)
    # dilated = np.pad(dilated,((50,50),(50,50)),mode='constant')
    # cv2.imshow('dilated', dilated)
    # coords = np.column_stack(np.where(dilated > 0))
    # angle = cv2.minAreaRect(coords)
    # print(angle)

    # 使用连接组件寻找并删除边框线条
    # >>速度比霍夫变换快5~10倍 25ms左右
    # >>计算每个连通区域坐上右下点的索引坐标与其质心的距离，距离大的即为线条
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImage, connectivity, cv2.CV_8U)
    labels = np.array(labels)
    maxnum = [(i, stats[i][-1], centroids[i]) for i in range(len(stats))]
    maxnum = sorted(maxnum, key=lambda s: s[1], reverse=True)
    if len(maxnum) <= 1:
        return 0
    for i, (label, count, centroid) in enumerate(maxnum[1:]):
        cood = np.array(np.where(labels == label))
        distance1 = np.linalg.norm(cood[:,0]-centroid[::-1])
        distance2 = np.linalg.norm(cood[:,-1]-centroid[::-1])
        if distance1 > minCenterDistance or distance2 > minCenterDistance:
            binaryImage[labels == label] = 0
        else:
            break
    cv2.imshow('after process', binaryImage*255)

    minRotate = 0
    minCount = -1
    (cX, cY) = (maskW // 2, maskH // 2)
    points = np.column_stack(np.where(binaryImage > 0))[:pointsNum].astype(np.int16)
    for rotate in range(-angleThres, angleThres):
        rotatePoints = rotate_points(points, rotate, cX, cY)
        rotatePoints = np.clip(rotatePoints[:,0], 0, maskH-1)
        hist, bins = np.histogram(rotatePoints, maskH, [0, maskH])
        # 横向统计非零元素个数 越少则说明姿态越正
        zeroCount = np.sum(hist > toWidth/50)
        if zeroCount <= minCount or minCount == -1:
            minCount = zeroCount
            minRotate = rotate

    # print("over: rotate = ", minRotate)
    return minRotate

if __name__ == '__main__':
    import time
    Path = 'testrotate.jpg'
    cv_img = cv2.imdecode(np.fromfile(Path, dtype=np.uint8), -1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    for agl in range(-60, 60):
        img = cv_img.copy()
        img = rotate_bound(img, agl)
        cv2.imshow('rotate', img)
        t = time.time()
        angle = findangle(img)
        print(agl, angle,'Time:' , time.time()-t)
        img = rotate_bound(img, -angle)
        cv2.imshow('after', img)
        cv2.waitKey(00)




