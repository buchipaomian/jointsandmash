#coding=utf-8
import cv2
import numpy as np
import math
from delaunay2D import Delaunay2D
import json
from generate_mask import *

import matplotlib.pyplot as plt
import matplotlib.tri
import matplotlib.collections

from simple_weight import *
from datetime import datetime

import random as rand
"""

点的生成抄了大佬的项目
https://github.com/MauriceGit/Delaunay_Triangulation
大佬把这个项目抛弃了是有道理的，真的逻辑太烂了
但是生成点的过程其实就是先转灰度然后求边缘，这样每一个边缘都是黑色，近边缘区域是灰色，不是边缘的地区就几乎为白色
接下来用随机生成，如果是黑色就有一半的几率产生点，灰色的概率大大降低，白色几乎不会有点

三角的生成抄了另一个大佬的项目
https://github.com/jmespadero/pyDelaunay2D
用的是v开头的一个插入算法，简单来说就是画一个超大的三角形把所有点都包进去，然后一个个加点，加点的同时求其外点在不在这个点阵里面，在的话下一个就插入外点，以此递归
最后删除多余的点
"""
def find_points(img,factor):
    """
    this is totally copied from old test
    """
    pass
def loadAndFilterImage(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    img_edge = cv2.Canny(img_blur,300,700)

    img_brighter = np.uint8(np.clip((img_edge + 20), 0, 255))
    img_blur2 = cv2.GaussianBlur(img_brighter,(5,5),0)

    return (img,img_blur2)

def findPointsFromImage(img, factor):
    points = []
    for row in range(len(img)):
        for col in range(len(img[row])):
            v = img[row][col]
            v = v**float(factor)/float(2**18)
            if np.random.random() < v:
                points.append([col, row])
    return points

def pure_net_format(count,width,height):
    points = []
    l = float(width)/float(height)
    print(count/l)
    k = int(math.sqrt(count/l))
    print(k)
    distance = height//k
    for i in range(k):
        for j in range(int(k*l)):
            points.append([j*distance,i*distance])
    return points,distance

def generateRandomPoints(count, sizeX, sizeY):
    points = []
    for i in range(count):
        p = [rand.randint(0,sizeX),rand.randint(0,sizeY)]
        if not p in points:
            points.append(p)
    # # print "Punkte generieren: %.2fs" % (time.clock()-start)
    return points

def generate_mash_2d(img,result_file_name,masks=None,point_list=None):
    (colorIm, blackIm) = loadAndFilterImage(img)
    """
    原项目这里有三个选项，分别对应提供了预设点/创建三角图片/创建voronoi图，这里省略这些选项，仅计算生成的节点
    但是可以提供mask，提供mask的目的是过滤掉本来没什么用的那些外部点（因为选取的是不规则图案，获得的是最小外接正方形）
    注意这里有一个大问题：cv2读取的图片与生成的mask按照惯例是h,w的格式
    而生成的点是(x,y)即(w,h)的格式
    还有，记得要删除所有的中文注释，因为有些情况下中文注释在服务器上会报错
    """
    #下面的是依托边缘生成点
    # temp_points = findPointsFromImage(blackIm, 2.1)#To-do:这个2.1算是个按照比例去求随机点的方法,2.1的时候边缘生成点的比例接近0.45
    #下面的是过去的大佬搞得纯随机
    h,w = img.shape[:2]
    temp_points = generateRandomPoints(1500, w-1, h-1)#原作者用的pil，所以差1
    # 下面的是没有随机直接上网格
    # h,w = img.shape[:2]
    # temp_points,_ = pure_net_format(2000,w,h)#2000约等于密度


    final_points = []
    #filter points
    if masks is not None:
        for p in temp_points:
            if masks[p[1]][p[0]] == 1:
                final_points.append(p)
    else:
        final_points = temp_points
    if point_list is not None:
        #这一步会把我们选取的几个点也加进去
        for p in point_list:
            if not p in final_points:
                final_points = [p] + final_points

    dt = Delaunay2D()
    for i,p in enumerate(final_points):
        dt.addPoint(p)
    triangles = dt.exportTriangles()
    #想了一下好像直接判断也很快，因为如果两个点不该连起来但是连起来了，那这两个点一定都是边缘点，那这两个点的中点一定在外部
    #这样n个三角形一次扫描就删干净了，只要不是那种特别曲折点特别少的应该都没有问题
    if masks is not None:
        temp_triangles = []
        for k in triangles:
            x0 = (final_points[k[0]][0] + final_points[k[1]][0])//2
            y0 = (final_points[k[0]][1] + final_points[k[1]][1])//2
            x1 = (final_points[k[1]][0] + final_points[k[2]][0])//2
            y1 = (final_points[k[1]][1] + final_points[k[2]][1])//2
            x2 = (final_points[k[0]][0] + final_points[k[2]][0])//2
            y2 = (final_points[k[0]][1] + final_points[k[2]][1])//2
            if masks[y0][x0]*masks[y1][x1]*masks[y2][x2] != 0:
                temp_triangles.append(k)
        triangles = temp_triangles
    result = {}
    result["vertices"] = final_points
    result["faces"] = triangles
    with open(result_file_name,'w') as result_file:
        json.dump(result,result_file)
    return result,final_points,triangles

def decide_belong_to(points,triangles,point_list):
    """
    这个会引用simple_weight.py里面计算权重的部分
    points是meshpoint，point_list是关节点
    """
    weight_result = []
    result = {}
    for p in point_list:
        k = points.index(p)
        weight_result.append(count_weight_limited(triangles,k))
        result[k] = []
    print(weight_result)
    for i,p in enumerate(points):
        weights = []
        for m in weight_result:
            if i in m.keys():
                weights.append(m[i])
            else:
                weights.append(9999)
        min_weight = min(weights)
        result[weights.index(min_weight)].append(p)

    return list(result.values())


if __name__ == "__main__":
    input_img_name = "cut.png"
    res_img = cv2.imread(input_img_name,cv2.IMREAD_UNCHANGED)
    point_list = [[1109,480],[223,795],[1112,1009]]
    mask_channel = res_img[:,:,-1]//255
    result_file_name = "result.json"
    imput_img = cv2.imread(input_img_name)
    result,final_points,triangles = generate_mash_2d(imput_img,result_file_name,mask_channel,point_list)
    print(datetime.now())
    #接下来就是决定谁拿哪个点的时刻，第一步是计算每一个点的权重
    decide_result = decide_belong_to(final_points,triangles,point_list)
    #下面就是输出看一下结果，这个decide结果就是一个length和关节点数量一样的，按照给定的关节点顺序给出每个点的点集
    print("#################################")
    print(datetime.now())
    # print(deside_result)

    colors = [(255,182,193),(0,0,255),(255,255,0)]
    for i,k in enumerate(decide_result):
        for m in k:
            cv2.circle(res_img,(m[0],m[1]),3,colors[i],3)
    cv2.namedWindow('img',0)
    cv2.imshow('img',res_img)
    cv2.waitKey()


"""
{
    "vertices" :[[1,1],[2,2],[1,3]],
    "faces":[[0,1,2],[0,1,3]]
}
"""
