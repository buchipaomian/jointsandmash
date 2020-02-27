"""
这个计划是这样的：计算点集中每一个点的weight，然后看看计算的时间有多久。
其实这个结果是可以预估的，既然我们可以知道模型输入图片的大小，那我们直接就可以用查表的方式获取每一个点的所以权重。
这里其实有一个非常好用的方案，但是数学实现比较复杂，所以暂时不用，大概思路是使用二重积分计算一个图上每一个点的总距离值，这样就可以快速计算出来某个点的特定位置权重，但该方案编程实现后未必比直接算快。
最快的思路是一次性将所有点的权重全部计算出来，目前来看可能需要接近1000M的内存和512**2*0.4秒的前置准备时间，但是使用python的速度会使得处理更多图片时这个速度更快，单张图片的计算时间会被大大减少（从n*0.4加速到n*512*512*hash访问时间）
"""
import cv2
import numpy as np
from datetime import datetime
import math
import random

def count_weight(point,area):
    h,w = area.shape[:2]
    x = point[1]
    y = point[0]
    result = area.copy()
    total = 0
    for i in range(h):
        for j in range(w):
            distance = math.sqrt((i - y)**2+(j-x)**2)
            total += distance
            result[i,j] = distance
    # print(result)
    result = result/total
    return result,total

def get_point(img):
    hint_number_mu = 10
    hint_number_sigma = 5
    sample_number = 1000
    samples =  np.clip(np.random.normal(hint_number_mu, hint_number_sigma, sample_number), 0, 100)
    # image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    image = img.copy()
    result = np.zeros(image.shape).astype(np.uint8)

    h,w,_ = image.shape
    final_points = []
    hint_number = int(samples[random.randint(0,sample_number-1)])
    # print(hint_number)
    for i in range(hint_number):
        #patch size 
        p = int(max(min(h/100.0,w/100.0),2))
        # sample location
        y = int(np.clip(np.random.normal((h-p)/2., (h-p)/4.), 0, h-p-1))
        x = int(np.clip(np.random.normal((w-p)/2., (w-p)/4.), 0, w-p-1))

        # add color point
        patch = image[y:y+p,x:x+p,:]
        patch = patch.reshape((-1,3))
        color = np.mean(patch,axis=0)
        # print(x,y)
        # cv2.circle(result,(x,y),2,(int(color[0]),int(color[1]),int(color[2])),-1)
        final_points.append([y,x,color])


    # return Image.fromarray(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    return final_points

def decide_belong_to(point_list,area = np.zeros((512,512))):
    """
    决定地图上每一个色块的颜色，
    这一步会需要大量的优化。
    """
    weight_result = []
    result = {}
    for p in point_list:
        result,_ = count_weight([p[0],p[1]],area)
        weight_result.append(result)
    # print(weight_result)
    h,w = area.shape[:2]
    colored_img = np.array([area,area,area])#this is the final result
    for i in range(h):
        for j in range(w):
            weights = []
            for m in weight_result:
                weights.append(m[i][j])
            min_weight = min(weights)
            index = weights.index(min_weight)
            colored_img[0][i][j] = point_list[index][2][0]
            colored_img[1][i][j] = point_list[index][2][1]
            colored_img[2][i][j] = point_list[index][2][2]
    return colored_img.transpose((1,2,0))

def generate_tables(area = np.zeros((512,512))):
    h,w = area.shape[:2]
    result_group = []
    for i in range(h):
        temp_result = []
        for j in range(w):
            result,total = count_weight([i,j],area)
            temp_result.append(result)
        result_group.append(temp_result)
    return result_group
"""
if __name__ == "__main__":
    print(datetime.now())
    area = np.zeros((512,512))
    point = [123,456]
    result_group = []
    # h,w = area.shape[:2]
    # for i in range(h):
    #     temp_result = []
    #     for j in range(w):
    #         result,total = count_weight([i,j],area)
    #         temp_result.append(result)
    #     result_group.append(temp_result)
    result,total = count_weight(point,area)
    print(result)
    print("##########################")
    print(total)
    print(datetime.now())
"""
if __name__ == "__main__":
    print(datetime.now())
    img = cv2.imread("test.png")
    k = get_point(img)
    print(k)
    result_img = decide_belong_to(k)
    cv2.imwrite("jjjjjjjjj.png",result_img)
    print(datetime.now())
