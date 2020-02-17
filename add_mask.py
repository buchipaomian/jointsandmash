import cv2
import numpy as np
import math
from delaunay2D import Delaunay2D
import json


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


def generate_mash_2d(img,result_file_name,masks=None):
    (colorIm, blackIm) = loadAndFilterImage(img)
    """
    原项目这里有三个选项，分别对应提供了预设点/创建三角图片/创建voronoi图，这里省略这些选项，仅计算生成的节点
    但是可以提供mask，提供mask的目的是过滤掉本来没什么用的那些外部点（因为选取的是不规则图案，获得的是最小外接正方形）
    注意这里有一个大问题：cv2读取的图片与生成的mask按照惯例是h,w的格式
    而生成的点是(x,y)即(w,h)的格式
    还有，记得要删除所有的中文注释，因为有些情况下中文注释在服务器上会报错
    """
    temp_points = findPointsFromImage(blackIm, 2.1)#To-do:这个2.1算是个按照比例去求随机点的方法,2.1的时候边缘生成点的比例接近0.45
    final_points = []
    #filter points
    if masks:
        for p in points:
            if masks[p[1],p[0]] == 1:
                final_points.append(p)
    else:
        final_points = temp_points

    dt = Delaunay2D()
    for p in final_points:
        dt.addPoint(p)
    triangles = dt.exportTriangles()
    result = {}
    result["vertices"] = final_points
    result["faces"] = triangles
    with open(result_file_name,'w') as result_file:
        json.dump(result,result_file)
    return result

if __name__ == "__main__":
    input_img_name = "test.png"
    result_file_name = "result.json"
    imput_img = cv2.imread(input_img_name)
    result = generate_mash_2d(imput_img,result_file_name)
    print(result)


"""
{
    "vertices" :[[1,1],[2,2],[1,3]],
    "faces":[[0,1,2],[0,1,3]]
}
"""