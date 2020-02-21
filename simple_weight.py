import numpy as np

def lope_distance(start_point,triangles,end_point):
    #this will be a really compeleted graph method
    return 0
def count_weight(points,triangles,start_point):
    weights = []
    total = 0
    for k in points:
        distance = lope_distance(start_point,triangles,k)
        total += distance
        weights.append(distance)
    if total == 0:
        return []
    result = weights/total
    return result


def count_weight_limited(triangles,start_point):
    #只有这一段代码是有用的，所以其它的部分不要管他
    #大概的逻辑是，计算每一个点距离目标点的距离，由于我们用的三角形插入模式可以使点整体较为均匀，所以这个距离可以用两点之间最短路径来替代
    weights = {}
    weights[start_point] = 0.0
    tris = triangles.copy()
    k = len(tris)
    while len(tris) != 0:
        temp = tris.copy()
        for i in tris:
            if i[0] in weights.keys():
                if i[1] in weights.keys():
                    weights[i[1]] = min(weights[i[0]]+1,weights[i[1]])
                else:
                    weights[i[1]] = weights[i[0]]+1
                if i[2] in weights.keys():
                    weights[i[2]] = min(weights[i[0]]+1,weights[i[2]])
                else:
                    weights[i[2]] = weights[i[0]]+1
            if i[1] in weights.keys():
                if i[0] in weights.keys():
                    weights[i[0]] = min(weights[i[1]]+1,weights[i[0]])
                else:
                    weights[i[0]] = weights[i[1]]+1
                if i[2] in weights.keys():
                    weights[i[2]] = min(weights[i[1]]+1,weights[i[2]])
                else:
                    weights[i[2]] = weights[i[1]]+1
            if i[2] in weights.keys():
                if i[1] in weights.keys():
                    weights[i[1]] = min(weights[i[2]]+1,weights[i[1]])
                else:
                    weights[i[1]] = weights[i[2]]+1
                if i[2] in weights.keys():
                    weights[i[1]] = min(weights[i[2]]+1,weights[i[1]])
                else:
                    weights[i[1]] = weights[i[2]]+1
            if i[0] in weights.keys() or i[1] in weights.keys() or i[2] in weights.keys():
                temp.remove(i)
        m = len(temp)
        if m == k:
            print("error ocured")
            tris = []
        else:
            k = m
            tris = temp
    total = 0
    print(weights)
    for k in weights:
        total += weights[k]
    if total == 0:
        return []
    for k in weights:
        weights[k] = weights[k]/total
    return weights