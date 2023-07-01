from cmath import nan
from ctypes import *
import numpy as np
import math

###############
#c++初始化
###############


def cpp_cls(x,y):
    pDLL= CDLL('/home/zhaofa/code/OpenPCDet/pcdet/models/model_utils/fuzzy_code/libfuzzy.so')#加载C++动态库
    fun=pDLL.getData
    fun.argtypes =(c_float,c_float)#
    fun.restype =c_float
    cls=[]
    for x0,y0 in zip(x,y):  #x代表体积，y代表密度，这里是做模糊分类
        # if math.isnan(fun(x0,y0)):
        #     print("ok")
        res=round(fun(x0,y0))+1
        cls.append(res)
    return np.array(cls)