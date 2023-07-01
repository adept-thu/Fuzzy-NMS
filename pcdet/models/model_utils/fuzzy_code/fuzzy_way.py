# -*- coding: utf-8 -*-
# @Time    : 2021/9/24 19:56
# @Author  : ZFC
# @File    : fuzzy_way.py
# @Task    :
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl


class fuzzy_sim():
    def __init__(self):
        #设置各个变量范围：
        self.in_volume_range = np.arange(0, 35, 1, np.float32)#体积取值（0，25）
        self.in_density_range = np.arange(0, 1.1, 0.1, np.float32)#密度取值（0，1）
        self.out_thresh_range = np.arange(0, 1.1, 0.1, np.float32)#iou阈值取（0，1.1）

        # 创建模糊控制变量
        self.in_volume = ctrl.Antecedent(self.in_volume_range, 'volume')
        self.in_density = ctrl.Antecedent(self.in_density_range, 'density')
        self.out_thresh = ctrl.Consequent(self.out_thresh_range, 'thresh')

        # 定义模糊集和其隶属度函数(三角函数)
        #volume（小）=N，volume（中）=M，volume（大）=P
        #density（小）=N，density（中）=M，density（大）=P
        #thresh（小）=N，thresh（中）=M，thresh（大）=P


        # 定义模糊集和其隶属度函数(隶属度函数可以调整)
        self.in_volume['N'] = fuzz.trimf(self.in_volume_range, [0, 0, 6])#行人与自行车体积[-1,-1,-0.6571]
        self.in_volume['P'] = fuzz.trimf(self.in_volume_range, [4, 12, 35])#汽车体积[-0.771,-0.3145,0.3714] [0.35,1,1]

        self.in_density['N'] = fuzz.trimf(self.in_density_range, [0, 0, 0.1])#小密度[-1,-1,-0.8]
        self.in_density['M'] = fuzz.trimf(self.in_density_range, [0.1, 0.2, 0.7])#中等密度,[-0.81,-0.6,0.4]
        self.in_density['P'] = fuzz.trimf(self.out_thresh_range, [0.5, 1.0, 1.0])#大密度,[0,1,1]

        self.out_thresh['N'] = fuzz.trimf(self.out_thresh_range, [0, 0, 0.3])#小密度    label:0
        self.out_thresh['M'] = fuzz.trimf(self.out_thresh_range, [0.3, 0.5, 0.7])#非小密度，行人与自行车体积   label:1
        self.out_thresh['P'] = fuzz.trimf(self.out_thresh_range, [0.7, 1.0, 1.0])#非小密度，汽车体积   label:2

        # 设定输出thresh的解模糊方法——质心解模糊方式
        self.out_thresh.defuzzify_method='centroid'

        #步骤3.建立模糊控制规则，并初始化控制系统和运行环境。（&：与，|：或）
        # 输出为N的规则
        self.ruleN = ctrl.Rule(antecedent=((self.in_volume['N'] & self.in_density['N']) |
                                           (self.in_volume['P'] & self.in_density['N'])),
                               consequent=self.out_thresh['N'], label='rule N')
        # 输出为P的规则
        self.ruleM = ctrl.Rule(antecedent=((self.in_volume['N'] & self.in_density['M']) |
                                           (self.in_volume['N'] & self.in_density['P'])),
                               consequent=self.out_thresh['M'], label='rule M')
        # 输出为P的规则
        self.ruleP = ctrl.Rule(antecedent=((self.in_volume['P'] & self.in_density['M']) |
                                           (self.in_volume['P'] & self.in_density['P'])),
                               consequent=self.out_thresh['P'], label='rule P')

        # 系统和运行环境初始化
        self.system = ctrl.ControlSystem(rules=[self.ruleN, self.ruleM,self.ruleP])
        self.sim = ctrl.ControlSystemSimulation(self.system)

    def run(self,vol,den):#输入vol和den必须为矩阵：[]
        self.cls=[]
        assert len(vol)==len(den), '密度和体积的长度要保持一致'#表达式为假时，抛出AssertionError错误，并将输出信号“   ”
        for i in range(len(vol)):
            self.sim.input['volume'] =vol[i]
            self.sim.input['density'] =den[i]
            self.sim.compute()  #运行系统
            self.output_powder = self.sim.output['thresh']
            if self.output_powder<0.3:
                self.cls.append(0)
            elif self.output_powder>0.7:
                self.cls.append(2)
            else:
                self.cls.append(1)
        return np.array(self.cls) #self.cls为分类的结果，0：out_thresh['N']，1：out_thresh['M']，2：out_thresh['P']

#sim_example=fuzzy_sim()
#print(sim_example.run([3,6,12,5],[0.1,0.4,0.02,0.8]))#输入必须为矩阵,返回矩阵