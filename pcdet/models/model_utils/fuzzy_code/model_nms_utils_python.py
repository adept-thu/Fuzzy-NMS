import torch
from ...ops.iou3d_nms import iou3d_nms_utils
from .fuzzy_code.DBSCAN_plot import DBSCAN_plot,plot_DB
from .fuzzy_code.volume_cal import volume_cal
import sys
from .fuzzy_code.fuzzy_way import fuzzy_sim
import numpy as np

####################
#fuzzy_python
####################

def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):

    #初始化模糊方法
    sim_TS=fuzzy_sim()

    src_box_scores = box_scores #获取所有框的评分,box_scores是torch.Tensor类型
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)#获取大于阈值,由true和false组成,scores_mask是torch.Tensor类型
        box_scores = box_scores[scores_mask]#只保留True的部分
        box_preds = box_preds[scores_mask]

    selected =[]
    if box_scores.shape[0] > 0:
        # topk找出box_scores中最大的k个元素,最多有4096个
        #indices为选出来的值的下标，如[351, 350, 349, 348,....
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        #上面是根据得分过滤后得到的框和框的大小，从这里开始，下面计算体积和密度
        #添加部分

        #获取体积
        dxyz=boxes_for_nms[:, 3:6]
        box_pred_volume = volume_cal(dxyz.cpu().numpy())  #每个框的体积,.cpu().numpy()：转成numpy数据
        #print("框的体积数量：",box_pred_volume.shape) #输出每个框
        #print("框的体积：",box_pred_volume[:20])  # 输出每个框


        #获取密度
        data_xyz=boxes_for_nms[:,0:3]#获取所有框的坐标
        box_pred_density=DBSCAN_plot(data_xyz.cpu().numpy())#获得密度
        #print("框的密度数量",box_pred_density.shape)
        #print("框的密度", box_pred_density[:20])

        #由密度、体积获取各个框的分类(可以按分类在空间中绘制按类别的情况)
        box_pred_cls=sim_TS.run(box_pred_volume,box_pred_density)
        #print("框的类别数量", box_pred_cls.shape)
        #print("框的类别", box_pred_cls[:20])
        #plot_DB(data_xyz.cpu().numpy(),box_pred_cls)#绘制模糊分类后的情况

        #sys.exit(0) #正常退出程序

        #对每个类别划分，得分和框
        score_thresh_two=[0.2,0.01,0.01]#每个类别（0、1、2）的得分过滤，主要是对类别0过滤掉噪声
        iou_thresh_two=[0.01,0.5,0.01]#每个类别（0、1、2）的iou过滤，主要是对类别1过滤掉质量低的框

        #重新过滤
        for i in range(3):
            if i in box_pred_cls:
                #按类别划分框体
                cls_mask=torch.from_numpy((box_pred_cls== i))#类型是numpy.ndarray，并将其转成torch，因为nonzero只能用于torch
                cls_mask_idxs=cls_mask.nonzero(as_tuple=False).view(-1)#获取非0数值的下标
                cls_scores = box_scores_nms[cls_mask]
                cls_box = boxes_for_nms[cls_mask]

                #按得分对类别过滤
                cls_scores_mask = (cls_scores >= score_thresh_two[i])
                cls_scores_mask_idxs=cls_scores_mask.nonzero(as_tuple=False).view(-1)#获取非0数值的下标
                cls_scores_nms = cls_scores[cls_scores_mask]  # 超过阈值的框的得分
                cls_box_nms =cls_box[cls_scores_mask]  # 这些框的预测结果

                #加入nms
                #keep_id：为选中框的下标，例如[ 0,  1,  6,  8,....
                keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                        cls_box_nms[:, 0:7], cls_scores_nms, iou_thresh_two[i], **nms_config
                )
                #disx:为排序前的下标,#先转numpy，后转list，方便保存
                disx=indices[cls_mask_idxs[cls_scores_mask_idxs[keep_idx[:nms_config.NMS_POST_MAXSIZE]]]].cpu().numpy().tolist()
                selected+=disx #把选中的内容保存
            else:
                continue

    if score_thresh is not None:
        #nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
        #view()返回的数据和传入的tensor一样，只是形状不同,-1:将张量变形成一维的向量形式
        original_idxs = scores_mask.nonzero(as_tuple=False).view(-1)
        selected = original_idxs[selected]

    return selected, src_box_scores[selected]

def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:为每个类别设nms阈值
        cls_scores: (N, num_class)，三个类别的框的得分
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:SCORE_THRESH: 0.1  #分数阈值

    Returns:
    num_class=3,即三种类别
    N=321408，即321408个框
    """
    pred_scores, pred_labels, pred_boxes = [], [], []

    for k in range(cls_scores.shape[1]):#三种类别依次nms
        if score_thresh is not None:#按类别设置阈值，获取得分高过阈值的框
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]#超过阈值的框的得分
            cur_box_preds = box_preds[scores_mask]#这些框的预测结果
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            #topk找出box_scores中最大的k个元素,最多有4096个
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH[k], **nms_config
            )#主要修改，根据数据的格式来动态修改SCORE_THRESH，NMS_THRESH
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])#每种类别保留的框不一致，
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
