"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch

from ...utils import common_utils
from . import iou3d_nms_cuda



def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou




def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou




#返回3D框的iou
def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d




#使用iou的nms
def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]#按列排序，返回值为排序后的下标
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()#返回一个内存连续的有相同数据的tensor，如果原tensor内存连续则返回原tensor
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)#这里会报Error!,当输入框的大小为0时会报。

    return order[keep[:num_out].cuda()].contiguous(), None




def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None




def soft_nms(boxes, scores, thresh, pre_maxsize=None,sigma=0.5, top_k=-1):
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]#按列排序，返回值为排序后的下标
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()#返回一个内存连续的有相同数据的tensor，如果原tensor内存连续则返回原tensor
    scores=scores[order].contiguous()
    
    keep=[]
    while boxes.size(0) > 0:
        max_score_index = torch.argmax(scores)
        keep.append(order[max_score_index])

        if len(keep) == top_k > 0 or boxes.size(0) == 1:
            break
        
        cur_box = boxes[max_score_index, :]
        boxes[max_score_index, :] = boxes[-1, :]
        boxes = boxes[:-1, :]
        scores[max_score_index] = scores[-1]
        scores= scores[:-1]
        order[max_score_index] = order[-1]
        order= order[:-1]

        ious =boxes_iou3d_gpu(cur_box.unsqueeze(0), boxes[:, 0:7])
 
        #如果没有这句就是Hard-NMS了
        scores= scores * torch.exp(-(ious * ious) / sigma) 
 
        boxes = boxes[scores[0] > thresh]
        order= order[scores[0]> thresh]
        scores = scores[scores > thresh]
        

    if len(keep) > 0:
        return torch.tensor(keep).cuda().contiguous(),None
    else:
        return torch.tensor([]),None




#返回3D框的Diou
def boxes_Diou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    #中心点距离的平方
    d2=((boxes_a[:,:3]-boxes_b[:,:3])**2).sum(dim=1)+1e-16

    #包含两个box的最小box的对角线长度
    left_top=torch.tensor([-1,1,1]).cuda()
    right_bottom=torch.tensor([1,-1,-1]).cuda()
    c2=(((boxes_a[:,:3]+left_top*boxes_a[:,3:6])-(boxes_b[:,:3]+right_bottom*boxes_b[:,3:6]))**2).sum(dim=1)+1e-16

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)-d2/c2
    # print("iou3d",iou3d)

    return iou3d




#使用Diou的nms
def hard_nms_Diou(boxes, scores, thresh, top_k=-1,pre_maxsize=None):
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]#按列排序，返回值为排序后的下标
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()#返回一个内存连续的有相同数据的tensor，如果原tensor内存连续则返回原tensor
    scores=scores[order].contiguous()
    
    keep=[]
    while len(order) > 0:
        keep.append(order[0])         # 将这个最大的存在结果中
        if 0 < top_k == len(keep) or len(order) == 1:
            break
        cur_box = boxes[0,:]       # 当前第一个也就是最高概率的box
        order = order[1:]     
        boxes = boxes[1:,:]        # 剩下其余的box
        ious =boxes_Diou3d_gpu(cur_box.unsqueeze(0), boxes[:, 0:7])

        boxes = boxes[ious[0]<= thresh]
        order= order[ious[0]<= thresh]
 
    if len(keep) > 0:
        return torch.tensor(keep).cuda().contiguous(),None
    else:
        return torch.tensor([]),None


