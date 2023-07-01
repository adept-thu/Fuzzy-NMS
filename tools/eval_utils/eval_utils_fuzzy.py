import pickle
import time
import sys

import numpy as np
import torch
import tqdm
import copy
import json
import pickle


from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.model_utils.fuzzy_nms import model_nms_utils_cpp #fuzzy nms

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:#dist_test在测试时，通常为false
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    nms_dicts_all = []#添加，用来保存模型推理得到的结果
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    start = time.time()#查看运行时间
    for i, batch_dict in enumerate(dataloader):
        #i是第i个批次(batach size=4)的数据，其中共有943个批次
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            # pred_dicts, ret_dict = model(batch_dict)#原始
            pred_dicts, ret_dict,nms_dicts = model(batch_dict) #更改，bat_dict为一次推理的结果
        
        nms_dicts_all.append(nms_dicts)#添加，推理结果
        disp_dict = {}
        
        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()
    end = time.time()#结束时间
    print("model Running and nms time: %s seconds"%(end - start))#输出时间
    
    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')


    #遍历fuzzy nms
    nms_dicts_all_to_gpu(nms_dicts_all)#将数据传输到GPU
    ergodic_score_iou(cfg,nms_dicts_all,dataloader,result_dict)#遍历
    return ret_dict




def ergodic_score_iou(cfg,nms_dicts_all,dataloader,result_dict):
    err=0.500 #误差

    #这里是原始baseline的结果，保存下来用来做对比
    baseline_carR40=[result_dict['Car_3d/easy_R40'],result_dict['Car_3d/moderate_R40'],result_dict['Car_3d/hard_R40']]
    baseline_PedestrianR40=[result_dict['Pedestrian_3d/easy_R40'],result_dict['Pedestrian_3d/moderate_R40'],result_dict['Pedestrian_3d/hard_R40']]
    baseline_CyclistR40=[result_dict['Cyclist_3d/easy_R40'],result_dict['Cyclist_3d/moderate_R40'],result_dict['Cyclist_3d/hard_R40']]
    baseline_allR40=torch.tensor(baseline_carR40+baseline_PedestrianR40+baseline_CyclistR40)-err
    # print("baseline_carR40",baseline_carR40)
    # print("baseline_PedestrianR40",baseline_PedestrianR40)
    # print("baseline_CyclistR40",baseline_CyclistR40)

    #score一般是0.1，iou一般是0.01，更加细粒度了。不需要从0开始，只会白白增加工作量，因为有基础的score和iou早就过滤结束了
    for _0_iou in np.arange(0.0,0.6,0.1):
        for _1_iou in np.arange(0.0,0.8,0.1):#循环到0.6，类别1为自行车和行人，score本身普遍低
            for _2_score in np.arange(0.1,0.8,0.1):#当yaml文件中的iou为0.01时，从0.0开始循环，0.0即0.01，为0.1时从0.1开始循环即可
                score=[0.1,0.1,_2_score]
                iou=[_0_iou,_1_iou,0.0]
                path="nms_output/pointrcnn_iou_final.txt"
                print("score:",score,"  iou:",iou)#输出过程
                print("保存路径：",path)
                
                #初始化score和iou
                model_nms_utils_cpp._init_score_iou(score,iou)
                # start evaluation,返回值是预测的结果
                pre_result=nms_eval_boundding_box(cfg,nms_dicts_all,dataloader)
                
                carR40=[pre_result['Car_3d/easy_R40'],pre_result['Car_3d/moderate_R40'],pre_result['Car_3d/hard_R40']]
                PedestrianR40=[pre_result['Pedestrian_3d/easy_R40'],pre_result['Pedestrian_3d/moderate_R40'],pre_result['Pedestrian_3d/hard_R40']]
                CyclistR40=[pre_result['Cyclist_3d/easy_R40'],pre_result['Cyclist_3d/moderate_R40'],pre_result['Cyclist_3d/hard_R40']]
                allR40=torch.tensor(carR40+PedestrianR40+CyclistR40)
                if(torch.all(allR40>baseline_allR40)):
                    file = open(path, 'a')
                    file.write("score: "+str(score)+'\t')
                    file.write("iou: "+str(iou)+'\t')
                    file.write("carR40: "+str(carR40)+'\t')
                    file.write("PedestrianR40: "+str(PedestrianR40)+'\t')
                    file.write("CyclistR40: "+str(CyclistR40)+'\t')
                    file.write("delta: "+str((allR40-baseline_allR40-err).numpy())+'\n')#保存的的是真实误差
                    file.close()
                    print("find!!!!!!!!!!!")


#model_cfg=cfg.MODEL
def nms_eval_boundding_box(cfg,nms_dicts_all,dataloader):
    dataset = dataloader.dataset
    class_names = dataset.class_names
    
    start = time.time()#开始记录的时间
    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    det_annos = []
    for i, batch_dict in enumerate(dataloader):
        nms_dicts = nms_dicts_all[i]
        load_data_to_gpu(batch_dict)#必须转到GPU
        pred_dicts=nms_post_processing(cfg.MODEL,nms_dicts)#原始
        
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=None
        )
        det_annos += annos
        progress_bar.update(1)
    progress_bar.close()
    end = time.time()#结束时间
    print("only nms Running time: %s seconds"%(end - start))#输出总的运行时间

    #对得到的结果进行评价，result_dict_new是得到的评价结果
    result_str, result_dict_new = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=None
    )
    
    print(result_dict_new)
    return result_dict_new


def nms_post_processing(cfg_model,nms_dicts):
    pred_dicts = []
    post_process_cfg = cfg_model.POST_PROCESSING
    for nms_dict in nms_dicts:
        cls_preds=nms_dict["nms_scores"]
        label_preds=nms_dict["nms_labels"]
        box_preds=nms_dict['nms_boxes']

        selected, selected_scores,unuse = model_nms_utils_cpp.class_agnostic_nms(
            box_scores=cls_preds, box_preds=box_preds,
            nms_config=post_process_cfg.NMS_CONFIG,
            score_thresh=post_process_cfg.SCORE_THRESH
        )

        final_scores = selected_scores
        final_labels = label_preds[selected]
        final_boxes = box_preds[selected]

        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels
        }

        pred_dicts.append(record_dict)
    return pred_dicts


#将nms_dicts_all转到GPU
def nms_dicts_all_to_gpu(nms_dicts_all):
    for nms_dicts in nms_dicts_all:
        for nms_dict in nms_dicts:
            nms_dict["nms_scores"]=nms_dict["nms_scores"].cuda()
            nms_dict["nms_labels"]=nms_dict["nms_labels"].cuda()
            nms_dict['nms_boxes']=nms_dict['nms_boxes'].cuda()



if __name__ == '__main__':
    pass
