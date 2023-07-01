该部分用来储存KITTI下的模型文件（.pth）


测试时的代码如下：
注意：test.py文件可以选择GPU

pointpillars（libfuzzy.so、score thresh：0.1、iou thresh：0.01）
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt /home/zhaofa/code/OpenPCDet/models/pointpillar.pth --extra_tag nms_test


second（libfuzzy.so、score thresh：0.1、iou thresh：0.01）
python test.py --cfg_file cfgs/kitti_models/second.yaml --batch_size 4 --ckpt /home/zhaofa/code/OpenPCDet/models/second_7862.pth --extra_tag nms_test


second_iou（libfuzzy.so、score thresh：0.1、iou thresh：0.01）
python test.py --cfg_file cfgs/kitti_models/second_iou.yaml --batch_size 4 --ckpt /home/zhaofa/code/OpenPCDet/models/second_iou7909.pth --extra_tag nms_test


pointrcnn(roi head、libfuzzy.so、score thresh：0.1、iou thresh：0.1) 
python test.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size 4 --ckpt /home/zhaofa/code/OpenPCDet/models/pointrcnn_7870.pth --extra_tag nms_test


pointrcnn_iou(roi head、libfuzzy.so、score thresh：0.1、iou thresh：0.1) 
python test.py --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml --batch_size 4 --ckpt /home/zhaofa/code/OpenPCDet/models/pointrcnn_iou_7875.pth --extra_tag nms_test



pv_rcnn(roi head、libfuzzy.so、score thresh：0.1、iou thresh：0.1)
python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 4 --ckpt /home/zhaofa/code/OpenPCDet/models/pv_rcnn_8369.pth --extra_tag nms_test


PartA2(roi head、libfuzzy.so、score thresh：0.1、iou thresh：0.1)
python test.py --cfg_file cfgs/kitti_models/PartA2.yaml --batch_size 4 --ckpt /home/zhaofa/code/OpenPCDet/models/PartA2_7940.pth --extra_tag nms_test


PartA2_free(roi head、libfuzzy.so、score thresh：0.1、iou thresh：0.1)
python test.py --cfg_file cfgs/kitti_models/PartA2_free.yaml --batch_size 4 --ckpt /home/zhaofa/code/OpenPCDet/models/PartA2_free_7872.pth --extra_tag nms_test
