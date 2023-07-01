from .detector3d_template import Detector3DTemplate
import time

class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.times=0

    def forward(self, batch_dict):
        # start = time.time()#查看运行时间
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # end = time.time()
        # print("model Running time: %s seconds"%(end - start))

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            start = time.time()#查看运行时间
            pred_dicts, recall_dicts,nms_dicts = self.post_processing(batch_dict)
            end = time.time()
            self.times+=end-start
            # print("post process Running time: %s seconds"%(end - start))

            return pred_dicts, recall_dicts,nms_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
