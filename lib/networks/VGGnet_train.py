import tensorflow as tf
from networks.network import Network


#define

n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('target/bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='target/conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='target/conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='target/pool1')
             .conv(3, 3, 128, 1, 1, name='target/conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='target/conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='target/pool2')
             .conv(3, 3, 256, 1, 1, name='target/conv3_1')
             .conv(3, 3, 256, 1, 1, name='target/conv3_2')
             .conv(3, 3, 256, 1, 1, name='target/conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='target/pool3')
             .conv(3, 3, 512, 1, 1, name='target/conv4_1')
             .conv(3, 3, 512, 1, 1, name='target/conv4_2')
             .conv(3, 3, 512, 1, 1, name='target/conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='target/pool4')
             .conv(3, 3, 512, 1, 1, name='target/conv5_1')
             .conv(3, 3, 512, 1, 1, name='target/conv5_2')
             .conv(3, 3, 512, 1, 1, name='target/conv5_3'))
        #========= RPN ============
        (self.feed('target/conv5_3')
             .conv(3,3,512,1,1,name='target/rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='target/rpn_cls_score'))

        (self.feed('target/rpn_cls_score','gt_boxes','im_info','data')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'target/rpn-data' ))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('target/rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, name='target/rpn_bbox_pred'))

        #========= RoI Proposal ============
        (self.feed('target/rpn_cls_score')
             .reshape_layer(2,name = 'target/rpn_cls_score_reshape')
             .softmax(name='target/rpn_cls_prob'))

        (self.feed('target/rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'target/rpn_cls_prob_reshape'))

        (self.feed('target/rpn_cls_prob_reshape','target/rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'target/rpn_rois'))

        (self.feed('target/rpn_rois','gt_boxes')
             .proposal_target_layer(n_classes,name = 'target/roi-data'))


        #========= RCNN ============
        (self.feed('target/conv5_3', 'target/roi-data')
             .roi_pool(7, 7, 1.0/16, name='target/pool_5')
             .fc(4096, name='target/fc6')
             .dropout(0.5, name='target/drop6')
             .fc(4096, name='target/fc7')
             .dropout(0.5, name='target/drop7')
             .fc(n_classes, relu=False, name='target/cls_score')
             .softmax(name='target/cls_prob'))

        (self.feed('target/drop7')
             .fc(n_classes*4, relu=False, name='target/bbox_pred'))

