"""Training and testing unbiased learning to rank algorithms.

See the following paper for more information about different algorithms.
    
    * Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18
    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf   # zcr
# tf.disable_v2_behavior()    # zcr: for tackling the error "AttributeError: module 'tensorflow' has no attribute 'placeholder'"
import json
# import ultra  # zcr
from termcolor import cprint
from utils.data_utils import read_data, merge_TFSummary, output_ranklist
from utils.sys_tools import find_class

from utils.common_tools import print_dict

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # zcr --> useless

# import warnings
# warnings.filterwarnings('ignore')   # zcr --> useless


#rank list size should be read from data
tf.app.flags.DEFINE_string("data_dir", "./tests/data/", "The directory of the experimental dataset.")
tf.app.flags.DEFINE_string("train_data_prefix", "train", "The name prefix of the training data in data_dir.")
tf.app.flags.DEFINE_string("valid_data_prefix", "valid", "The name prefix of the validation data in data_dir.")
tf.app.flags.DEFINE_string("test_data_prefix", "test", "The name prefix of the test data in data_dir.")
tf.app.flags.DEFINE_string("model_dir", "./tests/tmp_model/", "The directory for model and intermediate outputs.")
tf.app.flags.DEFINE_string("output_dir", "./tests/tmp_output/", "The directory to output results.")

# model 
tf.app.flags.DEFINE_string("setting_file", "./example/offline_setting/dla_exp_settings.json", "A json file that contains all the settings of the algorithm.")

# general training parameters
tf.app.flags.DEFINE_integer("batch_size", 256,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("max_list_cutoff", 0,
                            "The maximum number of top documents to consider in each rank list (0: no limit).")
tf.app.flags.DEFINE_integer("selection_bias_cutoff", 10,
                            "The maximum number of top documents to be shown to user (which creates selection bias) in each rank list (0: no limit).")
tf.app.flags.DEFINE_integer("max_train_iteration", 10000,
                            "Limit on the iterations of training (0: no limit).")
tf.app.flags.DEFINE_integer("start_saving_iteration", 0,
                            "The minimum number of iterations before starting to test and save models. (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 50,
                            "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_boolean("test_while_train", False,
                            "Set to True to test models during the training process.")
tf.app.flags.DEFINE_boolean("test_only", False,
                            "Set to True for testing models only.")

FLAGS = tf.app.flags.FLAGS


def create_model(session, exp_settings, data_set, forward_only):
    """Create model and initialize or load parameters in session.
    
        Args:
            session: (tf.Session) The session used to run tensorflow models
            exp_settings: (dictionary) The dictionary containing the model settings.
            data_set: (Raw_data) The dataset used to build the input layer.
            forward_only: Set true to conduct prediction only, false to conduct training.
    """
    
    model = find_class(exp_settings['learning_algorithm'])(data_set, exp_settings, forward_only)
    '''
    分别是 DLA 和 DNN / DLCM ...（ranking model）
    '''

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    # cprint('ckpt: {}'.format(ckpt), 'yellow')
    '''
    model_checkpoint_path: "./tmp_model/ultra.learning_algorithm.DLA.ckpt-50"
    all_model_checkpoint_paths: "./tmp_model/ultra.learning_algorithm.DLA.ckpt-50"
    '''
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train(exp_settings):
    # Prepare data.
    print("Reading data in %s" % FLAGS.data_dir)
    train_set = read_data(FLAGS.data_dir, FLAGS.train_data_prefix, FLAGS.max_list_cutoff)
    # cprint(train_set, 'green')  # <utils.data_utils.Raw_data object at 0x7f9347482d00>
    find_class(exp_settings['train_input_feed']).preprocess_data(train_set, exp_settings['train_input_hparams'], exp_settings)
    valid_set = read_data(FLAGS.data_dir, FLAGS.valid_data_prefix, FLAGS.max_list_cutoff)
    find_class(exp_settings['train_input_feed']).preprocess_data(valid_set, exp_settings['train_input_hparams'], exp_settings)

    print("Train Rank list size %d" % train_set.rank_list_size) # 9
    print("Valid Rank list size %d" % valid_set.rank_list_size) # 9
    exp_settings['max_candidate_num'] = max(train_set.rank_list_size, valid_set.rank_list_size)
    test_set = None
    if FLAGS.test_while_train:
        test_set = read_data(FLAGS.data_dir, FLAGS.test_data_prefix, FLAGS.max_list_cutoff)
        find_class(exp_settings['train_input_feed']).preprocess_data(test_set, exp_settings['train_input_hparams'], exp_settings)
        print("Test Rank list size %d" % test_set.rank_list_size)
        exp_settings['max_candidate_num'] = max(test_set.rank_list_size, exp_settings['max_candidate_num'])
        test_set.pad(exp_settings['max_candidate_num'])

    if 'selection_bias_cutoff' not in exp_settings: # check if there is a limit on the number of items per training query.
        exp_settings['selection_bias_cutoff'] = FLAGS.selection_bias_cutoff if FLAGS.selection_bias_cutoff > 0 else exp_settings['max_candidate_num']
    
    exp_settings['selection_bias_cutoff'] = min(exp_settings['selection_bias_cutoff'], exp_settings['max_candidate_num'])
    print('Users can only see the top %d documents for each query in training.' % exp_settings['selection_bias_cutoff'])
    
    # Pad data
    train_set.pad(exp_settings['max_candidate_num'])
    valid_set.pad(exp_settings['max_candidate_num'])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # tf.get_variable_scope().reuse_variables() # zcr --> useless for the error `ValueError: Variable dnn_W_0 does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?`
        # Create model based on the input layer.
        print("Creating model...")
        model = create_model(sess, exp_settings, train_set, False)
        #model.print_info()

        # Create data feed
        train_input_feed = find_class(exp_settings['train_input_feed'])(model, FLAGS.batch_size, exp_settings['train_input_hparams'], sess)
        valid_input_feed = find_class(exp_settings['valid_input_feed'])(model, FLAGS.batch_size, exp_settings['valid_input_hparams'], sess)
        test_input_feed = None
        if FLAGS.test_while_train:
            test_input_feed = find_class(exp_settings['test_input_feed'])(model, FLAGS.batch_size, exp_settings['test_input_hparams'], sess)

        # Create tensorboard summarizations.
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, 'train_log'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, 'valid_log'))
        test_writer = None
        if FLAGS.test_while_train:
            test_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, 'test_log'))

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        best_perf = None
        while True:
            # Get a batch and make a step.
            start_time = time.time()
            input_feed, info_map = train_input_feed.get_batch(train_set, check_validation=True)
            # cprint('input_feed: {}'.format(input_feed), 'green')
            # cprint('info_map: {}'.format(info_map), 'green')
            # cprint('len(info_map[rank_list_idxs]): {}'.format(len(info_map['rank_list_idxs'])), 'green')    # 256
            # cprint('len(info_map[input_list]): {}'.format(len(info_map['input_list'])), 'green')    # 256
            # cprint('len(info_map[click_list]): {}'.format(len(info_map['click_list'])), 'green')    # 256
            # cprint('len(info_map[letor_features]): {}'.format(len(info_map['letor_features'])), 'green')    # 1479
            cprint('info_map[rank_list_idxs]: {}'.format(info_map['rank_list_idxs']), 'green')
            '''
            [12, 7, 17, 0, 0, 13, 9, 8, 10, 4, 18, 8, 10, 6, 5, 15, 14, 10, 6, 3, 16, 1, 10, 0, 18, 1, 19, 15, 3, 2, 18, 7, 6, 8, 13, 4, 11, 11, 5, 2, 10, 1, 19, 2, 14, 6, 18, 14, 9, 1, 5, 11, 19, 4, 6, 12, 15, 11, 19, 9, 15, 3, 4, 16, 6, 6, 7, 0, 10, 17, 4, 14, 8, 14, 10, 8, 13, 6, 14, 17, 4, 1, 6, 1, 7, 0, 15, 3, 14, 4, 6, 6, 17, 19, 7, 3, 7, 7, 14, 18, 0, 16, 14, 16, 10, 9, 15, 6, 0, 12, 17, 9, 4, 2, 16, 17, 10, 16, 4, 2, 12, 12, 13, 14, 4, 17, 6, 9, 1, 3, 12, 19, 17, 10, 3, 4, 15, 19, 17, 0, 5, 10, 19, 8, 7, 4, 17, 17, 0, 12, 14, 7, 9, 0, 6, 10, 12, 15, 2, 5, 19, 7, 19, 16, 6, 2, 11, 1, 17, 3, 1, 10, 9, 0, 16, 12, 17, 19, 12, 1, 1, 18, 3, 19, 12, 13, 16, 4, 1, 2, 19, 15, 3, 12, 2, 12, 9, 18, 5, 13, 13, 2, 4, 10, 6, 4, 4, 9, 0, 0, 6, 15, 1, 11, 1, 15, 19, 8, 19, 3, 9, 1, 19, 4, 14, 18, 13, 0, 8, 11, 6, 17, 1, 18, 16, 14, 14, 14, 4, 13, 13, 4, 3, 8, 6, 14, 3, 19, 2, 2, 19, 0, 2, 9, 18, 0]
            '''
            cprint('info_map[input_list]: {}'.format(info_map['input_list']), 'green')
            '''
            是一个list的list，内部list中的item个数为9
            [[0, 1, 2, 3, 4, 5, 6, 1487, 1487], [7, 8, 9, 10, 11, 12, 13, 1487, 1487], [14, 15, 16, 17, 18, 19, 20, 1487, 1487], [21, 22, 23, 24, 1487, 1487, 1487, 1487, 1487], [25, 26, 27, 28, 1487, 1487, 1487, 1487, 1487], [29, 30, 31, 32, 33, 34, 1487, 1487, 1487], [35, 36, 37, 1487, 1487, 1487, 1487, 1487, 1487], [38, 39, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [40, 41, 42, 43, 44, 45, 46, 47, 48], [49, 50, 51, 52, 53, 1487, 1487, 1487, 1487], [54, 55, 56, 1487, 1487, 1487, 1487, 1487, 1487], [57, 58, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [59, 60, 61, 62, 63, 64, 65, 66, 67], [68, 69, 70, 71, 72, 73, 74, 1487, 1487], [75, 76, 77, 78, 1487, 1487, 1487, 1487, 1487], [79, 80, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [81, 82, 83, 84, 85, 86, 87, 88, 89], [90, 91, 92, 93, 94, 95, 96, 97, 98], [99, 100, 101, 102, 103, 104, 105, 1487, 1487], [106, 107, 108, 1487, 1487, 1487, 1487, 1487, 1487], [109, 110, 111, 112, 113, 114, 115, 116, 1487], [117, 118, 119, 120, 1487, 1487, 1487, 1487, 1487], [121, 122, 123, 124, 125, 126, 127, 128, 129], [130, 131, 132, 133, 1487, 1487, 1487, 1487, 1487], [134, 135, 136, 1487, 1487, 1487, 1487, 1487, 1487], [137, 138, 139, 140, 1487, 1487, 1487, 1487, 1487], [141, 142, 143, 144, 145, 146, 147, 148, 149], [150, 151, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [152, 153, 154, 1487, 1487, 1487, 1487, 1487, 1487], [155, 156, 157, 158, 159, 160, 161, 162, 163], [164, 165, 166, 1487, 1487, 1487, 1487, 1487, 1487], [167, 168, 169, 170, 171, 172, 173, 1487, 1487], [174, 175, 176, 177, 178, 179, 180, 1487, 1487], [181, 182, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [183, 184, 185, 186, 187, 188, 1487, 1487, 1487], [189, 190, 191, 192, 193, 1487, 1487, 1487, 1487], [194, 195, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [196, 197, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [198, 199, 200, 201, 1487, 1487, 1487, 1487, 1487], [202, 203, 204, 205, 206, 207, 208, 209, 210], [211, 212, 213, 214, 215, 216, 217, 218, 219], [220, 221, 222, 223, 1487, 1487, 1487, 1487, 1487], [224, 225, 226, 227, 228, 229, 230, 231, 232], [233, 234, 235, 236, 237, 238, 239, 240, 241], [242, 243, 244, 245, 246, 247, 248, 249, 250], [251, 252, 253, 254, 255, 256, 257, 1487, 1487], [258, 259, 260, 1487, 1487, 1487, 1487, 1487, 1487], [261, 262, 263, 264, 265, 266, 267, 268, 269], [270, 271, 272, 1487, 1487, 1487, 1487, 1487, 1487], [273, 274, 275, 276, 1487, 1487, 1487, 1487, 1487], [277, 278, 279, 280, 1487, 1487, 1487, 1487, 1487], [281, 282, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [283, 284, 285, 286, 287, 288, 289, 290, 291], [292, 293, 294, 295, 296, 1487, 1487, 1487, 1487], [297, 298, 299, 300, 301, 302, 303, 1487, 1487], [304, 305, 306, 307, 308, 309, 310, 1487, 1487], [311, 312, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [313, 314, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [315, 316, 317, 318, 319, 320, 321, 322, 323], [324, 325, 326, 1487, 1487, 1487, 1487, 1487, 1487], [327, 328, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [329, 330, 331, 1487, 1487, 1487, 1487, 1487, 1487], [332, 333, 334, 335, 336, 1487, 1487, 1487, 1487], [337, 338, 339, 340, 341, 342, 343, 344, 1487], [345, 346, 347, 348, 349, 350, 351, 1487, 1487], [352, 353, 354, 355, 356, 357, 358, 1487, 1487], [359, 360, 361, 362, 363, 364, 365, 1487, 1487], [366, 367, 368, 369, 1487, 1487, 1487, 1487, 1487], [370, 371, 372, 373, 374, 375, 376, 377, 378], [379, 380, 381, 382, 383, 384, 385, 1487, 1487], [386, 387, 388, 389, 390, 1487, 1487, 1487, 1487], [391, 392, 393, 394, 395, 396, 397, 398, 399], [400, 401, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [402, 403, 404, 405, 406, 407, 408, 409, 410], [411, 412, 413, 414, 415, 416, 417, 418, 419], [420, 421, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [422, 423, 424, 425, 426, 427, 1487, 1487, 1487], [428, 429, 430, 431, 432, 433, 434, 1487, 1487], [435, 436, 437, 438, 439, 440, 441, 442, 443], [444, 445, 446, 447, 448, 449, 450, 1487, 1487], [451, 452, 453, 454, 455, 1487, 1487, 1487, 1487], [456, 457, 458, 459, 1487, 1487, 1487, 1487, 1487], [460, 461, 462, 463, 464, 465, 466, 1487, 1487], [467, 468, 469, 470, 1487, 1487, 1487, 1487, 1487], [471, 472, 473, 474, 475, 476, 477, 1487, 1487], [478, 479, 480, 481, 1487, 1487, 1487, 1487, 1487], [482, 483, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [484, 485, 486, 1487, 1487, 1487, 1487, 1487, 1487], [487, 488, 489, 490, 491, 492, 493, 494, 495], [496, 497, 498, 499, 500, 1487, 1487, 1487, 1487], [501, 502, 503, 504, 505, 506, 507, 1487, 1487], [508, 509, 510, 511, 512, 513, 514, 1487, 1487], [515, 516, 517, 518, 519, 520, 521, 1487, 1487], [522, 523, 524, 525, 526, 527, 528, 529, 530], [531, 532, 533, 534, 535, 536, 537, 1487, 1487], [538, 539, 540, 1487, 1487, 1487, 1487, 1487, 1487], [541, 542, 543, 544, 545, 546, 547, 1487, 1487], [548, 549, 550, 551, 552, 553, 554, 1487, 1487], [555, 556, 557, 558, 559, 560, 561, 562, 563], [564, 565, 566, 1487, 1487, 1487, 1487, 1487, 1487], [567, 568, 569, 570, 1487, 1487, 1487, 1487, 1487], [571, 572, 573, 574, 575, 576, 577, 578, 1487], [579, 580, 581, 582, 583, 584, 585, 586, 587], [588, 589, 590, 591, 592, 593, 594, 595, 1487], [596, 597, 598, 599, 600, 601, 602, 603, 604], [605, 606, 607, 1487, 1487, 1487, 1487, 1487, 1487], [608, 609, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [610, 611, 612, 613, 614, 615, 616, 1487, 1487], [617, 618, 619, 620, 1487, 1487, 1487, 1487, 1487], [621, 622, 623, 624, 625, 626, 627, 1487, 1487], [628, 629, 630, 631, 632, 633, 634, 1487, 1487], [635, 636, 637, 1487, 1487, 1487, 1487, 1487, 1487], [638, 639, 640, 641, 642, 1487, 1487, 1487, 1487], [643, 644, 645, 646, 647, 648, 649, 650, 651], [652, 653, 654, 655, 656, 657, 658, 659, 1487], [660, 661, 662, 663, 664, 665, 666, 1487, 1487], [667, 668, 669, 670, 671, 672, 673, 674, 675], [676, 677, 678, 679, 680, 681, 682, 683, 1487], [684, 685, 686, 687, 688, 1487, 1487, 1487, 1487], [689, 690, 691, 692, 693, 694, 695, 696, 697], [698, 699, 700, 701, 702, 703, 704, 1487, 1487], [705, 706, 707, 708, 709, 710, 711, 1487, 1487], [712, 713, 714, 715, 716, 717, 1487, 1487, 1487], [718, 719, 720, 721, 722, 723, 724, 725, 726], [727, 728, 729, 730, 731, 1487, 1487, 1487, 1487], [732, 733, 734, 735, 736, 737, 738, 1487, 1487], [739, 740, 741, 742, 743, 744, 745, 1487, 1487], [746, 747, 748, 1487, 1487, 1487, 1487, 1487, 1487], [749, 750, 751, 752, 1487, 1487, 1487, 1487, 1487], [753, 754, 755, 1487, 1487, 1487, 1487, 1487, 1487], [756, 757, 758, 759, 760, 761, 762, 1487, 1487], [763, 764, 765, 766, 767, 768, 769, 770, 771], [772, 773, 774, 775, 776, 777, 778, 1487, 1487], [779, 780, 781, 782, 783, 784, 785, 786, 787], [788, 789, 790, 1487, 1487, 1487, 1487, 1487, 1487], [791, 792, 793, 794, 795, 1487, 1487, 1487, 1487], [796, 797, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [798, 799, 800, 801, 802, 803, 804, 805, 806], [807, 808, 809, 810, 811, 812, 813, 1487, 1487], [814, 815, 816, 817, 1487, 1487, 1487, 1487, 1487], [818, 819, 820, 821, 1487, 1487, 1487, 1487, 1487], [822, 823, 824, 825, 826, 827, 828, 829, 830], [831, 832, 833, 834, 835, 836, 837, 838, 839], [840, 841, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [842, 843, 844, 845, 846, 847, 848, 1487, 1487], [849, 850, 851, 852, 853, 1487, 1487, 1487, 1487], [854, 855, 856, 857, 858, 859, 860, 1487, 1487], [861, 862, 863, 864, 865, 866, 867, 1487, 1487], [868, 869, 870, 871, 1487, 1487, 1487, 1487, 1487], [872, 873, 874, 875, 876, 877, 878, 1487, 1487], [879, 880, 881, 882, 883, 884, 885, 886, 887], [888, 889, 890, 891, 892, 893, 894, 1487, 1487], [895, 896, 897, 1487, 1487, 1487, 1487, 1487, 1487], [898, 899, 900, 901, 1487, 1487, 1487, 1487, 1487], [902, 903, 904, 905, 906, 907, 908, 1487, 1487], [909, 910, 911, 912, 913, 914, 915, 916, 917], [918, 919, 920, 921, 922, 923, 924, 1487, 1487], [925, 926, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [927, 928, 929, 930, 931, 932, 933, 934, 935], [936, 937, 938, 939, 1487, 1487, 1487, 1487, 1487], [940, 941, 942, 943, 944, 945, 946, 947, 948], [949, 950, 951, 952, 953, 954, 955, 1487, 1487], [956, 957, 958, 959, 960, 961, 962, 963, 964], [965, 966, 967, 968, 969, 970, 971, 972, 1487], [973, 974, 975, 976, 977, 978, 979, 1487, 1487], [980, 981, 982, 983, 984, 985, 986, 987, 988], [989, 990, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [991, 992, 993, 994, 1487, 1487, 1487, 1487, 1487], [995, 996, 997, 998, 999, 1000, 1001, 1487, 1487], [1002, 1003, 1004, 1487, 1487, 1487, 1487, 1487, 1487], [1005, 1006, 1007, 1008, 1487, 1487, 1487, 1487, 1487], [1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017], [1018, 1019, 1020, 1487, 1487, 1487, 1487, 1487, 1487], [1021, 1022, 1023, 1024, 1487, 1487, 1487, 1487, 1487], [1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1487], [1033, 1034, 1035, 1036, 1037, 1038, 1039, 1487, 1487], [1040, 1041, 1042, 1043, 1044, 1045, 1046, 1487, 1487], [1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055], [1056, 1057, 1058, 1059, 1060, 1061, 1062, 1487, 1487], [1063, 1064, 1065, 1066, 1487, 1487, 1487, 1487, 1487], [1067, 1068, 1069, 1070, 1487, 1487, 1487, 1487, 1487], [1071, 1072, 1073, 1487, 1487, 1487, 1487, 1487, 1487], [1074, 1075, 1076, 1487, 1487, 1487, 1487, 1487, 1487], [1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085], [1086, 1087, 1088, 1089, 1090, 1091, 1092, 1487, 1487], [1093, 1094, 1095, 1096, 1097, 1098, 1487, 1487, 1487], [1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1487], [1107, 1108, 1109, 1110, 1111, 1487, 1487, 1487, 1487], [1112, 1113, 1114, 1115, 1487, 1487, 1487, 1487, 1487], [1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124], [1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133], [1134, 1135, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [1136, 1137, 1138, 1487, 1487, 1487, 1487, 1487, 1487], [1139, 1140, 1141, 1142, 1143, 1144, 1145, 1487, 1487], [1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154], [1155, 1156, 1157, 1158, 1159, 1160, 1161, 1487, 1487], [1162, 1163, 1164, 1487, 1487, 1487, 1487, 1487, 1487], [1165, 1166, 1167, 1487, 1487, 1487, 1487, 1487, 1487], [1168, 1169, 1170, 1171, 1487, 1487, 1487, 1487, 1487], [1172, 1173, 1174, 1175, 1176, 1177, 1487, 1487, 1487], [1178, 1179, 1180, 1181, 1182, 1183, 1487, 1487, 1487], [1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192], [1193, 1194, 1195, 1196, 1197, 1487, 1487, 1487, 1487], [1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206], [1207, 1208, 1209, 1210, 1211, 1212, 1213, 1487, 1487], [1214, 1215, 1216, 1217, 1218, 1487, 1487, 1487, 1487], [1219, 1220, 1221, 1222, 1223, 1487, 1487, 1487, 1487], [1224, 1225, 1226, 1487, 1487, 1487, 1487, 1487, 1487], [1227, 1228, 1229, 1230, 1487, 1487, 1487, 1487, 1487], [1231, 1232, 1233, 1234, 1487, 1487, 1487, 1487, 1487], [1235, 1236, 1237, 1238, 1239, 1240, 1241, 1487, 1487], [1242, 1243, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [1244, 1245, 1246, 1247, 1487, 1487, 1487, 1487, 1487], [1248, 1249, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [1250, 1251, 1252, 1253, 1487, 1487, 1487, 1487, 1487], [1254, 1255, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264], [1265, 1266, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275], [1276, 1277, 1278, 1487, 1487, 1487, 1487, 1487, 1487], [1279, 1280, 1281, 1487, 1487, 1487, 1487, 1487, 1487], [1282, 1283, 1284, 1285, 1487, 1487, 1487, 1487, 1487], [1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294], [1295, 1296, 1297, 1298, 1299, 1487, 1487, 1487, 1487], [1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308], [1309, 1310, 1311, 1487, 1487, 1487, 1487, 1487, 1487], [1312, 1313, 1314, 1315, 1316, 1317, 1487, 1487, 1487], [1318, 1319, 1320, 1321, 1487, 1487, 1487, 1487, 1487], [1322, 1323, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [1324, 1325, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [1326, 1327, 1328, 1329, 1330, 1331, 1332, 1487, 1487], [1333, 1334, 1335, 1336, 1337, 1338, 1339, 1487, 1487], [1340, 1341, 1342, 1343, 1487, 1487, 1487, 1487, 1487], [1344, 1345, 1346, 1487, 1487, 1487, 1487, 1487, 1487], [1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1487], [1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363], [1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372], [1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381], [1382, 1383, 1384, 1385, 1386, 1487, 1487, 1487, 1487], [1387, 1388, 1389, 1390, 1391, 1392, 1487, 1487, 1487], [1393, 1394, 1395, 1396, 1397, 1398, 1487, 1487, 1487], [1399, 1400, 1401, 1402, 1403, 1487, 1487, 1487, 1487], [1404, 1405, 1406, 1487, 1487, 1487, 1487, 1487, 1487], [1407, 1408, 1487, 1487, 1487, 1487, 1487, 1487, 1487], [1409, 1410, 1411, 1412, 1413, 1414, 1415, 1487, 1487], [1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424], [1425, 1426, 1427, 1487, 1487, 1487, 1487, 1487, 1487], [1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436], [1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445], [1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454], [1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463], [1464, 1465, 1466, 1467, 1487, 1487, 1487, 1487, 1487], [1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476], [1477, 1478, 1479, 1487, 1487, 1487, 1487, 1487, 1487], [1480, 1481, 1482, 1487, 1487, 1487, 1487, 1487, 1487], [1483, 1484, 1485, 1486, 1487, 1487, 1487, 1487, 1487]]
            '''
            cprint('info_map[click_list]: {}'.format(info_map['click_list']), 'green')
            '''
            是一个list的list，内部list中的item个数为9， 数值为1的表示点击，为0的表示不点击。
            [[0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]]
            '''
            exit()
            step_loss, _, summary = model.step(sess, input_feed, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            train_writer.add_summary(summary, model.global_step.eval())

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                cprint("global step {} learning rate {:.4f} step-time {:.2f} loss {:.4f}".format(model.global_step.eval(), model.learning_rate.eval(), step_time, loss), 'green')
                previous_losses.append(loss)
                # Validate model
                def validate_model(data_set, data_input_feed):
                    it = 0
                    count_batch = 0.0
                    summary_list = []
                    batch_size_list = []
                    while it < len(data_set.initial_list):
                        input_feed, info_map = data_input_feed.get_next_batch(it, data_set, check_validation=False)
                        _, _, summary = model.step(sess, input_feed, True)
                        summary_list.append(summary)
                        batch_size_list.append(len(info_map['input_list']))
                        it += batch_size_list[-1]
                        count_batch += 1.0
                    return merge_TFSummary(summary_list, batch_size_list)
                    
                valid_summary = validate_model(valid_set, valid_input_feed)
                valid_writer.add_summary(valid_summary, model.global_step.eval())
                cprint("[Valid]: %s" % (' '.join(['%s: %.3f' % (x.tag, x.simple_value) for x in valid_summary.value])), 'green')

                if FLAGS.test_while_train:
                    test_summary = validate_model(test_set, test_input_feed)
                    test_writer.add_summary(test_summary, model.global_step.eval())
                    cprint("[Test]: %s" % (' '.join(['%s:%.3f' % (x.tag, x.simple_value) for x in test_summary.value])), 'green')

                # Save checkpoint if the objective metric on the validation set is better
                if "objective_metric" in exp_settings:
                    for x in valid_summary.value:
                        if x.tag == exp_settings["objective_metric"]:
                            if current_step >= FLAGS.start_saving_iteration:
                                if best_perf == None or best_perf < x.simple_value:
                                    checkpoint_path = os.path.join(FLAGS.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
                                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                                    best_perf = x.simple_value
                                    print('Save model, valid %s:%.3f' % (x.tag, best_perf))
                                    break
                # Save checkpoint if there is no objective metic
                if best_perf == None and current_step > FLAGS.start_saving_iteration:
                    checkpoint_path = os.path.join(FLAGS.model_dir, "%s.ckpt" % exp_settings['learning_algorithm'])
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                if loss == float('inf'):
                    break

                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

                if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                    break



def test(exp_settings):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load test data.
        print("Reading data in %s" % FLAGS.data_dir)
        test_set = read_data(FLAGS.data_dir, FLAGS.test_data_prefix, FLAGS.max_list_cutoff)
        find_class(exp_settings['train_input_feed']).preprocess_data(test_set, exp_settings['train_input_hparams'], exp_settings)
        exp_settings['max_candidate_num'] = test_set.rank_list_size

        test_set.pad(exp_settings['max_candidate_num'])
        
        # Create model and load parameters.
        model = create_model(sess, exp_settings, test_set, True)

        # Create input feed
        test_input_feed = find_class(exp_settings['test_input_feed'])(model, FLAGS.batch_size, exp_settings['test_input_hparams'], sess)

        test_writer = tf.summary.FileWriter(FLAGS.model_dir + '/test_log')

        rerank_scores = []
        summary_list = []
        # Start testing.
        
        it = 0
        count_batch = 0.0
        batch_size_list = []
        while it < len(test_set.initial_list):
            input_feed, info_map = test_input_feed.get_next_batch(it, test_set, check_validation=False)
            _, output_logits, summary = model.step(sess, input_feed, True)
            summary_list.append(summary)
            batch_size_list.append(len(info_map['input_list']))
            for x in range(batch_size_list[-1]):
                rerank_scores.append(output_logits[x])
            it += batch_size_list[-1]
            count_batch += 1.0
            print("Testing {:.0%} finished".format(float(it)/len(test_set.initial_list)), end="\r", flush=True)
            
        print("\n[Done]")
        test_summary = merge_TFSummary(summary_list, batch_size_list)
        test_writer.add_summary(test_summary, it)
        cprint("[Eval]: %s" % (' '.join(['%s: %.3f' % (x.tag, x.simple_value) for x in test_summary.value])), 'green')

        #get rerank indexes with new scores
        rerank_lists = []
        for i in range(len(rerank_scores)):
            scores = rerank_scores[i]
            rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))

        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        output_ranklist(test_set, rerank_scores, FLAGS.output_dir, FLAGS.test_data_prefix)


    return



def main(_):
    exp_settings = json.load(open(FLAGS.setting_file))
    # print(type(exp_settings))   # <class 'dict'>
    print_dict(exp_settings) # zcr
    '''
    '''
    # exit()
    if FLAGS.test_only:
        cprint('test only mode!', 'green')
        test(exp_settings)
    else:
        tf.get_variable_scope().reuse_variables()   # zcr
        cprint('train mode!', 'green')
        train(exp_settings)


if __name__ == '__main__':
    # tf.reset_default_graph()    # zcr
    # tf.get_variable_scope().reuse_variables() # zcr
    # warnings.filterwarnings('ignore')   # zcr --> useless
    tf.app.run()


