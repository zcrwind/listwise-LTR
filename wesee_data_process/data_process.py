import sys
import numpy as np
import tensorflow as tf
import os
import glob
import json
from datetime import datetime, timedelta
from multiprocessing import Process, Manager
from collections import Counter


def get_config(config_path):
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config

workspace = os.path.join(os.path.dirname(__file__), os.path.pardir)
workspace = os.path.realpath(workspace)
conf_dir = os.path.join(workspace, "config")
env_conf = get_config(os.path.join(conf_dir, "env_config.json"))
feature_conf = get_config(os.path.join(conf_dir, "feature_slot_config.json"))

user_feature_conf, user_feature_dim = feature_conf["user"], env_conf["num_state"]
item_feature_conf, item_feature_dim = feature_conf["item"], env_conf["num_item"]
action_merge_conf, action_merge_dim = feature_conf["action_merge"], env_conf["num_action"]
item_behavior_dim = env_conf["num_behavior"]

def feature_str2vec(feature_list, slot_conf, vec_len):
    vec = np.zeros(shape=(vec_len,), dtype=np.float)
    for pair in feature_list:
        pkid, skey, val = pair.split(":")[:3]
        if pkid not in slot_conf:
            continue
        if skey not in slot_conf[pkid]:
            continue
        slot = slot_conf[pkid][skey]

        if slot < vec_len:
            vec[slot] = float(val)
    return vec


def process_trace(trace):
    trace_action = None
    item_feature_list = list()
    behavior_list = list()

    user_feature = trace["user_feature"]
    user_feature = feature_str2vec(user_feature, user_feature_conf, user_feature_dim)

    item_list = trace["item_list"]
    for item in item_list:
        feedid = item["fid"]
        source_id = item["sid"]
        if source_id != "0":
            continue
        # action
        action_merge = item["action_merge"]
        if len(action_merge) < action_merge_dim:
            continue
        if trace_action is None:
            trace_action = feature_str2vec(action_merge, action_merge_conf, action_merge_dim)
        # item feature
        item_feature = item["feature"]
        item_feature = feature_str2vec(item_feature, item_feature_conf, item_feature_dim)
        # user behavior
        video_len = float(item["len"]) / 1000
        playtime = float(item["playtime"]) / 1000
        completion = float(item["completion"]) / 100
        if completion <= 0 and video_len<=0:
            completion = playtime / 15.0
        finish = 1 if completion>=1 else 0
        skip5s = 1 if playtime<=5 else 0
        skip3s = 1 if playtime<=3 else 0

        op_list = [0] * 14
        for op in item["op_str"].split(" "):
            op = int(op)
            if op < 14:
                op_list[op] = 1
        _, like, comment, share, play, follow, _, _, dislike, rd_comment, rd_person_page, _, rg_bgm, rd_topic = op_list
        play_info = [playtime, completion, finish, play, skip5s, skip3s]
        interact = [like, comment, share, follow, dislike, rd_comment, rd_person_page, rg_bgm, rd_topic]
        behavior = play_info + interact
        behavior = np.array(behavior, dtype=np.float)

        # effective item
        item_feature_list.append(item_feature)
        behavior_list.append(behavior)
    return user_feature, trace_action, item_feature_list, behavior_list


class SmfwStatistics(object):
    def __init__(self):
        self.session_cnt = Counter()
        self.trace_cnt = Counter()
        self.action_mean = dict()

    def merge_mean(self, mean1, n1, mean2, n2):
        a1 = n1/(n1+n2)
        a2 = n2/(n1+n2)
        return n1+n2, mean1*a1+mean2*a2

    def add(self, smfw_str, trace_action_list):
        self.session_cnt[smfw_str] += 1
        self.trace_cnt[smfw_str], self.action_mean[smfw_str] = self.merge_mean(
            np.mean(trace_action_list, axis=0),
            len(trace_action_list),
            self.action_mean.get(smfw_str, 0),
            self.trace_cnt[smfw_str]
        )

    def update(self, smfw_st):
        self.session_cnt.update(smfw_st.session_cnt)
        for smfw_str in smfw_st.trace_cnt:
            self.trace_cnt[smfw_str], self.action_mean[smfw_str] = self.merge_mean(
                smfw_st.action_mean[smfw_str], smfw_st.trace_cnt[smfw_str],
                self.action_mean.get(smfw_str, 0), self.trace_cnt[smfw_str]
            )

    def statistics(self):
        trace_cnt, action_mean = 0, 0
        for k, v in self.trace_cnt.items():
            print(action_mean)
            trace_cnt, action_mean = self.merge_mean(
                action_mean, trace_cnt,
                self.action_mean[k], v
            )

        result = {
            "smfw_cnt": dict(self.session_cnt),
            "smfw_trace_cnt": dict(self.trace_cnt),
            "smfw_action_mean": dict([(k, list(v)) for k, v in self.action_mean.items()]),
            "cnt": sum(self.session_cnt.values()),
            "trace_cnt": sum(self.trace_cnt.values()),
            "action_mean": list(action_mean)
        }
        print(result)
        return result


def preprocess(p_num, input_path, train_path):
    smfw_st = SmfwStatistics()
    if os.path.exists(train_path):
        raise Exception("output path: %s, already exist.", train_path)
    tf_writer = tf.io.TFRecordWriter(train_path)
    with open(input_path, 'r') as fp:
        for line in fp:
            line = line.strip()
            info = json.loads(line)
            uid = info["uid"]
            session_cnt = info["session_cnt"]
            trace_cnt = info["trace_cnt"]
            item_cnt = info["item_cnt"]
            session_list = info["session_list"]

            smfw, smfw_str = 0, '0'
            for tmp in info["smfw"]:
                smfw_str = tmp.split(":")[1]
                smfw = int(smfw_str)

            # if item_cnt < 20:
            #     continue
            for session in session_list:
                trace_feature_list = list()
                trace_action_list = list()
                trace_end_list = list()

                trace_item_list = list()
                trace_behavior_list = list()

                trace_end_idx = 0
                trace_list = session["trace_list"]
                for trace in trace_list:
                    user_feature, action, item_list, behavior_list = process_trace(trace)
                    effective_item_cnt = len(item_list)
                    if action is not None and effective_item_cnt > 0:
                        # trace
                        trace_feature_list.append(user_feature)
                        trace_action_list.append(action)

                        trace_end_idx += effective_item_cnt
                        trace_end_list.append(trace_end_idx)
                        # items
                        trace_item_list.extend(item_list)
                        trace_behavior_list.extend(behavior_list)

                if len(trace_end_list) > 0:
                    features = tf.train.Features(
                        feature={
                            "smfw": tf.train.Feature(int64_list=tf.train.Int64List(value=[smfw])),
                            # "p_num": tf.train.Feature(int64_list=tf.train.Int64List(value=[p_num])),
                        }
                    )
                    feature_lists = tf.train.FeatureLists(
                        feature_list={
                            "trace_feature": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=value)) for value in trace_feature_list]),
                            "trace_action": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=value)) for value in trace_action_list]),
                            "trace_end": tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) for value in trace_end_list]),
                            "item_feature": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=value)) for value in trace_item_list]),
                            "item_behavior": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=value)) for value in trace_behavior_list]),
                        }
                    )
                    seq_example = tf.train.SequenceExample(
                        context=features,
                        feature_lists=feature_lists
                    )
                    # 将序列化的example写到TFRecord中
                    tf_writer.write(seq_example.SerializeToString())
                    smfw_st.add(smfw_str, trace_action_list)
    tf_writer.close()
    return smfw_st


def parallel_process(input_dir, output_dir, process_num=10):
    if not os.path.exists(output_dir):
        print("create dir: {}".format(output_dir))
        os.mkdir(output_dir)
    filenames = glob.glob(os.path.join(input_dir, '*'))
    filenames = list(filter(lambda x: not x.endswith(".tfr"), filenames))
    filenum = len(filenames)
    chunk_size = filenum // process_num + 1

    def process_one(p_num, return_dict, chunk):
        smfw_st = SmfwStatistics()
        for input_file in chunk:
            file_name = input_file.split('/')[-1]
            output_file = os.path.join(output_dir, file_name+".tfr")
            if os.path.exists(output_file):
                print("Error: output_file({}) already exist".format(output_file), file=sys.stderr)
                continue
            block_smfw_st = preprocess(p_num, input_file, output_file)
            smfw_st.update(block_smfw_st)
        return_dict[p_num] = smfw_st

    p_list = list()
    manager = Manager()
    return_dict = manager.dict()
    p_num = 0
    for i in range(0, filenum, chunk_size):
        start, end = i, i + chunk_size
        chunk = filenames[start:end]
        p = Process(target=process_one, args=[p_num, return_dict, chunk])
        p.start()
        p_list.append(p)
        p_num += 1

    for p in p_list:
        p.join()

    smfw_st = SmfwStatistics()
    for p_result in return_dict.values():
        smfw_st.update(p_result)

    with open(os.path.join(output_dir, "statistics.json"), 'w') as fp:
        json.dump(smfw_st.statistics(), fp)


def get_dataset(*args):
    input_paths = args

    features = {
        "smfw": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        # "p_num": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }
    feature_lists = {
        "trace_feature": tf.io.FixedLenSequenceFeature(shape=[user_feature_dim], dtype=tf.float32),
        "trace_action": tf.io.FixedLenSequenceFeature(shape=[action_merge_dim], dtype=tf.float32),
        "trace_end": tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64),

        "item_feature": tf.io.FixedLenSequenceFeature(shape=[item_feature_dim], dtype=tf.float32),
        "item_behavior": tf.io.FixedLenSequenceFeature(shape=[item_behavior_dim], dtype=tf.float32),
    }

    def _parse(raw_record):
        context, seq = tf.io.parse_single_sequence_example(raw_record, context_features=features, sequence_features=feature_lists)
        return context, seq

    smfw_cnt = Counter()
    filenames = list()
    for input_path in input_paths:
        statistics_path = os.path.join(input_path, "statistics.json")
        if not os.path.exists(statistics_path):
            continue
        filenames.append(input_path)
        tmp = get_config(statistics_path)["smfw_cnt"]
        smfw_cnt.update(Counter(tmp))
    statistics_info = {
        "smfw_cnt": dict(smfw_cnt)
    }

    files = tf.data.Dataset.from_tensor_slices([glob.glob(os.path.join(x, "*.tfr")) for x in filenames])
    print("# Get Dataset filenames({})".format(filenames), file=sys.stderr)
    raw_dataset = files.interleave(tf.data.TFRecordDataset)
    dataset = raw_dataset.map(_parse)

    return dataset, env_conf, statistics_info


def get_days_dataset(dataspace, date, days):
    current_day = datetime.strptime(date, "%Y-%m-%d")
    input_paths = list()
    for i in range(days):
        day =  current_day - timedelta(days=i)
        day = day.strftime("%Y-%m-%d")
        input_paths.append(os.path.join(dataspace, day))
    print("# Try to get days dataset: {}".format(", ".join(input_paths)))
    return get_dataset(*input_paths)


def test_dateset(dataspace, date):
    # dataset, env_conf, statistics_info = get_dataset(*args)
    dataset, env_conf, statistics_info = get_days_dataset(dataspace, date, 3)
    cnt = 0
    for context, seq in dataset:
        # print(context["p_num"], context["smfw"], seq["trace_end"].shape)
        print(context["smfw"], seq["trace_end"].shape, seq["trace_feature"].shape)
        cnt += 1
        if cnt > 20 :
            break
    print(statistics_info)

if __name__ == '__main__':
    import sys
    import argparse
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataspace', type=str, default="./data/")
        parser.add_argument('--date', type=str, default="2021-04-23")
        parser.add_argument('--input_suffix', type=str, default="trajectory")
        parser.add_argument('--output_suffix', type=str, default="tf_record")
        parser.add_argument('--process_num', type=int, default=20)
        parser.add_argument('--test', type=lambda x: str(x).lower()=='true', default=False)
        args, _ = parser.parse_known_args()

        print("\nData Preprocess Args:", args)
        return args

    args = get_args()
    input_path = os.path.join(args.dataspace, args.input_suffix, args.date)
    output_path = os.path.join(args.dataspace, args.output_suffix, args.date)
    if args.test:
        test_dateset(os.path.join(args.dataspace, args.output_suffix), args.date)
    else:
        parallel_process(input_path, output_path, args.process_num)
    pass

    