# -*- coding: UTF-8 -*-
import sys
import traceback
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os
import numpy as np
import json
from pytoolkit import TDWSQLProvider
import time
from datetime import datetime, timedelta
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_id", help="", type=str, default="g_sng_weishi_weishi_rec")
    parser.add_argument("--gaia_id", help="", type=str, default="3124")
    parser.add_argument("--date", help="", type=str, default="20210327")
    parser.add_argument("--days_ago", help="", type=int, default=0)
    parser.add_argument("--sample_div", help="", type=str, default="default:5")
    parser.add_argument("--output_subfix", help="", type=str, default="rl_trajectory_sample")
    parser.add_argument("--output_partitions", help="", type=int, default=250)
    args = parser.parse_known_args()[0]
    return args


def jPath(filepath):
    jPathClass = sc._gateway.jvm.org.apache.hadoop.fs.Path
    return jPathClass(filepath)

def jFileSystem():
    jFileSystemClass = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoop_configuration = sc._jsc.hadoopConfiguration()
    return jFileSystemClass.get(hadoop_configuration)

def delete_hdfs_file(filepath):
    fs = jFileSystem()
    del_path_obj = jPath(filepath)
    if fs.exists(del_path_obj):
        fs.delete(del_path_obj)
        print("HDFS file {} deleted.".format(filepath))

def getBetweenDay(begin_date, end_date):
    date_list = []
    begin_date = datetime.strptime(begin_date, "%Y%m%d%H")
    end_date = datetime.strptime(end_date, "%Y%m%d%H")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y%m%d%H")
        date_list.append(date_str)
        begin_date += timedelta(hours=1)
    return date_list

def dumpHdfsJson(obj, output_path): # zcr：貌似没有被调用过
    obj_json = json.dumps(obj)
    json_rdd = sc.parallelize([obj_json])
    delete_hdfs_file(output_path)
    json_rdd.saveAsTextFile(output_path)
    
def loadHdfsJson(input_path):
    rdd = sc.textFile(input_path)
    
    result = list()
    for row in rdd.collect():
        if len(row.strip()) != 0:
            result.append(json.loads(row))
    return result


def feature_mapping(feature_list, slot_conf):
    result = dict()
    for k in slot_conf:
        result[k] = list()
    for pair in feature_list:
        pkid, skey = pair.split(":")[:2]
        for k, v in slot_conf.items():
            if pkid in v:
                result[k].append(pair)
    return result

'''
下面函数中传入的row中包含如下内容：
tdbank_imp_date='2021060300',
ip='9.226.243.61'
time_info='2021-06-03 00:25:25'
unionid='QEA/g+z0EAwrHahdMWi9pQ=='
feedid='6YzMo0mWT1LOlmHeu'
trace_user_infokey='And-1615436333442530-1622650644448-168'
sample_id='mlplatfomr_sample_test'
ts=1622650875000
feature_str='475614320:57:1.0:1194876806 475614320:56:1.0:370593793 475614320:57:1.0:96750074 475614320:56:1.0:2117485810 [... (省略很多这样的东西，它们之间通过空格隔开)]'
sample_type=2
op_str='6 4'
playtime=17446
completion=109.0
video_len=15936
usr_gid=48
is_auc=1622650646004
log_id='QEA/g+z0EAwrHahdMWi9pQ==_75PVoV9T11LOeEBSg_1622651125887'
ad_position='video'
sum_play_time=17446
sum_completion=109.0
page_id='10001001'
is_base=0
isgenpai=0
origin_playtime=17446
origin_completion=109.48
sum_origin_playtime=17446
sum_origin_completion=109.48
person_id='1615436333442530'
comment_label=0
ref_page_id=''
abnormal_type=0
source_id='0'
mt_score_string='like_model:0.009970995;comment_model:0.033321667;mix_regression:0.834758699;share_model:0.007108706;finish_model:0.247735396;quit:0.361684412;skip:0.315289855;click-profile:0.027213201;concern_model:0.027597053;click-comment:0.066230528'
abtype_str='0'
ext='{"ymd":0,"session_num":3,"os":"android","pos":5,"play_end_type":0,"rid":["1168","1169"]}'
genpai_dur=0
session_id=1622650166885

我们listwise要从上面这些特征中提取user、item、session、request的特征，并且构造label。

playtime=17446
completion=109.0
video_len=15936
video_len是视频的物理时长，应该是以毫秒来计的，也就是15936代表一个4分钟的视频（这个是zcr推测的）
playtime是视频的播放时长，因为可以多次播放，所以可能超过物理时长
completion=109.0就是用 playtime / video_len 得到的，也就是109.0%的完播率

person_id='1615436333442530' 和 unionid 之间的关系是什么

session特征取哪些
request特征取哪些
'''
def extract_detail(row):
    unionid = row.unionid_list # 问了启华，unionid是user_id，trace_id是request_id

    page_id = row.page_id
    if page_id != '10001001': # 只看推荐页（主页），忽略其他的
        return None
    
    feature_str = row.feature_str
    feedid = row.feedid # video id
    op_str = row.op_str
    playtime = row.playtime
    completion = row.completion
    video_len = row.video_len
    ts = row.ts
    mt_score_string = row.mt_score_string
    source_id = row.source_id   # 0：从算法出来的而非运营
    session_id = row.session_id
    trace_user_infokey = row.trace_user_infokey
    
    time_info = row.time_info
    
#     if mt_score_string == "":
#         return None
#     if source_id != '0':
#         return None
    
    feature_str = feature_str.strip()
    feature_list = feature_str.split(" ")
    feature_map = feature_mapping(feature_list, slot_conf)

    es_smfw = feature_map["es_smfw"]
    smfw_no = '0'
    for tmp in es_smfw:
        smfw_no = tmp.split(":")[1]
    # random sample
    div = sample_div.get(smfw_no, sample_div["default"])
    if div > 1 and hash(unionid) % div != 0:
        return None

    user_feature = feature_map["user"]
    item_feature = feature_map["item"]
    action_merge = feature_map["action_merge"]
    action_funnel = feature_map["action_funnel"]
    
#     mt_score = dict()
#     if mt_score_string.strip() != "":
#         for pair in mt_score_string.strip().split(";"):
#             model, score = pair.split(":")
#             mt_score[model] = score
    
    item = {
        "fid": feedid,
        "sid": source_id,
        "feature": item_feature,
        "len": video_len,
        "playtime": playtime,
        "completion": completion,
        "op_str": op_str,
        "mt_score": mt_score_string,
        "action_merge": action_merge,
        "action_funnel": action_funnel,
        "ts": ts,
#         "time": time_info
    }
    return unionid, [(page_id, es_smfw, session_id, trace_user_infokey, user_feature, item)]
    
def reduce_list(x, y):
    if x is None or y is None:
        return None
    x.extend(y)
    if (len(x)>5000):
        return None
    return x


def process_uid_list(row):
    uid, info_list = row
    trace_map = dict()
    smfw_no = list()
    for info in info_list:
        page_id, es_smfw, session_id, trace_user_infokey, user_feature, item = info
        if not smfw_no:
            smfw_no = es_smfw
        ts = item["ts"]
        if trace_user_infokey not in trace_map:
            trace = dict()
            trace["session_id"] = session_id
            trace["id"] = trace_user_infokey
            trace["ts"] = ts
            trace["user_feature"] = user_feature
            trace["item_list"] = [item, ]
            
            trace_map[trace_user_infokey] = trace
        else:
            trace = trace_map[trace_user_infokey]
            if ts < trace["ts"]:
                trace["ts"] = ts
            trace["item_list"].append(item)
            
    session_map = dict()
    for trace in trace_map.values():
        trace["item_list"].sort(key=lambda x: x["ts"])
        session_id = trace.pop("session_id")
        ts = trace["ts"]
        if session_id not in session_map:
            session = dict()
            session_map[session_id] = session
            session["id"] = session_id
            session["ts"] = ts
            session["trace_list"] = [trace, ]
        else:
            session = session_map[session_id]
            if ts < session["ts"]:
                session["ts"] = ts
            session["trace_list"].append(trace)
    
    session_list = list(session_map.values())
    session_list.sort(key=lambda x: x["ts"])
    for session in session_list:
        session["trace_list"].sort(key=lambda x: x["ts"])
        
    session_cnt = len(session_list)
    trace_cnt = len(trace_map)
    item_cnt = len(info_list)

    result = {
        "uid": uid, # --> user id
        "smfw": smfw_no,
        "session_cnt": session_cnt,
        "trace_cnt": trace_cnt,
        "item_cnt": item_cnt,
        "session_list": session_list
    }
    return json.dumps(result)



if __name__ == '__main__':
    ### config
    args = get_args()
    group_id = args.group_id
    gaia_id = args.gaia_id

    # init spark
    os.environ['GROUP_ID'] = group_id # 从左侧文件树根目录pools.html中选取
    os.environ['GAIA_ID'] = gaia_id # 从左侧文件树根目录pools.html中选取
    # spark = SparkSession.builder.getOrCreate()
    spark = SparkSession.builder.config('spark.driver.maxResultSize', '4096g')\
        .config('spark.driver.memory', '16g')\
        .config("spark.executor.instances", 1000)\
        .config('spark.executor.cores', 4)\
        .config("spark.default.parallelism", 8000)\
        .config('spark.executor.memory', '15g')\
        .config('spark.hadoop.fs.defaultFS', 'hdfs://ss-sng-dc-v2/')\
        .getOrCreate()
    sc = spark.sparkContext

    # hadoop_conf = sc._jsc.hadoopConfiguration()
    # fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    spark

    # config date
    day = args.date
    days_ago = args.days_ago
    day = datetime.strptime(day, "%Y%m%d") - timedelta(days=days_ago)
    day = day.strftime("%Y%m%d")
    outdate = datetime.strptime(day, "%Y%m%d") - timedelta(days=7)
    outdate = outdate.strftime("%Y%m%d")

    begin_date = day + "00"
    end_date = day + "23"
    output_subfix = args.output_subfix
    output_partitions = args.output_partitions
    # config smfw 
    sample_div = dict()
    for pair in args.sample_div.split("_"):
        k, v = pair.split(":")
        sample_div[k] = int(v)

    ### config path
    workspace = "hdfs://ss-sng-dc-v2/stage/interface/PCG/g_sng_weishi_weishi_rec/rl_mtl_fushion/"
    config_path = workspace + "config/"
    pkid_skey_pair_path = config_path + "pkid_skey_pair/"
    slot_conf_path = config_path + "pksk_slot_config/"
    output_path = workspace + "{}/{}/".format(output_subfix, day)
    outdated_path = workspace + "{}/{}/".format(output_subfix, outdate)

    print("workspace", workspace)
    print("config_path", config_path)
    print("pkid_skey_pair_path", pkid_skey_pair_path)
    print("slot_conf_path", slot_conf_path)
    print("output_path:", output_path)
    print("outdated_path", outdated_path)


    ### config feature
    slot_conf = loadHdfsJson(slot_conf_path)
    slot_conf = slot_conf[0]
    '''
    zcr：
    slot_conf中包含的是我们要哪些特征，包含这几项：user、item、action_merge、action_funnel、es_smfw。
    每项中都是一个或者多个如下面这样子的字典：
    '1258264061': {
        '835586839': 300,
        '835586840': 301,
        '835586841': 302,
        '835586842': 303,
        '835586843': 304,
        '835586844': 305,
        '835586845': 306,
        '835586846': 307
    },
    '1607491425': {
        '48': 308,
        '49': 309,
        '50': 310,
        '51': 311,
        '52': 312,
        '53': 313,
        '54': 314,
        '55': 315
    },
    '369466128': {
        '48': 316,
        '49': 317,
        '50': 318,
        '51': 319,
        '52': 320,
        '53': 321,
        '54': 322,
        '55': 323,
        '56': 324
    },
    可以看到，最里层的dict中的value是连续的编号。·
    '''

    ### Create rdd
    partitions = ["p_{}".format(i) for i in getBetweenDay(begin_date, end_date)]
    print("InputPartitions:", len(partitions), partitions)
    '''
    InputPartitions: 24
    ['p_2021060300', 'p_2021060301', 'p_2021060302', 'p_2021060303', 'p_2021060304', 'p_2021060305', 'p_2021060306', 'p_2021060307', 'p_2021060308', 'p_2021060309', 'p_2021060310', 'p_2021060311', 'p_2021060312', 'p_2021060313', 'p_2021060314', 'p_2021060315', 'p_2021060316', 'p_2021060317', 'p_2021060318', 'p_2021060319', 'p_2021060320', 'p_2021060321', 'p_2021060322', 'p_2021060323']
    '''
    # partitions = ["p_{}".format(i) for i in getStepDay(begin_date, 4)]


    provider = TDWSQLProvider(spark, db="wesee")
    rdd = provider.table('mlplatform_dsl_tdwtest_data_repli', partitions).rdd
    num_partition = rdd.getNumPartitions() // 10
    print("RddNumPartitions:", rdd.getNumPartitions())  # 393825 --> 20210603的数据
    # print(rdd.take(1))
    '''
    上面这一行的输出太多了，可以参见zcr mac的本地目录下的hive_rdd.txt
    是从数据表中读取数据
    '''

    # 下面这个rdd就是上面 rdd = provider.table('mlplatform_dsl_tdwtest_data_repli', partitions).rdd 读表得到的rdd
    unionid_list = rdd.map(extract_detail).filter(lambda x: x is not None)
    # print(type(unionid_list))   # <class 'pyspark.rdd.PipelinedRDD'>
    unionid_list = unionid_list.reduceByKey(reduce_list, numPartitions=num_partition).filter(lambda x: x[1] is not None)
    unionid_list.take(1)

    result_rdd = unionid_list.map(process_uid_list)
    # save
    delete_hdfs_file(output_path)
    result_rdd.coalesce(output_partitions).saveAsTextFile(output_path)
    delete_hdfs_file(outdated_path)

    