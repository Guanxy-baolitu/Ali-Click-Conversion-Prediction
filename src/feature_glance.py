#-*-coding:utf-8-*-
"""
统计特征信息
"""
import re
import pickle
import os
import matplotlib.pyplot as plt

def feature_glance(feature_paths, feature_glance_path):
    """
    从文件中读取feature，查看不同field_id中feature_id及feature_value情况，并将结果保存
    
    Args:
    feature_paths: 特征文件路径 type: list
    feature_glance_path: 结果文件路径 type: str

    features数据结构: dict 
    {field_id: {feature_id: set(feature_value)}}
    """
    features = dict()
    for feature_path in feature_paths:
        with open(feature_path, 'r', encoding='utf-8') as f:
            for line in f:
                feature_list = line.strip().split(',')[-1]
                feature_list = feature_list.split('\x01')
                for feature in feature_list:
                    [field_id, feature_id, feature_value] = re.split('\x02|\x03', feature)
                    feature_id, feature_value = int(feature_id), float(feature_value)
                    if field_id not in features.keys():
                        features[field_id] = dict()
                    if feature_id not in features[field_id].keys():
                        features[field_id][feature_id] = set()
                    features[field_id][feature_id].add(feature_value) 
    # 将features保存为文件
    with open(feature_glance_path, 'wb') as wf:
        pickle.dump(features, wf)


def feature_glance_digest(feature_glance_path, feature_glance_digest_path):
    """
    从features中提取特征摘要信息

    args:
    feature_glance_path: feature_glance输出文件路径 type: str
    feature_glance_digest_path: 摘要信息保存路径 type: str
    """
    with open(feature_glance_path, 'rb') as f:
        features = pickle.load(f)
    with open(feature_glance_digest_path, 'w', encoding='utf-8') as wf:
        for field_id in features.keys():
            min_feature_id, max_feature_id = min(features[field_id].keys()), max(features[field_id].keys())
            continuous_num, categorical_num = 0, 0
            for feature_id, values in features[field_id].items():
                if max(values) == 1.0 and min(values) == 1.0:
                    categorical_num += 1
                else:
                    continuous_num += 1
            # print('---------------------------')
            # print('field_id:', field_id)
            # print('min feature_id:', min_feature_id)
            # print('max feature_id:', max_feature_id)
            # print('span of feature id:', max_feature_id - min_feature_id + 1)
            # print('num of features:', len(features[field_id]))
            # print('num of continuous:', continuous_num)
            # print('num of categorical:', categorical_num)
            wf.write('---------------------------\n')
            wf.write('field_id: {}\n'.format(field_id))
            wf.write('min feature_id: {}\n'.format(min_feature_id))
            wf.write('max feature_id: {}\n'.format(max_feature_id))
            wf.write('span of feature id: {}\n'.format(max_feature_id - min_feature_id + 1))
            wf.write('num of features: {}\n'.format(len(features[field_id])))
            wf.write('num of continuous: {}\n'.format(continuous_num))
            wf.write('num of categorical: {}\n'.format(categorical_num))

def feature_num_hist(data_path, pos = 1):
    """
    每条记录中非0特征数量直方图

    args:
    data_path: 数据路径 type: str
    pos: 特征数量在数据文件中的位置 int
    """
    feature_num = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split(',')
            feature_num.append(int(data[pos]))
        plt.hist(feature_num, 100)
        plt.show()
        print('max feature num:', max(feature_num))
        print('min feature num:', min(feature_num))
        print('num of records:', len(feature_num))

if __name__ == '__main__':
    data_path = os.path.join('data', 'part_sample_skeleton_train.csv')
    feature_num_hist(data_path, pos = 4)
    pass