#-*-coding:utf-8-*-

import re
import os

def csv_to_libsvm(feature_path, libsvm_path):
    """
    将csv格式的特征文件保存为libsvm格式
    
    args:
    feature_path: csv格式的特征文件路径，其中col=2为label type: str
    libsvm_path: libsvm路径
    """
    with open(feature_path, 'r', encoding='utf-8') as f:
        with open(libsvm_path, 'w', encoding='utf-8') as wf:
            for line in f:
                data = line.strip().split(',')
                label, feature_list = data[2], data[-1]
                feature_list = feature_list.split('\x01')
                format_feature = []
                for feature in feature_list:
                    [field_id, feature_id, feature_value] = re.split('\x02|\x03', feature)
                    feature_id = int(feature_id)
                    format_feature.append((feature_id, feature_value))
                format_feature = sorted(format_feature, key=lambda feature: feature[0])
                wf.write(label)
                for feature in format_feature:
                    wf.write(' {}:{}'.format(feature[0], feature[1]))
                wf.write('\n')

if __name__ == '__main__':
    feature_path = os.path.join('data', 'part_sample_skeleton_train.csv')
    libsvm_path = os.path.join('data', 'test_pre.txt')
    csv_to_libsvm(feature_path, libsvm_path)
    pass