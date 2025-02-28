import docx2txt
import numpy as np
# import math
# import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
import joblib
multioutput_rf = joblib.load('/gjh/MultioutputRegressor_RandomForest.joblib')
with_out_NBP_rf = joblib.load('/gjh/xueyang/Multioutput_RF_without_NBP_save_model.joblib')
# multioutput_rf.set_params({'n_jobs': -1})
# multioutput_rf = joblib.load('/gjh/xueyang/mult_svm.joblib')
basic_cols = ['Gender', 'Height', 'Weight', 'Age', 'Body Surface Area']

def interp(x,y,EMBEDDING_DIM=80):
    # 假设你有一组数据点
    # x = raw_data['t']  # 生成一个一维数组作为x坐标
    # y = raw_data['pressure']              # 生成一些y值

    # 创建一个cubic插值函数
    f = interp1d(x, y, kind='cubic')

    # 现在你想在新的x点上评估插值函数
    x_new = np.linspace(min(x), max(x), EMBEDDING_DIM)  # 生成一个更密集的x坐标数组
    y_new = f(x_new)                 # 使用插值函数计算新的y值
    return y_new

# 换新的函数
# def get_one_wave(waveform,x_data,middle_one,middle_two):

def get_one_wave(waveform,x_data,pak_ind):
    left,right=x_data[pak_ind][0],x_data[pak_ind][1]


    hr=60/(right-left)
    left=waveform.index[waveform['Time [s]']==left]
    right=waveform.index[waveform['Time [s]']==right]
    #转int
    left=left.to_numpy()[0]
    right = right.to_numpy()[0]
    #取出这个区间的数据
    x=waveform['Time [s]'].to_numpy()[left+1:right+1]
    y=waveform['Pressure [mmHg]'][left+1:right+1].to_numpy()
    return x,y,hr

# 从beats文件中找到nbp sys和nbp dia的数据，还有event最后一个非空的索引，返回的类型是一个字典
def info_from_beats(beats, forward=0):
    # 最后一个Event的
    t = beats['NBP Sys [mmHg]']

    # NBP Sys 列中最后一个非空的位置
    fvi = list(t[~t.isna()].index)[-1]
    nbp_sys = beats['NBP Sys [mmHg]'][fvi]
    nbp_dia = beats['NBP Dia [mmHg]'][fvi]

    NOTNA = beats.index[beats['Event'].notna()]  # Index([0, 1, 2, 40, 56, 72], dtype='int64') in beats.csv
    last_optimize_quality_index = NOTNA[-1] - forward  #
    # MAP_REAL = beats['Beat Mean [mmHg]'][last_optimize_quality_index + 1:].to_numpy()

    # 根据beats的索引找到wave对应的索引
    a = beats['Time [s]'][last_optimize_quality_index:-1].to_numpy()


    return {'nbp_sys': nbp_sys,
            'nbp_dia': nbp_dia,
            'last_event_index_in_beats': a[0],  # 最后一个有效Event 对应的beats索引，非必要
            # 'MAP_REAL': MAP_REAL,
            }

# 根据参数key从doc文档中找出真实值
def REAL_from_docx(docx_path,key='CI'):
    doc_content = docx2txt.process(docx_path)
    return float(doc_content[doc_content.find(key):].split('\n')[2])

def get_person_basic_info(person_data):
    basic_info=[]
    for col in basic_cols:
        to_append=person_data.loc[col].values[0]
        if col=='Gender':
            to_append={'M': 0, 'F': 1}[str(to_append[0])]
        else:
            item = str(to_append).split(' ')[0]
            item = item.split('\'')[-1] if not item.isdigit() else item
            to_append=float(item)
        basic_info.append(to_append)
    return basic_info

# 这个函数的返回值应该保证最后一个是CO的预测值的list，其余的值可以修改位置，但是需要与另一个文件的TO_EVAL_KEY一一对应
def sample_one_person(waveform,basic_info,beats_info,model_select_option):
    # ------------------------------获取一个人的基本属性

    basic_info.append(beats_info['nbp_sys'])
    basic_info.append(beats_info['nbp_dia'])
    # pre_embedding_sum=0
    # find peaks 的参数
    prominence = 6
    width = 30
    rel_height = 1.0
    distance = 40
    # ------------------从Event前k个单位开始，记录索引

    # MAP_REAL=beats_info['MAP_REAL']
    ind = waveform.index[waveform['Time [s]'] >= beats_info['last_event_index_in_beats']].to_numpy()[0]
    
    MAP_cal_list,CO_cal_list = [],[]
    SV_cal_list,SVR_cal_list = [],[]
    # CO_cal_list = []

    # --------------从Event前3个单位开始，获取波形，包括峰谷集合
    x_data = waveform['Time [s]'].to_numpy()[ind:]
    y_data = waveform['Pressure [mmHg]'].to_numpy()[ind:]
    peaks, _ = find_peaks(y_data, prominence=prominence, width=width, rel_height=rel_height, distance=distance)  # prominence参数可以根据您的数据调整
    # valleys, _ = find_peaks(-y_data, prominence=prominence)  # 负号将y_data反转以检测波谷
    # peaks_and_valleys = np.sort(np.concatenate((peaks, valleys)))

    pre_period = 2

    y_data_pre = waveform['Pressure [mmHg]'].to_numpy()[:ind]
    peaks_pre, _ = find_peaks(y_data_pre,prominence=prominence, width=width, rel_height=rel_height, distance=distance)
    peaks_pre = peaks_pre[-(pre_period + 1):]
    avg_num_period = 0.0
    for i in range(len(peaks_pre)-1):
        avg_num_period += peaks_pre[i+1] - peaks_pre[i]
    avg_num_period = avg_num_period / pre_period

    # ------------------------------遍历每一个波
    feature_matrx = np.empty((0,88))
    flag_i = 0
    for i,k in enumerate(range(0,len(peaks)-1)):
        pak_ind = peaks[k:k + 2]
        # if i>=len(MAP_REAL) or len(pak_ind)!=4:
        #     break
        # if math.isnan(MAP_REAL[i]):
        #     continue
        if len(pak_ind)!=2:
            break
        # if 1==0:
        #     continue
        else:
            #-----------------------------------取出这个区间的数据
            # middle_one = peaks_and_valleys[k:k + 4]
            # middle_two = peaks_and_valleys[k + 2:k + 6]
            # if len(middle_one)!=4 or len(middle_two)!=4:
            #     break
            x,y,hr=get_one_wave(waveform,x_data,pak_ind)
            if len(x) < 0.8*avg_num_period or len(x) > 1.2*avg_num_period:
                if i == 0:
                    flag_i = 1
                continue
            embedding_y=interp(x,y,80)#前rank个需要被平均
            # rank=i+1
            if i == 0 or flag_i == 1:
                flag_i = 0
                embedding_y_weight_sum = embedding_y
            else:
                embedding_y_weight_sum = embedding_y_weight_sum * 0.5 + embedding_y * 0.5
            # pre_embedding_sum+=embedding_y_weight_sum
            predict_x = basic_info + (list(np.concatenate([[hr],embedding_y_weight_sum])))
            feature_matrx = np.vstack((feature_matrx, predict_x))


    # results = multioutput_rf.predict(np.array(predict_x).reshape(1,-1))
    
    if model_select_option == "使用":
        results = multioutput_rf.predict(feature_matrx)
    else:
        feature_matrx = feature_matrx[:,[i for i in range(feature_matrx.shape[1]) if i not in [5, 6]]] # 过滤掉NBP的特征数据
        results = with_out_NBP_rf.predict(feature_matrx)
    CO_cal_list.append(results[:,0])
    # area = np.trapz(y=y, x=x)#梯形法算面积https://blog.csdn.net/weixin_44338705/article/details/89203791
    # MAP_cal=area/(60/hr)#hr=60/(right-left)
    # SV_cal_list.append(results[0,1])
    # SVR_cal_list.append(results[0,2])
    # MAP_cal_list.append(results[0,3])
    # MAP_cal_list.append(np.round(MAP_cal))
    MAP_cal_list.append(results[:,3])
    SVR_cal_list.append(results[:,2])
    SV_cal_list.append(results[:,1])
    MAP_FAKE=np.round(np.mean(results[:,3]),decimals=0)
    CO_FAKE=np.round(np.mean(CO_cal_list),decimals=1)
    CI_FAKE=CO_FAKE/basic_info[-3]#最后一个是bsa
    SVR_FAKE=np.mean(results[:,2])
    SV_FAKE=np.round(np.mean(results[:,1]),decimals=0)

    return MAP_FAKE,CO_FAKE,np.round(CI_FAKE,decimals=1),np.round(SVR_FAKE,decimals=0),SV_FAKE,CO_cal_list,SV_cal_list,SVR_cal_list,MAP_cal_list




# 以下代码用来做测试的
if __name__ == '__main__':

    ##-------------------------------------------------------------------------------单个样本的测试，允许手动指定beats相关信息
    import time
    start_time = time.time()
    
    waveform = pd.read_csv('./after_process/after_process/a_10/RAW/10-waveform.csv')
    beats = pd.read_csv('./after_process/after_process/a_10/RAW/10-beats.csv')
    person_data = pd.read_csv('./after_process/after_process/a_10/RAW/10-summary.csv', index_col='Entry Name')
    docx_path = './after_process/after_process/a_10/10.docx'

    TO_EVAL_KEY=['MAP','CO','CI','SVR','SV']

    #beats_info 可以自己指定如下字典，用函数从文档读取也可以
    # {'nbp_sys': nbp_sys,
    #  'nbp_dia': nbp_dia,
    #  'last_event_index_in_beats': last_optimize_quality_index,  # 最后一个有效Event 对应的beats索引，非必要
    #  'MAP_REAL': MAP_REAL,
    #  }
    beats_info=info_from_beats(beats,forward=2)#forward 是最后一个 event向前多少个单位
    basic_info=get_person_basic_info(person_data) #也可以指定具体的数值，['Gender', 'Height', 'Weight', 'Age', 'Body Surface Area']=[0, 171.0, 87.0, 51.0, 2.03]
    RES=sample_one_person(waveform,basic_info,beats_info)

    compare_dict=dict(zip(TO_EVAL_KEY,RES))

    compare_dict={k:{'fake':v,'real':REAL_from_docx(docx_path,k)} for k,v in compare_dict.items()}

    print(compare_dict)
    end_time = time.time()
    print(f"测试一个受试者数据的时间: {end_time-start_time}秒")

    ##------------------------------------------------------------------------------------------------------计算指标用的
    #计算a数据集
    # import os
    # if 1==0:
    #     target_path='./datas/a'
    #     for filename in os.listdir(target_path):
    #         inner_path=os.path.join(target_path,filename)
    #         docx_path=os.path.join(inner_path,'{}.docx'.format(filename))
    #         inner_inner_path=os.path.join(inner_path,'RAW')
    #         waveform = pd.read_csv(os.path.join(inner_inner_path,'{}-waveform.csv'.format(filename)))
    #         beats = pd.read_csv(os.path.join(inner_inner_path,'{}-beats.csv'.format(filename)))
    #         person_data=pd.read_csv(os.path.join(inner_inner_path, '{}-summary.csv'.format(filename)), index_col='Entry Name')

    #         beats_info = info_from_beats(beats, forward=0)  # forward 是最后一个 event向前多少个单位
    #         basic_info = get_person_basic_info(person_data)
    #         RES = sample_one_person(waveform, basic_info, beats_info)
    #         if math.isnan(RES[0]):
    #             continue
    #         compare_dict = dict(zip(TO_EVAL_KEY, RES))
    #         compare_dict = {k: {'fake': v, 'real': REAL_from_docx(docx_path, k)} for k, v in compare_dict.items()}
    #         print(compare_dict)

