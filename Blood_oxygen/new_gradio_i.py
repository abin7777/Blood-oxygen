# import docx2txt
import numpy as np
# import math
# import matplotlib.pyplot as plt
import gradio as gr
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
from cal_0527._new_cal_others_v2 import info_from_beats,get_person_basic_info,sample_one_person,REAL_from_docx

# 定义从最后一个事件往前取的周期数
forward = 4

def main(waveform_csv, beats_csv=None, summary_csv=None, docx_path=None,nbp_sys=None, nbp_dia=None, last_event_index=None,
         gender=None,height=None,weight=None,age=None,bsa=None,output_options=None, model_select_option=None):
    try:
        # 读文件
        waveform = pd.read_csv(waveform_csv.name)

        TO_EVAL_KEY = ['MAP', 'CO', 'CI', 'SVR','SV',]
        
        # if beats_csv is not None:
        #     # 读文件
        #     beats = pd.read_csv(beats_csv.name)
        #     # 提取信息
        #     global forward
        #     beats_info = info_from_beats(beats, forward=forward)
        # else:
        #     # 无文件路径就通过传参来赋值
        #     beats_info = {'nbp_sys': nbp_sys,'nbp_dia': nbp_dia,'last_event_index_in_beats': last_event_index,}

        if nbp_sys and nbp_dia and last_event_index:
            # 数据框不为空，直接使用数据框中的数据
            beats_info = {'nbp_sys': nbp_sys,'nbp_dia': nbp_dia,'last_event_index_in_beats': last_event_index,}
        else:
            # 读文件
            beats = pd.read_csv(beats_csv.name)
            # 提取信息
            global forward
            beats_info = info_from_beats(beats, forward=forward)

        # 跟上面的同一个模式
        # if summary_csv is not None:
        #     person_data = pd.read_csv(summary_csv.name, index_col='Entry Name')
        #     basic_info = get_person_basic_info(person_data)
        # else:
        #     basic_info =[gender,height,weight,age,bsa]

        if gender and height and weight and age and bsa:
            basic_info =[gender,height,weight,age,bsa]
        else:
            person_data = pd.read_csv(summary_csv.name, index_col='Entry Name')
            basic_info = get_person_basic_info(person_data)


        # 用waveform beats 个人数据 来提取特征
        RES = sample_one_person(waveform, basic_info, beats_info, model_select_option)
        co_list=RES[-4]
        sv_list=RES[-3]
        svr_list=RES[-2]
        map_list=RES[-1]
        RES=RES[:-4]
        # 合成一个字典 健['MAP', 'CO', 'CI', 'SVR'] 和他们对应的值
        compare_dict = dict(zip(TO_EVAL_KEY, RES))
        if docx_path is not None:
            compare_dict = {k: {'fake': v, 'real': REAL_from_docx(docx_path.name, k)} for k, v in compare_dict.items()}
        else:
            compare_dict = {k: {'fake': v, 'real': None} for k, v in compare_dict.items()}

        compare_dict['co_list']=co_list
        compare_dict['sv_list']=sv_list
        compare_dict['svr_list']=svr_list
        compare_dict['map_list']=map_list
        if 'all' not in output_options:
            return {key: compare_dict[key] for key in [x for x in output_options if x!='all']}
        else:
            return compare_dict
        # return {'basic_info':basic_info,'beats_info':beats_info,'res':compare_dict}
    except Exception as e:
        return {"error": str(e)}


def clear_inputs(output_options):
    return [None]*12+[output_options]+[None]

# def clear_inputs():
#     return [None]*13

# 返回一个列表，列表中的数据分别表示beats info的字典的values部分
# 上传csv时调用的函数


def update_beats_info(nbp_sys, nbp_dia, last_event_index,beats_csv):
    
    if nbp_sys and nbp_dia and last_event_index :
        return [nbp_sys, nbp_dia, last_event_index]
    elif beats_csv is not None:
        beats = pd.read_csv(beats_csv.name)
        global forward
        beats_info = info_from_beats(beats, forward=forward)
        return [float(x) for x in list(beats_info.values())]
    else:
        return [None]*3

  
def update_summary_info(gender, height,weight, age, bsa,summary_csv):
    
    if height and weight and age and bsa:
        return [gender, height,weight, age, bsa]
    elif summary_csv is not None:
        person_data = pd.read_csv(summary_csv.name, index_col='Entry Name')
        basic_info = get_person_basic_info(person_data)
        return [float(x) for x in basic_info]
    else:
        return [None]*5
    
def handle_output_options(selected_options):
    if "all" in selected_options and len(selected_options) > 1:
        return [x for x in selected_options if x!='all']
    else:
        return selected_options

# 创建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            waveform_csv = gr.File(label="Upload Waveform CSV")
            docx_path = gr.File(label="Upload DOCX File")
            model_select_option = gr.Radio(["使用", "不使用"], label="模型", info="是否使用NBP数据", value="使用")
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear")

        with gr.Column():
            beats_csv = gr.File(label="Upload Beats CSV")
            gr.Markdown('不上传csv才会生效')
            with gr.Row():
                nbp_sys = gr.Number(label="NBP Sys (example:125)")
                nbp_dia = gr.Number(label="NBP Dia ( example:85)")
                last_event_index = gr.Number(label="Start time (example:116.61)")

        with gr.Column():
            summary_csv = gr.File(label="Upload Summary CSV")
            gr.Markdown('不上传csv才会生效')
            with gr.Row():
                height = gr.Number(label="Height (example:171)")
                weight = gr.Number(label="Weight (example:87)")
                age = gr.Number(label="Age (example:51)")
                gender = gr.Number(label="Gender (description:[{'M': 0, 'F': 1}])")
                bsa = gr.Number(label="Body Surface Area (example:2.03)")
        with gr.Column():
            output_options = gr.CheckboxGroup(choices=['MAP', 'CO', 'CI', 'SVR', 'SV','co_list','sv_list','svr_list','map_list','all'], label="Select Output Options",value=['all'])
            output = gr.JSON(label="Output")

    output_options.change(fn=handle_output_options, inputs=output_options, outputs=output_options)
    beats_csv.change(fn=update_beats_info, inputs=[nbp_sys, nbp_dia, last_event_index,beats_csv], outputs=[nbp_sys, nbp_dia, last_event_index])
    summary_csv.change(fn=update_summary_info,inputs=[gender, height,weight, age, bsa,summary_csv],outputs=[gender, height,weight, age, bsa])
    submit_button.click(
        main,
        inputs=[waveform_csv, beats_csv, summary_csv, docx_path, nbp_sys, nbp_dia, last_event_index, gender, height,
                weight, age, bsa,output_options, model_select_option],
        outputs=[output]
    )

# 清空操作，调用的函数 clear_inputs
    clear_button.click(
        clear_inputs,
        inputs=output_options,
        outputs=[waveform_csv, beats_csv, summary_csv, docx_path, nbp_sys, nbp_dia, last_event_index, gender, height,
                 weight, age, bsa,output_options,output]
    )


if __name__ == '__main__':
    # 允许其他机器访问
    demo.launch(server_name="0.0.0.0", server_port=9200)
    # 只允许本机访问
    # demo.launch(server_port=9200)















# iface = gr.Interface(
#     fn=main,
#     inputs=[
#         # -----files
#         gr.File(label="Upload Waveform CSV"),
#         gr.File(label="Upload Beats CSV )"),
#         gr.File(label="Upload Summary CSV",scale=0),
#         gr.File(label="Upload DOCX File"),
#         #------beats info
#         gr.Number(label="NBP Sys (only if no Beats CSV ,it works,example:125)"),
#         gr.Number(label="NBP Dia (only if no Beats CSV ,it works,example:85)"),
#         gr.Number(label="Last Event Index (only if no Beats CSV ,it works,example:112)"),
#         #------summary info
#         gr.Number(label="Gender (only if no Summary CSV ,it works,example:0,description:[{'M': 0, 'F': 1}])"),
#         gr.Number(label="Height (only if no Summary CSV ,it works,example:171)"),
#         gr.Number(label="Weight (only if no Summary CSV ,it works,example:87)"),
#         gr.Number(label="Age (only if no Summary CSV ,it works,example:51)"),
#         gr.Number(label="Body Surface Area (only if no Summary CSV ,it works,example:2.03)")
#     ],
#     outputs="json",
#     title="Predictor Interface",
#     description="Upload the required CSV and DOCX files to get predictions."
# )


