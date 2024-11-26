import os
import pandas as pd    
import streamlit as st
import json

st.set_page_config(layout="wide")
st.title("题库配置")
dataset_path = "dataset"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

tasks = {  
    "基础理论考核": "",  
    "应用场景研判": "",  
    "实战技能演练": "",  
    "内生安全评析": "",  
}
if os.path.exists('dataset/tasks_config.json'):
    with open('dataset/tasks_config.json', 'r', encoding='utf-8') as f:
        tasks = json.load(f)
        
tabs = st.tabs(tasks.keys())

for task_name, tab in zip(tasks.keys(), tabs):
    with tab:  
        st.write(f"### {task_name}")

        # 获取该任务的现有数据集文件  
        task_dataset_path = tasks[task_name]
        dataset_files = [f for f in os.listdir(dataset_path) if f.startswith(f"{task_dataset_path}") and f.endswith('.csv')]

        # 显示现有数据集
        if dataset_files:
            dataset_file = dataset_files[0]
            task_dataset_path = os.path.join(dataset_path, dataset_file)
            with st.expander(f"{dataset_file}"):
                try:
                    dataset_df = pd.read_csv(task_dataset_path)
                    st.dataframe(dataset_df)  # 显示题库  
                except Exception as e:  
                    st.error(f"读取题库 {dataset_file} 时出错: {e}")  
        else:  
            st.warning("尚未上传数据集")  

        # 上传自定义数据集
        uploaded_file = st.file_uploader(f"题库上传（csv文件）", key=task_name)  
        if uploaded_file is not None:  
            custom_dataset_path = os.path.join(dataset_path, f"{uploaded_file.name}")  

            df = pd.read_csv(uploaded_file)  
            st.write(df)  
            st.success(f"{uploaded_file.name} 上传成功！")  

            if st.button(f"更新题库", key=f"add_{task_name}"):
                with open(custom_dataset_path, "wb") as f: f.write(uploaded_file.getbuffer())
                for old_file in dataset_files:
                    os.remove(os.path.join(dataset_path, old_file))

                # 更新任务的数据集路径  
                tasks[task_name] = f"{uploaded_file.name}"  
                with open('dataset/tasks_config.json', 'w') as f:  
                    json.dump(tasks, f, ensure_ascii=False, indent=4)  
                st.experimental_rerun()