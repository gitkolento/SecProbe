import os
import pandas as pd    
import streamlit as st
import json

st.title("模型配置")

# Model Registration Section
models_path = "models"
models_file = os.path.join(models_path, "models.json")
# Load the model list from the JSON file
if os.path.exists(models_file):
    with open(models_file, "r") as f:
        models_list = json.load(f)
else:
    models_list = []

# Display all registered models
st.subheader("查看注册模型")
with st.expander("模型信息"):
    if models_list:
        st.json(models_list)
    else:
        st.write("当前没有已注册的模型。")

st.subheader("模型注册")
model_name = st.text_input("模型名称")
model_url = st.text_input("模型URL")
model_key = st.text_input("模型密钥")


register_model = st.button("注册模型")
if model_name and model_url and model_key and register_model:
    if any(model['name'] == model_name for model in models_list):
        st.error(f"模型 {model_name} 已存在！")
    else:
        new_model = {"name": model_name, "url": model_url, "api_key": model_key}
        models_list.append(new_model)
        
        # Save the updated model list to the JSON file
        with open(models_file, "w") as f:
            json.dump(models_list, f, indent=4)
        
        st.session_state.success_add_message = f"模型 {model_name} 注册成功！"
        st.experimental_rerun()
        # st.success(f"模型 {model_name} 注册成功！")

if "success_add_message" in st.session_state:
    st.success(st.session_state.success_add_message)
    del st.session_state.success_add_message

# Delete a model
st.subheader("模型删除")
model_names = [model['name'] for model in models_list]
model_to_delete = st.selectbox("选择需要删除的模型", model_names)

if model_to_delete and st.button("删除模型"):
    try:
        models_list = [model for model in models_list if model['name'] != model_to_delete]
        
        # Save the updated model list to the JSON file
        with open(models_file, "w") as f:
            json.dump(models_list, f, indent=4)
            
        # st.success(f"模型 {model_to_delete} 删除成功！")
        st.session_state.success_del_message = f"模型 {model_to_delete} 删除成功！"
        st.experimental_rerun()
    except Exception as e:
        st.error(f"删除模型时出错: {e}")

if "success_del_message" in st.session_state:
    st.success(st.session_state.success_del_message)
    del st.session_state.success_del_message