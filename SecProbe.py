import streamlit as st

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; margin-top: -70px;'>SecProbe: 任务驱动式大模型安全专业能力评测平台</h1>", unsafe_allow_html=True)


st.markdown(
    '''
    本方案依据“**需求牵引、理论支撑、实践检验**”思路，面向大语言模型安全专业能力测评需求，聚焦**安全能力标准化、评估方法系统化、应用开发规范化**等目标，
    依托**靶向式网络安全文本可控生成、智能代理驱动的网络攻击模拟、基于对比学习的有害意图识别**和**模型动态反馈的自动智能评测**等核心技术，
    广泛参考总结现有大模型安全能力评估工具，设计构建了**SecProbe任务驱动式大模型安全专业能力评测体系**，该体系具备以下四大优势特点：**演进式安全能力评估任务、
    多层级网络安全分类法、评测流程自动化与智能化、题库动态生成优化机制**。在保证模型评测可行性高效性前提下，评测体系通过**理论知识问答检验、
    实战应用能力抽查、多维评分标准设置**等手段，对大模型安全专业能力进行全方位、多角度综合测评。
'''
)


col1, col2 = st.columns([1.31, 1])
col1.image("assets/框架.png", use_container_width=True)
col2.image("assets/数据集.png", use_container_width=True)
col2.image("assets/智能体.png", use_container_width=True)

