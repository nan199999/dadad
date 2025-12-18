# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from nptdms import TdmsFile
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# --- 网页配置 ---
st.set_page_config(page_title="NanoIndentation Cloud", page_icon="🔬", layout="wide")

# --- CSS样式优化 ---
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50;}
    .stButton>button {width: 100%; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("🔬 纳米压痕数据在线分析 (TDMS版)")
st.markdown("上传 LabVIEW 生成的 `.tdms` 文件，即可在网页端自动计算硬度与模量。")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 分析参数")
    area_coeff = st.number_input("压头面积系数 C0 (Berkovich=24.5)", value=24.5)
    epsilon = st.number_input("几何常数 ε (默认0.75)", value=0.75)
    fit_top = st.slider("卸载拟合范围 (Top %)", 10, 100, 50, 5) / 100.0
    st.info("说明：此工具使用 Oliver-Pharr 方法进行计算。")

# --- 核心函数 ---
@st.cache_data
def parse_tdms(file):
    try:
        tdms_file = TdmsFile.read(file)
        data = {}
        for group in tdms_file.groups():
            for channel in group.channels():
                name = f"{channel.name}"
                data[name] = channel[:]
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
        return df
    except Exception as e:
        return None

def power_law(h, B, hf, m):
    return B * np.power((h - hf), m)

# --- 主界面 ---
uploaded_file = st.file_uploader("📂 请拖入或选择 .tdms 文件", type=["tdms"])

if uploaded_file:
    df = parse_tdms(uploaded_file)
    
    if df is not None:
        st.success(f"✅ 文件加载成功！包含 {len(df.columns)} 个数据通道")
        
        c1, c2 = st.columns(2)
        cols = df.columns.tolist()
        
        def get_idx(options, keys):
            for i, opt in enumerate(options):
                if any(k in opt.lower() for k in keys): return i
            return 0
            
        load_idx = get_idx(cols, ['load', 'force', 'mn', 'p'])
        disp_idx = get_idx(cols, ['disp', 'depth', 'h', 'nm'])
        
        with c1:
            col_load = st.selectbox("选择载荷 (Load, mN)", cols, index=load_idx)
        with c2:
            col_disp = st.selectbox("选择位移 (Depth, nm)", cols, index=disp_idx)
            
        P = df[col_load].dropna().values
        h = df[col_disp].dropna().values
        
        P = P - P[0]
        h = h - h[0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h, y=P, mode='lines', name='实验数据', line=dict(color='#1f77b4')))
        fig.update_layout(title="P-h 曲线", xaxis_title="位移 (nm)", yaxis_title="载荷 (mN)", hovermode="x")
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🚀 点击开始 Oliver-Pharr 分析", type="primary"):
            try:
                imax = np.argmax(P)
                Pmax, hmax = P[imax], h[imax]
                
                unload_P = P[imax:]
                unload_h = h[imax:]
                
                limit = Pmax * (1 - fit_top)
                mask = unload_P > limit
                P_fit = unload_P[mask]
                h_fit = unload_h[mask]
                
                p0 = [Pmax/hmax**2, hmax/2, 2.0]
                bounds = ([0, -np.inf, 1.0], [np.inf, hmax, 10.0])
                popt, _ = curve_fit(power_law, h_fit, P_fit, p0=p0, bounds=bounds, maxfev=5000)
                B, hf, m = popt
                
                S = B * m * (hmax - hf)**(m-1)
                hc = hmax - epsilon * (Pmax / S)
                Ac = area_coeff * hc**2
                H = (Pmax / Ac) * 1000 
                Er = (np.sqrt(np.pi)/2) * (S / np.sqrt(Ac)) * 1000
                
                st.markdown("### 📊 分析结果")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("硬度 (H)", f"{H:.3f} GPa")
                k2.metric("折算模量 (Er)", f"{Er:.3f} GPa")
                k3.metric("最大载荷", f"{Pmax:.2f} mN")
                k4.metric("接触深度", f"{hc:.2f} nm")
                
                x_sim = np.linspace(min(h_fit), max(h_fit), 50)
                y_sim = power_law(x_sim, *popt)
                fig.add_trace(go.Scatter(x=x_sim, y=y_sim, mode='lines', name='拟合曲线', line=dict(color='red', dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"计算出错: {str(e)}")
                
    else:
        st.error("无法解析该文件，请确认是有效的 TDMS 文件。")
