import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')
from matplotlib_utils import render_matplotlib_figure, create_zscore_analysis_plot, create_iqr_analysis_plot, create_anomaly_plot

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="FAB KPI OOB 監控 Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_basic_css():
    """應用基本CSS樣式"""
    basic_css = """
    <style>
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
    .js-plotly-plot {
        background-color: #ffffff !important;
    }
    </style>
    """
    st.markdown(basic_css, unsafe_allow_html=True)


def prepare_plotly_data(x_data, y_data):
    """準備 Plotly 圖表數據，確保格式正確 - 轉換為 Python list"""
    x_clean = to_plotly_list(x_data)
    y_clean = to_plotly_list(y_data)
    
    # 確保數據長度一致
    min_len = min(len(x_clean), len(y_clean))
    x_clean = x_clean[:min_len]
    y_clean = y_clean[:min_len]
    
    return x_clean, y_clean

def ensure_data_format(df):
    """確保數據格式正確"""
    if df is None:
        return None
    
    # 確保必要欄位存在
    required_columns = ['FAB', 'VALUE', 'KPI', 'REPORT_TIME']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"缺少必要欄位: {required_columns}")
    
    # 複製數據避免修改原始數據
    df = df.copy()
    
    # 確保數據類型正確
    df['REPORT_TIME'] = pd.to_datetime(df['REPORT_TIME'], errors='coerce')
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    df['FAB'] = df['FAB'].astype(str)
    df['KPI'] = df['KPI'].astype(str)
    
    # 移除無效數值
    df = df.dropna(subset=['VALUE', 'REPORT_TIME'])
    
    # 確保索引是連續的
    df = df.sort_values(['FAB', 'KPI', 'REPORT_TIME']).reset_index(drop=True)
    
    return df

def main():
    # 應用基本CSS
    apply_basic_css()
    
    st.title("🏭 FAB KPI 時序資料異常監控 Dashboard")
    
    # Initialize session state for data
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'selected_fab' not in st.session_state:
        st.session_state.selected_fab = None
    if 'fab_data' not in st.session_state:
        st.session_state.fab_data = None
    if 'available_kpis' not in st.session_state:
        st.session_state.available_kpis = []
    if 'selected_kpi' not in st.session_state:
        st.session_state.selected_kpi = None
    
    # Sidebar navigation
    st.sidebar.title("🔍 分析方法")
    analysis_method = st.sidebar.radio(
        "選擇分析方法:",
        ["KPI 快速分析", "數據上傳與FAB選擇", "敘述統計分析", "統計方法偵測", "移動平均偵測", "季節性分解偵測", "時序分析", "KPI批量監控", "Level Shift 檢測", "趨勢動量分析", "異常趨勢分析"]
    )
    
    # 在側邊欄顯示當前選擇的資料狀態
    show_sidebar_data_status()
    
    # Route to different pages based on selection
    if analysis_method == "KPI 快速分析":
        kpi_quick_analysis_page()
    elif analysis_method == "數據上傳與FAB選擇":
        data_upload_fab_selection_page()
    elif analysis_method == "敘述統計分析":
        from descriptive_stats import descriptive_statistics_page
        descriptive_statistics_page()
    elif analysis_method == "統計方法偵測":
        statistical_detection_page()
    elif analysis_method == "移動平均偵測":
        moving_average_detection_page()
    elif analysis_method == "季節性分解偵測":
        seasonal_decomposition_page()
    elif analysis_method == "時序分析":
        from time_series_analysis import time_series_analysis_page
        time_series_analysis_page()
    elif analysis_method == "KPI批量監控":
        from batch_monitoring import batch_kpi_monitoring_page
        batch_kpi_monitoring_page()
    elif analysis_method == "Level Shift 檢測":
        from advanced_analysis import level_shift_detection_page
        level_shift_detection_page()
    elif analysis_method == "趨勢動量分析":
        from advanced_analysis import trend_momentum_analysis_page
        trend_momentum_analysis_page()
    elif analysis_method == "異常趨勢分析":
        from time_series_analysis import anomaly_trend_analysis_page
        anomaly_trend_analysis_page()

def show_sidebar_data_status():
    """在側邊欄顯示當前資料狀態"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 當前資料狀態")
    
    if st.session_state.raw_data is not None:
        st.sidebar.success("✅ 資料已載入")
        st.sidebar.write(f"📈 總筆數: {len(st.session_state.raw_data):,}")
        st.sidebar.write(f"🏭 FAB數: {st.session_state.raw_data['FAB'].nunique()}")
        st.sidebar.write(f"📊 KPI數: {st.session_state.raw_data['KPI'].nunique()}")
        
        if st.session_state.selected_fab:
            st.sidebar.info(f"🎯 已選FAB: {st.session_state.selected_fab}")
            if st.session_state.selected_kpi:
                st.sidebar.info(f"📈 已選KPI: {st.session_state.selected_kpi}")
        
        # 快速切換 FAB 和 KPI
        if st.session_state.raw_data is not None:
            available_fabs = sorted(st.session_state.raw_data['FAB'].unique())
            current_fab = st.sidebar.selectbox(
                "🏭 快速切換 FAB:",
                options=available_fabs,
                index=to_plotly_list(available_fabs.index(st.session_state.selected_fab)) if st.session_state.selected_fab in available_fabs else 0,
                key=to_plotly_list("sidebar_fab_selector"
            ))
            
            if current_fab != st.session_state.selected_fab:
                st.session_state.selected_fab = current_fab
                fab_data = st.session_state.raw_data[st.session_state.raw_data['FAB'] == current_fab]
                st.session_state.fab_data = fab_data
                st.session_state.available_kpis = sorted(fab_data['KPI'].unique())
                st.session_state.selected_kpi = st.session_state.available_kpis[0] if st.session_state.available_kpis else None
                st.info("🔄 請重新整理頁面以更新選擇")
            
            if st.session_state.available_kpis:
                current_kpi = st.sidebar.selectbox(
                    "📈 快速切換 KPI:",
                    options=st.session_state.available_kpis,
                    index=to_plotly_list(st.session_state.available_kpis.index(st.session_state.selected_kpi)) if st.session_state.selected_kpi in st.session_state.available_kpis else 0,
                    key=to_plotly_list("sidebar_kpi_selector"
                ))
                
                if current_kpi != st.session_state.selected_kpi:
                    st.session_state.selected_kpi = current_kpi
    else:
        st.sidebar.warning("⚠️ 尚未載入資料")
        if st.sidebar.button("🎯 載入範例資料", key=to_plotly_list("sidebar_load_sample")):
            sample_data = generate_fab_sample_data()
            sample_data = ensure_data_format(sample_data)
            st.session_state.raw_data = sample_data
            # 自動選擇第一個 FAB 和 KPI
            first_fab = sample_data['FAB'].iloc[0]
            st.session_state.selected_fab = first_fab
            fab_data = sample_data[sample_data['FAB'] == first_fab]
            st.session_state.fab_data = fab_data
            st.session_state.available_kpis = sorted(fab_data['KPI'].unique())
            st.session_state.selected_kpi = st.session_state.available_kpis[0] if st.session_state.available_kpis else None
            st.info("🔄 請重新整理頁面以更新選擇")
    
    # 移除主題切換功能以相容 Streamlit 1.12.0

def kpi_quick_analysis_page():
    """KPI 快速分析頁面 - 預設首頁"""
    st.header("🎯 KPI 快速分析")
    
    # 檢查是否有資料
    if st.session_state.raw_data is None:
        st.info("📁 開始分析前，請載入資料")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 使用範例資料")
            st.write("載入包含 4 個 FAB、12 種 KPI 的範例資料，包含真實的異常模式和特殊事件。")
            
            if st.button("🚀 載入範例資料開始分析", key=to_plotly_list("quick_load_sample")):
                with st.spinner("正在載入範例資料..."):
                    sample_data = generate_fab_sample_data()
                    sample_data = ensure_data_format(sample_data)
                    st.session_state.raw_data = sample_data
                    # 自動選擇第一個 FAB 和 KPI
                    first_fab = sample_data['FAB'].iloc[0]
                    st.session_state.selected_fab = first_fab
                    fab_data = sample_data[sample_data['FAB'] == first_fab]
                    st.session_state.fab_data = fab_data
                    st.session_state.available_kpis = sorted(fab_data['KPI'].unique())
                    st.session_state.selected_kpi = st.session_state.available_kpis[0] if st.session_state.available_kpis else None
                
                st.success("✅ 範例資料載入完成！🔄 請重新整理頁面")
        
        with col2:
            st.subheader("📁 上傳自己的資料")
            st.write("上傳符合 FAB KPI 格式的 CSV 或 Excel 檔案。")
            
            uploaded_file = st.file_uploader(
                "選擇檔案",
                type=['csv', 'xlsx', 'xls'],
                key=to_plotly_list("quick_upload"
            ))
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # 確保數據格式正確
                    try:
                        df = ensure_data_format(df)
                        st.session_state.raw_data = df
                        # 自動選擇第一個 FAB 和 KPI
                        first_fab = df['FAB'].iloc[0]
                        st.session_state.selected_fab = first_fab
                        fab_data = df[df['FAB'] == first_fab]
                        st.session_state.fab_data = fab_data
                        st.session_state.available_kpis = sorted(fab_data['KPI'].unique())
                        st.session_state.selected_kpi = st.session_state.available_kpis[0] if st.session_state.available_kpis else None
                        
                        st.success("✅ 資料上傳成功！🔄 請重新整理頁面")
                    except Exception as format_error:
                        st.error(f"❌ 數據格式錯誤: {str(format_error)}")
                        st.info("💡 請確保: 1) 包含 FAB, VALUE, KPI, REPORT_TIME 欄位 2) VALUE 為數值 3) REPORT_TIME 為有效日期格式")
                except Exception as e:
                    st.error(f"❌ 讀取檔案時發生錯誤: {str(e)}")
        
        return
    
    # 有資料時的快速分析介面
    st.subheader(f"🏭 {st.session_state.selected_fab} - {st.session_state.selected_kpi} 分析")
    
    # 獲取當前選擇的 KPI 資料
    if st.session_state.fab_data is not None and st.session_state.selected_kpi:
        kpi_data = st.session_state.fab_data[st.session_state.fab_data['KPI'] == st.session_state.selected_kpi].copy()
        kpi_data = kpi_data.sort_values('REPORT_TIME')
        
        if len(kpi_data) == 0:
            st.warning("⚠️ 選擇的 KPI 無資料")
            return
        
        # 快速統計
        display_kpi_quick_stats(kpi_data, st.session_state.selected_kpi, st.session_state.selected_fab)
        
        # 快速分析選項
        st.subheader("🔍 快速異常偵測")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 統計方法偵測", key=to_plotly_list("quick_statistical")):
                st.info("💡 請從左側選單選擇「統計方法偵測」進行分析")
        
        with col2:
            if st.button("📈 移動平均偵測", key=to_plotly_list("quick_ma")):
                st.info("💡 請從左側選單選擇「移動平均偵測」進行分析")
        
        with col3:
            if st.button("🔄 季節性分解", key=to_plotly_list("quick_seasonal")):
                st.info("💡 請從左側選單選擇「季節性分解偵測」進行分析")
        
        # 執行基本的異常偵測預覽
        st.subheader("📈 基本趨勢與異常預覽")
        
        # 簡單的 Z-Score 異常偵測
        values = kpi_data['VALUE'].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        z_scores = np.abs((values - mean_val) / std_val)
        outliers = z_scores > 2.0
        
        # 創建圖表
        fig = go.Figure()
        
        # 準備數據
        x_data, y_data = prepare_plotly_data(kpi_data['REPORT_TIME'], kpi_data['VALUE'])
        
        # 原始數據
        fig.add_trace(go.Scatter(
            x=to_plotly_list(x_data), y=to_plotly_list(y_data),
            mode='lines+markers',
            name='原始數據',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # 異常點
        if np.any(outliers):
            outlier_x, outlier_y = prepare_plotly_data(
                kpi_data[outliers]['REPORT_TIME'], 
                kpi_data[outliers]['VALUE']
            )
            
            fig.add_trace(go.Scatter(
                x=to_plotly_list(outlier_x), y=to_plotly_list(outlier_y),
                mode='markers',
                name='可能異常點',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        # 添加均值線
        fig.add_hline(y=to_plotly_list(mean_val), line_dash="dash", line_color="green", 
                     annotation_text=f"平均值: {mean_val:.2f}")
        
        # 添加 2σ 閾值線
        fig.add_hline(y=to_plotly_list(mean_val + 2*std_val), line_dash="dash", line_color="orange", 
                     annotation_text="上閾值 (2σ)")
        fig.add_hline(y=to_plotly_list(mean_val - 2*std_val), line_dash="dash", line_color="orange", 
                     annotation_text="下閾值 (2σ)")
        
        fig.update_layout(
            title=f"{st.session_state.selected_fab} - {st.session_state.selected_kpi} 趨勢圖",
            xaxis_title="時間",
            yaxis_title="數值",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig)
        
        # 顯示異常統計
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("資料點數", len(kpi_data))
        
        with col2:
            st.metric("可能異常點", np.sum(outliers))
        
        with col3:
            anomaly_rate = np.sum(outliers) / len(kpi_data) * 100
            st.metric("異常率", f"{anomaly_rate:.1f}%")
        
        with col4:
            trend_slope = np.polyfit(range(len(values)), values, 1)[0]
            trend_direction = "上升" if trend_slope > 0 else "下降" if trend_slope < 0 else "平穩"
            st.metric("趨勢", trend_direction)
        
        # 提供進階分析建議
        st.subheader("💡 分析建議")
        
        suggestions = get_analysis_suggestions(kpi_data, st.session_state.selected_kpi, outliers)
        for suggestion in suggestions:
            st.info(suggestion)

def display_kpi_quick_stats(kpi_data: pd.DataFrame, kpi_name: str, fab_name: str):
    """顯示 KPI 快速統計"""
    values = kpi_data['VALUE'].values
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("最新值", f"{values[-1]:.2f}")
    
    with col2:
        st.metric("平均值", f"{np.mean(values):.2f}")
    
    with col3:
        st.metric("最大值", f"{np.max(values):.2f}")
    
    with col4:
        st.metric("最小值", f"{np.min(values):.2f}")
    
    with col5:
        st.metric("標準差", f"{np.std(values):.2f}")

def get_analysis_suggestions(kpi_data: pd.DataFrame, kpi_name: str, outliers: np.ndarray) -> list:
    """根據 KPI 特性和數據提供分析建議"""
    suggestions = []
    
    values = kpi_data['VALUE'].values
    outlier_rate = np.sum(outliers) / len(outliers) * 100
    
    # 基於異常率的建議
    if outlier_rate > 5:
        suggestions.append(f"🔴 異常率較高 ({outlier_rate:.1f}%)，建議使用統計方法進行深度分析")
    elif outlier_rate > 2:
        suggestions.append(f"🟡 異常率中等 ({outlier_rate:.1f}%)，可考慮移動平均方法減少雜訊")
    else:
        suggestions.append(f"🟢 異常率較低 ({outlier_rate:.1f}%)，數據品質良好")
    
    # 基於 KPI 類型的建議
    if kpi_name in ['Yield', 'First_Pass_Yield', 'Quality_Score']:
        suggestions.append("💡 良率類指標建議關注連續下降趨勢，可使用移動平均偵測")
    elif kpi_name in ['Defect_Rate', 'Rework_Rate']:
        suggestions.append("💡 缺陷類指標建議監控突增異常，統計方法效果較好")
    elif kpi_name in ['Throughput', 'Equipment_Utilization', 'OEE']:
        suggestions.append("💡 效率類指標建議分析週期性模式，可嘗試季節性分解")
    elif kpi_name in ['Cycle_Time', 'WIP_Level']:
        suggestions.append("💡 時間/庫存類指標建議監控趨勢變化，移動平均方法適用")
    
    # 基於數據特性的建議
    data_range = len(kpi_data)
    if data_range > 90:
        suggestions.append("📈 數據充足，建議嘗試季節性分解分析長期趨勢")
    elif data_range < 30:
        suggestions.append("📊 數據較少，建議使用統計方法或移動平均分析")
    
    return suggestions

def data_upload_fab_selection_page():
    st.header("🏭 數據上傳與 FAB 選擇")
    
    # File upload
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "上傳 FAB KPI 資料文件 (CSV, Excel)", 
            type=['csv', 'xlsx', 'xls'],
            help="檔案應包含 FAB, VALUE, KPI, REPORT_TIME 欄位"
        )
    
    with col2:
        st.write("**📋 資料格式範例**")
        # 生成範例CSV內容
        sample_csv = """FAB,KPI,REPORT_TIME,VALUE
FAB12A,Yield,2024-01-01,92.5
FAB12A,Throughput,2024-01-01,850
FAB14B,Yield,2024-01-01,89.5"""
        
        st.download_button(
            label="📥 下載格式範例",
            data=sample_csv,
            file_name="fab_kpi_format_sample.csv",
            mime="text/csv",
            help="下載標準的 FAB KPI 資料格式範例"
        )
        
        if st.button("📖 格式說明"):
            st.info("""
            **標準格式要求:**
            - **FAB**: 工廠代碼 (如: FAB12A)
            - **KPI**: 指標名稱 (如: Yield)  
            - **REPORT_TIME**: 日期 (YYYY-MM-DD)
            - **VALUE**: 數值
            
            **注意事項:**
            - 欄位名稱需完全一致
            - 日期格式建議 YYYY-MM-DD
            - 數值欄位不可包含文字
            """)    
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ 成功載入數據: {df.shape[0]} 行, {df.shape[1]} 列")
            
            # Validate required columns
            required_columns = ['FAB', 'VALUE', 'KPI', 'REPORT_TIME']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ 缺少必要欄位: {', '.join(missing_columns)}")
                st.info("💡 檔案必須包含以下欄位: FAB, VALUE, KPI, REPORT_TIME")
                return
            
            # Convert REPORT_TIME to datetime
            try:
                df['REPORT_TIME'] = pd.to_datetime(df['REPORT_TIME'])
                df = df.sort_values(['FAB', 'KPI', 'REPORT_TIME'])
                st.success("✅ 時間欄位轉換完成")
            except Exception as e:
                st.error(f"❌ 時間欄位轉換失敗: {str(e)}")
                return
            
            # Store raw data
            st.session_state.raw_data = df
            
            # Display basic info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 數據預覽")
                st.dataframe(df.head(10))
            
            with col2:
                st.subheader("📈 數據統計")
                st.write(f"**總資料筆數:** {len(df):,}")
                st.write(f"**FAB 數量:** {df['FAB'].nunique()}")
                st.write(f"**KPI 種類:** {df['KPI'].nunique()}")
                st.write(f"**時間範圍:** {df['REPORT_TIME'].min().strftime('%Y-%m-%d')} ~ {df['REPORT_TIME'].max().strftime('%Y-%m-%d')}")
                
                st.write("**FAB 列表:**")
                for fab in sorted(df['FAB'].unique()):
                    kpi_count = df[df['FAB'] == fab]['KPI'].nunique()
                    data_points = len(df[df['FAB'] == fab])
                    st.write(f"- {fab} ({kpi_count} KPIs, {data_points:,} 筆)")
            
            # FAB Selection
            st.subheader("🏭 選擇 FAB")
            available_fabs = sorted(df['FAB'].unique())
            selected_fab = st.selectbox(
                "選擇要分析的 FAB:",
                options=available_fabs,
                help="選擇特定 FAB 進行 KPI 監控分析"
            )
            
            if selected_fab:
                fab_data = df[df['FAB'] == selected_fab].copy()
                available_kpis = sorted(fab_data['KPI'].unique())
                
                st.session_state.selected_fab = selected_fab
                st.session_state.fab_data = fab_data
                st.session_state.available_kpis = available_kpis
                
                st.success(f"✅ 已選擇 FAB: {selected_fab}")
                
                # Show FAB characteristics if it's sample data
                if 'FAB12A' in available_fabs:  # 檢查是否為範例數據
                    show_fab_characteristics(selected_fab)
                
                # Show FAB KPI overview
                st.subheader(f"📈 {selected_fab} KPI 概覽")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("KPI 數量", len(available_kpis))
                with col2:
                    st.metric("資料點數", len(fab_data))
                with col3:
                    latest_date = fab_data['REPORT_TIME'].max()
                    st.metric("最新數據", latest_date.strftime('%Y-%m-%d'))
                with col4:
                    # 計算整體資料品質評分
                    data_quality_score = calculate_data_quality_score(fab_data)
                    st.metric("資料品質", f"{data_quality_score:.1f}/10")
                
                # KPI selection for preview
                st.subheader("🔍 KPI 預覽")
                preview_kpis = st.multiselect(
                    "選擇要預覽的 KPI (最多5個):",
                    options=available_kpis,
                    default=available_kpis[:min(5, len(available_kpis))],
                                    )
                
                if preview_kpis:
                    # Create pivot table for visualization
                    pivot_data = fab_data[fab_data['KPI'].isin(preview_kpis)].pivot_table(
                        index=to_plotly_list('REPORT_TIME'), 
                        columns='KPI', 
                        values=to_plotly_list('VALUE'), 
                        aggfunc='mean'
                    ).reset_index()
                    
                    # Create time series plot
                    fig = go.Figure()
                    for kpi in preview_kpis:
                        if kpi in pivot_data.columns:
                            fig.add_trace(go.Scatter(
                                x=to_plotly_list(pivot_data['REPORT_TIME']), y=to_plotly_list(pivot_data[kpi]),
                                mode='lines+markers',
                                name=kpi,
                                line=dict(width=2),
                                marker=dict(size=4)
                            ))
                    
                    fig.update_layout(
                        title=f"{selected_fab} - KPI 時序資料趨勢",
                        xaxis_title="報告時間",
                        yaxis_title="數值",
                        hovermode='x unified',
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=to_plotly_list(1.02),
                            xanchor="right",
                            x=to_plotly_list(1
                        ))
                    )
                    
                    st.plotly_chart(fig)
                
                st.info("💡 FAB 數據已準備完成！請從左側選單選擇分析方法進行監控。")
                
        except Exception as e:
            st.error(f"❌ 讀取文件時發生錯誤: {str(e)}")
            st.info("💡 請確保檔案格式正確，且包含必要的欄位：FAB, VALUE, KPI, REPORT_TIME")
    else:
        st.info("📁 請上傳包含 FAB KPI 資料的文件開始分析")
        
        # Sample data option
        st.subheader("🎯 範例數據")
        
        with st.expander("查看範例數據說明", expanded=False):
            st.markdown("""
            ### 📋 範例數據特色
            
            **🏭 4 個不同特性的 FAB:**
            - **FAB12A** (28nm): 成熟製程，高穩定性
            - **FAB14B** (14nm): 量產爬坡中，中等穩定性  
            - **FAB16** (16nm): 成熟製程，高穩定性
            - **FAB18** (7nm): 新製程，較低穩定性
            
            **📊 12 種 FAB KPI 指標:**
            - Yield (良率), Throughput (產能), Defect_Rate (缺陷率)
            - Equipment_Utilization (設備利用率), Cycle_Time (週期時間)
            - WIP_Level (在製品水準), Cost_Per_Unit (單位成本)
            - Quality_Score (品質分數), OEE (整體設備效率)
            - First_Pass_Yield (首次通過良率), Rework_Rate (重工率)
            - Critical_Dimension (關鍵尺寸均勻性)
            
            **⚡ 特殊事件模擬:**
            - 設備故障、製程改善、原料短缺
            - 新配方導入、設備維護、產能擴充
            
            **📈 資料特性:**
            - 2年每日資料 (2023-2024)
            - 包含趨勢、季節性、週期性
            - 真實的異常模式和事件影響
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 載入範例數據"):
                with st.spinner("正在生成範例數據..."):
                    sample_data = generate_fab_sample_data()
                    sample_data = ensure_data_format(sample_data)
                    st.session_state.raw_data = sample_data
                st.success("✅ 範例數據已載入！請選擇 FAB 進行分析。🔄 請重新整理頁面")
        
        with col2:
            if st.button("📊 預覽範例統計"):
                with st.spinner("正在生成統計資訊..."):
                    sample_data = generate_fab_sample_data()
                    show_sample_data_preview(sample_data)

def show_sample_data_preview(sample_data: pd.DataFrame):
    """顯示範例數據預覽和統計"""
    st.subheader("📊 範例數據統計預覽")
    
    # 基本統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總資料筆數", f"{len(sample_data):,}")
    
    with col2:
        st.metric("FAB 數量", sample_data['FAB'].nunique())
    
    with col3:
        st.metric("KPI 種類", sample_data['KPI'].nunique())
    
    with col4:
        date_range = (sample_data['REPORT_TIME'].max() - sample_data['REPORT_TIME'].min()).days
        st.metric("時間跨度", f"{date_range} 天")
    
    # FAB 詳細統計
    st.subheader("🏭 各 FAB 統計資訊")
    
    fab_stats = []
    for fab in sample_data['FAB'].unique():
        fab_data = sample_data[sample_data['FAB'] == fab]
        
        # 計算異常率（簡單的 Z-Score 方法）
        anomaly_counts = {}
        for kpi in fab_data['KPI'].unique():
            kpi_values = fab_data[fab_data['KPI'] == kpi]['VALUE'].values
            z_scores = np.abs((kpi_values - np.mean(kpi_values)) / np.std(kpi_values))
            anomaly_counts[kpi] = np.sum(z_scores > 2)
        
        avg_anomaly_rate = np.mean(list(anomaly_counts.values())) / len(fab_data) * fab_data['KPI'].nunique() * 100
        
        fab_stats.append({
            'FAB': fab,
            'KPI數量': fab_data['KPI'].nunique(),
            '資料點數': len(fab_data),
            '平均異常率': f"{avg_anomaly_rate:.2f}%",
            '時間範圍': f"{fab_data['REPORT_TIME'].min().strftime('%Y-%m')} ~ {fab_data['REPORT_TIME'].max().strftime('%Y-%m')}"
        })
    
    fab_stats_df = pd.DataFrame(fab_stats)
    st.dataframe(fab_stats_df)
    
    # KPI 數值範圍統計
    st.subheader("📈 各 KPI 數值範圍")
    
    kpi_stats = []
    for kpi in sample_data['KPI'].unique():
        kpi_data = sample_data[sample_data['KPI'] == kpi]['VALUE']
        
        kpi_stats.append({
            'KPI': kpi,
            '最小值': f"{kpi_data.min():.2f}",
            '最大值': f"{kpi_data.max():.2f}",
            '平均值': f"{kpi_data.mean():.2f}",
            '標準差': f"{kpi_data.std():.2f}",
            '資料點數': len(kpi_data)
        })
    
    kpi_stats_df = pd.DataFrame(kpi_stats)
    st.dataframe(kpi_stats_df)
    
    # 特殊事件時間線
    st.subheader("⚡ 特殊事件時間線")
    
    events_info = [
        {'日期': '2023-03-15', '事件': '設備故障', '類型': '負面', '持續天數': 7},
        {'日期': '2023-06-20', '事件': '製程改善', '類型': '正面', '持續天數': 30},
        {'日期': '2023-09-10', '事件': '原料短缺', '類型': '負面', '持續天數': 14},
        {'日期': '2024-02-01', '事件': '新配方導入', '類型': '混合', '持續天數': 21},
        {'日期': '2024-07-15', '事件': '設備維護', '類型': '負面', '持續天數': 3},
        {'日期': '2024-10-01', '事件': '產能擴充', '類型': '正面', '持續天數': 45}
    ]
    
    events_df = pd.DataFrame(events_info)
    
    # 用顏色標示不同類型的事件
    def color_event_type(val):
        if val == '正面':
            return 'background-color: #d4edda'
        elif val == '負面':
            return 'background-color: #f8d7da'
        else:
            return 'background-color: #fff3cd'
    
    styled_events = events_df.style.applymap(color_event_type, subset=['類型'])
    st.dataframe(styled_events)
    
    # 簡單的 KPI 趨勢圖
    st.subheader("📊 主要 KPI 趨勢預覽")
    
    # 選擇一個 FAB 和幾個主要 KPI 做預覽
    preview_fab = sample_data['FAB'].iloc[0]
    main_kpis = ['Yield', 'Throughput', 'Defect_Rate', 'Equipment_Utilization']
    
    fab_preview_data = sample_data[sample_data['FAB'] == preview_fab]
    
    # 轉換為透視表格式
    pivot_preview = fab_preview_data[fab_preview_data['KPI'].isin(main_kpis)].pivot_table(
        index=to_plotly_list('REPORT_TIME'), columns='KPI', values=to_plotly_list('VALUE'), aggfunc='mean'
    ).reset_index()
    
    if not pivot_preview.empty:
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, kpi in enumerate(main_kpis):
            if kpi in pivot_preview.columns:
                fig.add_trace(go.Scatter(
                    x=to_plotly_list(pivot_preview['REPORT_TIME']), y=to_plotly_list(pivot_preview[kpi]),
                    mode='lines',
                    name=kpi,
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
        
        fig.update_layout(
            title=f"{preview_fab} - 主要 KPI 趨勢預覽",
            xaxis_title="時間",
            yaxis_title="數值",
            hovermode='x unified',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=to_plotly_list(1.02),
                xanchor="right",
                x=to_plotly_list(1
            ))
        )
        
        st.plotly_chart(fig)
    
    st.info("💡 以上為範例數據的統計預覽。點擊「載入範例數據」開始進行完整分析！")

def show_fab_characteristics(selected_fab: str):
    """顯示範例 FAB 的特性說明"""
    fab_info = {
        'FAB12A': {
            'tech_node': '28nm',
            'maturity': '成熟製程',
            'stability': '高穩定性',
            'description': '成熟的28nm製程廠，具有高穩定性和優異的良率表現',
            'characteristics': ['低變異性', '穩定產能', '成熟工藝'],
            'color': '#28a745'
        },
        'FAB14B': {
            'tech_node': '14nm',
            'maturity': '量產爬坡',
            'stability': '中等穩定性',
            'description': '14nm製程廠正在量產爬坡階段，效能持續優化中',
            'characteristics': ['中等變異性', '成長中產能', '學習曲線'],
            'color': '#ffc107'
        },
        'FAB16': {
            'tech_node': '16nm',
            'maturity': '成熟製程',
            'stability': '高穩定性',
            'description': '穩定的16nm製程廠，平衡了效能與成本',
            'characteristics': ['穩定表現', '平衡成本', '可靠工藝'],
            'color': '#17a2b8'
        },
        'FAB18': {
            'tech_node': '7nm',
            'maturity': '新製程',
            'stability': '較低穩定性',
            'description': '最先進的7nm製程廠，仍在工藝優化和穩定化階段',
            'characteristics': ['高變異性', '技術挑戰', '創新工藝'],
            'color': '#dc3545'
        }
    }
    
    if selected_fab in fab_info:
        info = fab_info[selected_fab]
        
        st.info(f"""
        **🏭 {selected_fab} 特性說明**
        
        **🔧 製程節點:** {info['tech_node']}  
        **📊 成熟度:** {info['maturity']}  
        **⚡ 穩定性:** {info['stability']}
        
        **📝 描述:** {info['description']}
        
        **🎯 主要特徵:** {' • '.join(info['characteristics'])}
        """)

def calculate_data_quality_score(fab_data: pd.DataFrame) -> float:
    """計算資料品質評分 (0-10)"""
    score = 10.0
    
    # 檢查缺失值
    missing_ratio = fab_data['VALUE'].isnull().sum() / len(fab_data)
    score -= missing_ratio * 3
    
    # 檢查異常值比例
    total_outliers = 0
    total_points = 0
    
    for kpi in fab_data['KPI'].unique():
        kpi_values = fab_data[fab_data['KPI'] == kpi]['VALUE'].values
        if len(kpi_values) > 0:
            z_scores = np.abs((kpi_values - np.mean(kpi_values)) / np.std(kpi_values))
            outliers = np.sum(z_scores > 3)
            total_outliers += outliers
            total_points += len(kpi_values)
    
    if total_points > 0:
        outlier_ratio = total_outliers / total_points
        score -= outlier_ratio * 30  # 異常值影響較大
    
    # 檢查資料完整性（時間連續性）
    dates = pd.to_datetime(fab_data['REPORT_TIME']).sort_values()
    expected_days = (dates.max() - dates.min()).days + 1
    actual_days = dates.nunique()
    completeness = actual_days / expected_days
    score -= (1 - completeness) * 2
    
    # 檢查 KPI 多樣性
    kpi_diversity = fab_data['KPI'].nunique() / 12  # 12 是最大 KPI 數量
    if kpi_diversity < 0.5:
        score -= (0.5 - kpi_diversity) * 4
    
    return max(0, min(10, score))

def generate_fab_sample_data():
    """生成 FAB KPI 範例數據 (使用真實數據生成器)"""
    from realistic_data_generator import generate_realistic_fab_sample_data
    return generate_realistic_fab_sample_data()

def generate_old_fab_sample_data():
    """原始 FAB KPI 範例數據生成器 (備用)"""
    np.random.seed(42)
    
    # 基本設定
    fabs = ['FAB12A', 'FAB14B', 'FAB16', 'FAB18']
    kpis = ['Yield', 'Throughput', 'Defect_Rate', 'Equipment_Utilization', 
            'Cycle_Time', 'WIP_Level', 'Cost_Per_Unit', 'Quality_Score', 
            'OEE', 'First_Pass_Yield', 'Rework_Rate', 'Critical_Dimension']
    
    # 生成兩年的日資料
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    
    data_list = []
    
    # 為不同 FAB 設定不同特性
    fab_characteristics = {
        'FAB12A': {'maturity': 'mature', 'stability': 'high', 'tech_node': '28nm'},
        'FAB14B': {'maturity': 'ramping', 'stability': 'medium', 'tech_node': '14nm'},  
        'FAB16': {'maturity': 'mature', 'stability': 'high', 'tech_node': '16nm'},
        'FAB18': {'maturity': 'new', 'stability': 'low', 'tech_node': '7nm'}
    }
    
    # 添加特殊事件影響
    special_events = {
        '2023-03-15': {'type': 'equipment_down', 'impact': 'negative', 'duration': 7},
        '2023-06-20': {'type': 'process_improvement', 'impact': 'positive', 'duration': 30},
        '2023-09-10': {'type': 'material_shortage', 'impact': 'negative', 'duration': 14},
        '2024-02-01': {'type': 'new_recipe', 'impact': 'mixed', 'duration': 21},
        '2024-07-15': {'type': 'maintenance', 'impact': 'negative', 'duration': 3},
        '2024-10-01': {'type': 'capacity_expansion', 'impact': 'positive', 'duration': 45}
    }
    
    for fab in fabs:
        fab_char = fab_characteristics[fab]
        
        for kpi in kpis:
            # 為每個 FAB-KPI 組合生成不同特性的數據
            base_values = generate_enhanced_kpi_data(
                kpi, len(dates), fab, fab_char, dates, special_events
            )
            
            for i, date in enumerate(dates):
                data_list.append({
                    'FAB': fab,
                    'KPI': kpi,
                    'REPORT_TIME': date,
                    'VALUE': base_values[i]
                })
    
    return pd.DataFrame(data_list)

def generate_enhanced_kpi_data(kpi: str, length: int, fab: str, fab_char: dict, 
                              dates: pd.DatetimeIndex, special_events: dict) -> np.ndarray:
    """生成增強的 KPI 時序資料，包含 FAB 特性和特殊事件"""
    np.random.seed(hash(kpi + fab) % 2147483647)
    
    # 根據 FAB 成熟度調整基礎參數
    stability_factor = {'high': 0.3, 'medium': 0.6, 'low': 1.0}[fab_char['stability']]
    maturity_factor = {'mature': 1.0, 'ramping': 0.85, 'new': 0.7}[fab_char['maturity']]
    
    if kpi == 'Yield':
        base = (94 - 2 * (1 - maturity_factor)) + np.random.randn() * stability_factor
        trend = np.cumsum(np.random.randn(length) * 0.008 * stability_factor)
        seasonal = 1.5 * np.sin(np.arange(length) * 2 * np.pi / 30)  # 月週期
        weekly = 0.8 * np.sin(np.arange(length) * 2 * np.pi / 7)   # 週週期
        noise = np.random.randn(length) * 0.4 * stability_factor
        values = base + trend + seasonal + weekly + noise
        values = np.clip(values, 82, 98.5)
        
    elif kpi == 'Throughput':
        base = (800 * maturity_factor) + np.random.randn() * 50
        trend = np.cumsum(np.random.randn(length) * 0.3 * stability_factor)
        seasonal = 60 * np.sin(np.arange(length) * 2 * np.pi / 7)  # 週週期  
        monthly = 30 * np.sin(np.arange(length) * 2 * np.pi / 30)  # 月週期
        noise = np.random.randn(length) * 15 * stability_factor
        values = base + trend + seasonal + monthly + noise
        values = np.clip(values, 100, 1500)
        
    elif kpi == 'Defect_Rate':
        base = (1.0 + 1.5 * (1 - maturity_factor)) + np.random.randn() * 0.2
        trend = np.cumsum(np.random.randn(length) * 0.002 * stability_factor)
        seasonal = 0.15 * np.sin(np.arange(length) * 2 * np.pi / 14)  # 雙週週期
        noise = np.random.randn(length) * 0.08 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 0.1, 6.0)
        
    elif kpi == 'Equipment_Utilization':
        base = (90 * maturity_factor) + np.random.randn() * 2
        trend = np.cumsum(np.random.randn(length) * 0.008 * stability_factor)
        seasonal = 4 * np.sin(np.arange(length) * 2 * np.pi / 7)  # 週週期
        noise = np.random.randn(length) * 1.2 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 65, 98)
        
    elif kpi == 'Cycle_Time':
        base = (40 + 15 * (1 - maturity_factor)) + np.random.randn() * 3
        trend = np.cumsum(np.random.randn(length) * 0.015 * stability_factor)
        seasonal = 4 * np.sin(np.arange(length) * 2 * np.pi / 30)  # 月週期
        noise = np.random.randn(length) * 1.8 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 20, 85)
        
    elif kpi == 'WIP_Level':
        base = (3000 * maturity_factor) + np.random.randn() * 150
        trend = np.cumsum(np.random.randn(length) * 1.5 * stability_factor)
        seasonal = 250 * np.sin(np.arange(length) * 2 * np.pi / 30)  # 月週期
        noise = np.random.randn(length) * 40 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 800, 6500)
        
    elif kpi == 'Cost_Per_Unit':
        base = (100 + 30 * (1 - maturity_factor)) + np.random.randn() * 8
        trend = np.cumsum(np.random.randn(length) * 0.04 * stability_factor)
        seasonal = 6 * np.sin(np.arange(length) * 2 * np.pi / 90)  # 季度週期
        noise = np.random.randn(length) * 2.5 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 50, 220)
        
    elif kpi == 'Quality_Score':
        base = (92 * maturity_factor) + np.random.randn() * 2
        trend = np.cumsum(np.random.randn(length) * 0.004 * stability_factor)
        seasonal = 1.5 * np.sin(np.arange(length) * 2 * np.pi / 30)  # 月週期
        noise = np.random.randn(length) * 0.8 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 70, 100)
        
    elif kpi == 'OEE':  # Overall Equipment Effectiveness
        base = (85 * maturity_factor) + np.random.randn() * 3
        trend = np.cumsum(np.random.randn(length) * 0.01 * stability_factor)
        seasonal = 3 * np.sin(np.arange(length) * 2 * np.pi / 7)  # 週週期
        noise = np.random.randn(length) * 1.5 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 60, 95)
        
    elif kpi == 'First_Pass_Yield':
        base = (89 * maturity_factor) + np.random.randn() * 2.5
        trend = np.cumsum(np.random.randn(length) * 0.006 * stability_factor)
        seasonal = 2 * np.sin(np.arange(length) * 2 * np.pi / 30)  # 月週期
        noise = np.random.randn(length) * 1.2 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 75, 98)
        
    elif kpi == 'Rework_Rate':
        base = (3.0 + 2.0 * (1 - maturity_factor)) + np.random.randn() * 0.4
        trend = np.cumsum(np.random.randn(length) * 0.003 * stability_factor)
        seasonal = 0.3 * np.sin(np.arange(length) * 2 * np.pi / 14)  # 雙週週期
        noise = np.random.randn(length) * 0.15 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 0.5, 12.0)
        
    elif kpi == 'Critical_Dimension':  # CD uniformity (nm)
        base = (2.5 + 1.0 * (1 - maturity_factor)) + np.random.randn() * 0.3
        trend = np.cumsum(np.random.randn(length) * 0.002 * stability_factor)
        seasonal = 0.2 * np.sin(np.arange(length) * 2 * np.pi / 7)  # 週週期
        noise = np.random.randn(length) * 0.1 * stability_factor
        values = base + trend + seasonal + noise
        values = np.clip(values, 1.0, 8.0)
    
    else:
        # 預設值
        values = 100 + np.cumsum(np.random.randn(length) * 0.5 * stability_factor)
    
    # 應用特殊事件影響
    values = apply_special_events(values, dates, special_events, kpi, fab_char)
    
    # 添加隨機異常值（根據 FAB 穩定性調整）
    outlier_prob = 0.015 * stability_factor  # 穩定性低的 FAB 異常更多
    outliers = np.random.random(length) < outlier_prob
    
    # 不同類型的異常
    outlier_types = np.random.choice(['spike', 'dip', 'shift'], size=np.sum(outliers))
    outlier_indices = np.where(outliers)[0]
    
    for i, outlier_type in enumerate(outlier_types):
        idx = outlier_indices[i]
        if outlier_type == 'spike':
            values[idx] *= 1 + np.random.uniform(0.2, 0.8)
        elif outlier_type == 'dip':
            values[idx] *= 1 - np.random.uniform(0.15, 0.6)
        elif outlier_type == 'shift':
            # 連續幾天的偏移
            shift_duration = np.random.randint(2, 8)
            shift_magnitude = np.random.uniform(-0.3, 0.3)
            end_idx = min(idx + shift_duration, length)
            values[idx:end_idx] *= (1 + shift_magnitude)
    
    return values

def apply_special_events(values: np.ndarray, dates: pd.DatetimeIndex, 
                        special_events: dict, kpi: str, fab_char: dict) -> np.ndarray:
    """應用特殊事件對 KPI 的影響"""
    values_modified = values.copy()
    
    for event_date_str, event_info in special_events.items():
        event_date = pd.to_datetime(event_date_str)
        
        # 找到事件開始的索引
        try:
            start_idx = dates.get_loc(event_date)
        except KeyError:
            continue
            
        end_idx = min(start_idx + event_info['duration'], len(values))
        
        # 根據事件類型和 KPI 計算影響
        impact_magnitude = calculate_event_impact(event_info, kpi, fab_char)
        
        if impact_magnitude != 0:
            # 應用漸進式影響（事件開始時影響最大，逐漸恢復）
            for i in range(start_idx, end_idx):
                decay_factor = 1 - (i - start_idx) / event_info['duration']
                values_modified[i] *= (1 + impact_magnitude * decay_factor)
    
    return values_modified

def calculate_event_impact(event_info: dict, kpi: str, fab_char: dict) -> float:
    """計算特殊事件對特定 KPI 的影響程度"""
    event_type = event_info['type']
    impact_direction = event_info['impact']
    
    # 基礎影響程度
    base_impacts = {
        'equipment_down': {
            'Yield': -0.15, 'Throughput': -0.25, 'Defect_Rate': 0.3,
            'Equipment_Utilization': -0.20, 'OEE': -0.18, 'Cycle_Time': 0.15
        },
        'process_improvement': {
            'Yield': 0.08, 'First_Pass_Yield': 0.10, 'Defect_Rate': -0.20,
            'Quality_Score': 0.12, 'Rework_Rate': -0.25, 'Cost_Per_Unit': -0.05
        },
        'material_shortage': {
            'Throughput': -0.18, 'WIP_Level': -0.15, 'Cycle_Time': 0.20,
            'Cost_Per_Unit': 0.08, 'Equipment_Utilization': -0.12
        },
        'new_recipe': {
            'Yield': -0.05, 'Defect_Rate': 0.15, 'Cycle_Time': 0.10,
            'First_Pass_Yield': -0.08, 'Critical_Dimension': 0.12
        },
        'maintenance': {
            'Equipment_Utilization': -0.30, 'Throughput': -0.20, 'OEE': -0.25,
            'Cycle_Time': 0.15
        },
        'capacity_expansion': {
            'Throughput': 0.15, 'WIP_Level': 0.20, 'Equipment_Utilization': 0.10,
            'Cost_Per_Unit': -0.08
        }
    }
    
    # 根據 FAB 成熟度調整影響程度
    maturity_multiplier = {'mature': 0.7, 'ramping': 1.0, 'new': 1.3}[fab_char['maturity']]
    
    base_impact = base_impacts.get(event_type, {}).get(kpi, 0)
    
    return base_impact * maturity_multiplier

def statistical_detection_page():
    st.header("📊 統計方法異常偵測")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 當前 KPI: **{st.session_state.selected_kpi}**")
    
    st.subheader("⚙️ 偵測參數設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        detection_method = st.selectbox(
            "選擇統計方法:",
            ["Z-Score", "IQR (四分位距)", "Modified Z-Score"]
        )
    
    with col2:
        if detection_method == "Z-Score" or detection_method == "Modified Z-Score":
            threshold = st.slider(
                "異常閾值 (標準差倍數):", 
                min_value=1.0, max_value=5.0, value=2.0, step=0.1
            )
        else:
            threshold = st.slider(
                "IQR 倍數閾值:", 
                min_value=1.0, max_value=3.0, value=1.5, step=0.1
            )
    
    # 預設使用當前選擇的 KPI，但允許切換
    current_kpi_index = available_kpis.index(st.session_state.selected_kpi) if st.session_state.selected_kpi in available_kpis else 0
    selected_kpi = st.selectbox(
        "選擇要分析的 KPI:",
        options=available_kpis,
        index=to_plotly_list(current_kpi_index
    ))
    
    # 更新 session state
    if selected_kpi != st.session_state.selected_kpi:
        st.session_state.selected_kpi = selected_kpi
    
    if st.button("🔍 執行異常偵測"):
        # 轉換資料格式
        kpi_data = fab_data[fab_data['KPI'] == selected_kpi].copy()
        kpi_data = kpi_data.sort_values('REPORT_TIME')
        
        if len(kpi_data) == 0:
            st.error("❌ 所選 KPI 無資料")
            return
            
        outliers_info = detect_statistical_outliers_fab(kpi_data, detection_method, threshold)
        
        st.subheader("📈 時序圖與異常點")
        
        # Create visualization with matplotlib
        dates = pd.to_datetime(kpi_data['REPORT_TIME'])
        values = kpi_data['VALUE'].values
        anomalies = outliers_info['outlier_indices']
        scores = outliers_info['scores']
        
        if detection_method == "Z-Score" or detection_method == "Modified Z-Score":
            fig = create_zscore_analysis_plot(
                dates=dates,
                values=values,
                z_scores=scores,
                threshold=threshold,
                outliers=anomalies,
                title=f"{selected_fab} - {selected_kpi}"
            )
        elif detection_method == "IQR (四分位距)":
            fig = create_iqr_analysis_plot(
                dates=dates,
                values=values,
                outliers=anomalies,
                iqr_multiplier=threshold,
                title=f"{selected_fab} - {selected_kpi}"
            )
        else:
            # Fallback to general anomaly plot
            fig = create_anomaly_plot(
                dates=dates,
                values=values,
                anomalies=anomalies,
                title=f"{selected_fab} - {selected_kpi}",
                method=detection_method,
                scores=scores,
                threshold=threshold
            )
        
        render_matplotlib_figure(fig)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("總數據點", len(kpi_data))
        
        with col2:
            st.metric("異常點數量", len(outliers_info['outlier_indices']))
        
        with col3:
            anomaly_rate = len(outliers_info['outlier_indices']) / len(kpi_data) * 100
            st.metric("異常率", f"{anomaly_rate:.2f}%")
        
        # Show outlier details
        if len(outliers_info['outlier_indices']) > 0:
            st.subheader("🔍 異常點詳細資訊")
            
            outlier_df = kpi_data.iloc[outliers_info['outlier_indices']].copy()
            outlier_df['異常程度'] = outliers_info['scores'][outliers_info['outlier_indices']]
            outlier_df = outlier_df.sort_values('異常程度', ascending=False)
            
            st.dataframe(outlier_df[['REPORT_TIME', 'VALUE', '異常程度']])
        else:
            st.success("✅ 未發現異常點")

def detect_statistical_outliers_fab(kpi_data, method, threshold):
    """適用於 FAB 資料結構的統計異常偵測"""
    data = kpi_data['VALUE'].values
    
    if method == "Z-Score":
        mean_val = np.mean(data)
        std_val = np.std(data)
        z_scores = np.abs((data - mean_val) / std_val)
        outlier_mask = z_scores > threshold
        scores = z_scores
        
    elif method == "Modified Z-Score":
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        modified_z_scores = 0.6745 * (data - median_val) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        scores = np.abs(modified_z_scores)
        mean_val = median_val
        std_val = mad
        
    elif method == "IQR (四分位距)":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        scores = np.maximum((Q1 - data) / IQR, (data - Q3) / IQR)
        mean_val = np.median(data)
        std_val = IQR
    
    outlier_indices = np.where(outlier_mask)[0]
    
    return {
        'outlier_indices': outlier_indices,
        'scores': scores,
        'mean': mean_val,
        'std': std_val
    }

def detect_statistical_outliers(df, kpi_column, date_column, method, threshold):
    data = df[kpi_column].values
    
    if method == "Z-Score":
        mean_val = np.mean(data)
        std_val = np.std(data)
        z_scores = np.abs((data - mean_val) / std_val)
        outlier_mask = z_scores > threshold
        scores = z_scores
        
    elif method == "Modified Z-Score":
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        modified_z_scores = 0.6745 * (data - median_val) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        scores = np.abs(modified_z_scores)
        mean_val = median_val
        std_val = mad
        
    elif method == "IQR (四分位距)":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        scores = np.maximum((Q1 - data) / IQR, (data - Q3) / IQR)
        mean_val = np.median(data)
        std_val = IQR
    
    outlier_indices = np.where(outlier_mask)[0]
    
    return {
        'outlier_indices': outlier_indices,
        'scores': scores,
        'mean': mean_val,
        'std': std_val
    }

def moving_average_detection_page():
    st.header("📈 移動平均異常偵測")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 當前 KPI: **{st.session_state.selected_kpi}**")
    
    st.subheader("⚙️ 移動平均參數設定")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ma_method = st.selectbox(
            "移動平均方法:",
            ["簡單移動平均 (SMA)", "指數移動平均 (EMA)", "滾動標準差偵測"]
        )
    
    with col2:
        window_size = st.slider(
            "窗口大小 (天):", 
            min_value=3, max_value=90, value=30, step=1
        )
    
    with col3:
        if ma_method == "滾動標準差偵測":
            threshold = st.slider(
                "標準差倍數閾值:", 
                min_value=1.0, max_value=5.0, value=2.0, step=0.1
            )
        else:
            threshold = st.slider(
                "偏離閾值 (%):", 
                min_value=5.0, max_value=50.0, value=15.0, step=1.0
            )
    
    # 預設使用當前選擇的 KPI，但允許切換
    current_kpi_index = available_kpis.index(st.session_state.selected_kpi) if st.session_state.selected_kpi in available_kpis else 0
    selected_kpi = st.selectbox(
        "選擇要分析的 KPI:",
        options=available_kpis,
        index=to_plotly_list(current_kpi_index
    ))
    
    # 更新 session state
    if selected_kpi != st.session_state.selected_kpi:
        st.session_state.selected_kpi = selected_kpi
    
    if st.button("🔍 執行移動平均偵測"):
        # 轉換資料格式
        kpi_data = fab_data[fab_data['KPI'] == selected_kpi].copy()
        kpi_data = kpi_data.sort_values('REPORT_TIME')
        
        if len(kpi_data) == 0:
            st.error("❌ 所選 KPI 無資料")
            return
            
        outliers_info = detect_moving_average_outliers_fab(
            kpi_data, ma_method, window_size, threshold
        )
        
        st.subheader("📈 時序圖與移動平均線")
        
        # Create visualization
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(kpi_data['VALUE']),
            mode='lines+markers',
            name='原始數據',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(outliers_info['moving_avg']),
            mode='lines',
            name=f'{ma_method} (窗口={window_size})',
            line=dict(color='green', width=2)
        ))
        
        # Add upper and lower bounds
        if 'upper_bound' in outliers_info:
            fig.add_trace(go.Scatter(
                x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(outliers_info['upper_bound']),
                mode='lines',
                name='上界',
                line=dict(color='orange', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(outliers_info['lower_bound']),
                mode='lines',
                name='下界',
                line=dict(color='orange', dash='dash')
            ))
        
        # Add outliers
        if len(outliers_info['outlier_indices']) > 0:
            outlier_dates = kpi_data.iloc[outliers_info['outlier_indices']]['REPORT_TIME']
            outlier_values = kpi_data.iloc[outliers_info['outlier_indices']]['VALUE']
            
            fig.add_trace(go.Scatter(
                x=to_plotly_list(outlier_dates), y=to_plotly_list(outlier_values),
                mode='markers',
                name='異常點',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(
            title=f"{selected_fab} - {selected_kpi} - {ma_method} 異常偵測",
            xaxis_title="報告時間",
            yaxis_title="數值",
            hovermode='x unified',
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("總數據點", len(kpi_data))
        
        with col2:
            st.metric("異常點數量", len(outliers_info['outlier_indices']))
        
        with col3:
            anomaly_rate = len(outliers_info['outlier_indices']) / len(kpi_data) * 100
            st.metric("異常率", f"{anomaly_rate:.2f}%")
        
        # Show outlier details
        if len(outliers_info['outlier_indices']) > 0:
            st.subheader("🔍 異常點詳細資訊")
            
            outlier_df = kpi_data.iloc[outliers_info['outlier_indices']].copy()
            outlier_df['移動平均值'] = outliers_info['moving_avg'][outliers_info['outlier_indices']]
            outlier_df['偏離程度'] = outliers_info['deviations'][outliers_info['outlier_indices']]
            outlier_df = outlier_df.sort_values('偏離程度', ascending=False)
            
            st.dataframe(outlier_df[['REPORT_TIME', 'VALUE', '移動平均值', '偏離程度']])
        else:
            st.success("✅ 未發現異常點")

def detect_moving_average_outliers_fab(kpi_data, method, window_size, threshold):
    """適用於 FAB 資料結構的移動平均異常偵測"""
    data = kpi_data['VALUE'].values
    
    if method == "簡單移動平均 (SMA)":
        moving_avg = pd.Series(data).rolling(window=window_size, center=False).mean().values
        deviations = np.abs((data - moving_avg) / moving_avg) * 100
        outlier_mask = deviations > threshold
        
        # Calculate bounds for visualization
        upper_bound = moving_avg * (1 + threshold/100)
        lower_bound = moving_avg * (1 - threshold/100)
        
    elif method == "指數移動平均 (EMA)":
        ema_series = pd.Series(data).ewm(span=window_size).mean()
        moving_avg = ema_series.values
        deviations = np.abs((data - moving_avg) / moving_avg) * 100
        outlier_mask = deviations > threshold
        
        # Calculate bounds for visualization
        upper_bound = moving_avg * (1 + threshold/100)
        lower_bound = moving_avg * (1 - threshold/100)
        
    elif method == "滾動標準差偵測":
        rolling_mean = pd.Series(data).rolling(window=window_size, center=False).mean()
        rolling_std = pd.Series(data).rolling(window=window_size, center=False).std()
        
        moving_avg = rolling_mean.values
        upper_bound = (rolling_mean + threshold * rolling_std).values
        lower_bound = (rolling_mean - threshold * rolling_std).values
        
        outlier_mask = (data > upper_bound) | (data < lower_bound)
        deviations = np.maximum(
            (data - upper_bound) / rolling_std.values,
            (lower_bound - data) / rolling_std.values
        )
        deviations = np.nan_to_num(deviations, 0)
    
    # Remove NaN values for first few points
    valid_mask = ~np.isnan(moving_avg)
    outlier_mask = outlier_mask & valid_mask
    
    outlier_indices = np.where(outlier_mask)[0]
    
    result = {
        'outlier_indices': outlier_indices,
        'deviations': deviations,
        'moving_avg': moving_avg
    }
    
    if 'upper_bound' in locals():
        result['upper_bound'] = upper_bound
        result['lower_bound'] = lower_bound
    
    return result

def detect_moving_average_outliers(df, kpi_column, date_column, method, window_size, threshold):
    data = df[kpi_column].values
    
    if method == "簡單移動平均 (SMA)":
        moving_avg = pd.Series(data).rolling(window=window_size, center=False).mean().values
        deviations = np.abs((data - moving_avg) / moving_avg) * 100
        outlier_mask = deviations > threshold
        
        # Calculate bounds for visualization
        upper_bound = moving_avg * (1 + threshold/100)
        lower_bound = moving_avg * (1 - threshold/100)
        
    elif method == "指數移動平均 (EMA)":
        ema_series = pd.Series(data).ewm(span=window_size).mean()
        moving_avg = ema_series.values
        deviations = np.abs((data - moving_avg) / moving_avg) * 100
        outlier_mask = deviations > threshold
        
        # Calculate bounds for visualization
        upper_bound = moving_avg * (1 + threshold/100)
        lower_bound = moving_avg * (1 - threshold/100)
        
    elif method == "滾動標準差偵測":
        rolling_mean = pd.Series(data).rolling(window=window_size, center=False).mean()
        rolling_std = pd.Series(data).rolling(window=window_size, center=False).std()
        
        moving_avg = rolling_mean.values
        upper_bound = (rolling_mean + threshold * rolling_std).values
        lower_bound = (rolling_mean - threshold * rolling_std).values
        
        outlier_mask = (data > upper_bound) | (data < lower_bound)
        deviations = np.maximum(
            (data - upper_bound) / rolling_std.values,
            (lower_bound - data) / rolling_std.values
        )
        deviations = np.nan_to_num(deviations, 0)
    
    # Remove NaN values for first few points
    valid_mask = ~np.isnan(moving_avg)
    outlier_mask = outlier_mask & valid_mask
    
    outlier_indices = np.where(outlier_mask)[0]
    
    result = {
        'outlier_indices': outlier_indices,
        'deviations': deviations,
        'moving_avg': moving_avg
    }
    
    if 'upper_bound' in locals():
        result['upper_bound'] = upper_bound
        result['lower_bound'] = lower_bound
    
    return result

def seasonal_decomposition_page():
    st.header("🔄 季節性分解異常偵測")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 當前 KPI: **{st.session_state.selected_kpi}**")
    
    st.subheader("⚙️ 季節性分解參數設定")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        decomposition_model = st.selectbox(
            "分解模型:",
            ["additive", "multiplicative"],
            format_func=lambda x: "加法模型" if x == "additive" else "乘法模型"
        )
    
    with col2:
        seasonal_period = st.selectbox(
            "季節週期:",
            [7, 30, 90, 365],
            index=to_plotly_list(0),
            format_func=lambda x: f"{x} 天"
        )
    
    with col3:
        threshold = st.slider(
            "殘差異常閾值 (標準差倍數):", 
            min_value=1.0, max_value=5.0, value=2.5, step=0.1
        )
    
    # 預設使用當前選擇的 KPI，但允許切換
    current_kpi_index = available_kpis.index(st.session_state.selected_kpi) if st.session_state.selected_kpi in available_kpis else 0
    selected_kpi = st.selectbox(
        "選擇要分析的 KPI:",
        options=available_kpis,
        index=to_plotly_list(current_kpi_index
    ))
    
    # 更新 session state
    if selected_kpi != st.session_state.selected_kpi:
        st.session_state.selected_kpi = selected_kpi
    
    if st.button("🔍 執行季節性分解偵測"):
        try:
            # 轉換資料格式
            kpi_data = fab_data[fab_data['KPI'] == selected_kpi].copy()
            kpi_data = kpi_data.sort_values('REPORT_TIME')
            
            if len(kpi_data) == 0:
                st.error("❌ 所選 KPI 無資料")
                return
                
            decomp_result = perform_seasonal_decomposition_fab(
                kpi_data, decomposition_model, seasonal_period, threshold
            )
            
            if decomp_result is None:
                st.error("❌ 數據點不足以進行季節性分解，至少需要兩個完整的季節週期")
                return
            
            # Show decomposition components
            st.subheader("📊 季節性分解結果")
            
            # Create subplots for decomposition
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('原始數據', '趨勢', '季節性', '殘差'),
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            # Original data
            fig.add_trace(
                go.Scatter(x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(kpi_data['VALUE']), 
                          mode='lines', name='原始數據', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(decomp_result['trend']), 
                          mode='lines', name='趨勢', line=dict(color='green')),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(decomp_result['seasonal']), 
                          mode='lines', name='季節性', line=dict(color='orange')),
                row=3, col=1
            )
            
            # Residuals with outliers
            fig.add_trace(
                go.Scatter(x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(decomp_result['resid']), 
                          mode='lines', name='殘差', line=dict(color='gray')),
                row=4, col=1
            )
            
            # Add outliers to residual plot
            if len(decomp_result['outlier_indices']) > 0:
                outlier_dates = kpi_data.iloc[decomp_result['outlier_indices']]['REPORT_TIME']
                outlier_residuals = decomp_result['resid'][decomp_result['outlier_indices']]
                
                fig.add_trace(
                    go.Scatter(x=to_plotly_list(outlier_dates), y=to_plotly_list(outlier_residuals),
                              mode='markers', name='異常點',
                              marker=dict(color='red', size=8, symbol='x')),
                    row=4, col=1
                )
            
            # Add threshold lines to residual plot
            residual_std = np.std(decomp_result['resid'])
            residual_mean = np.mean(decomp_result['resid'])
            
            fig.add_hline(y=to_plotly_list(residual_mean + threshold * residual_std), 
                         line_dash="dash", line_color="red", row=4, col=1)
            fig.add_hline(y=to_plotly_list(residual_mean - threshold * residual_std), 
                         line_dash="dash", line_color="red", row=4, col=1)
            
            fig.update_layout(height=800, showlegend=False)
            fig.update_xaxes(title_text="時間", row=4, col=1)
            
            st.plotly_chart(fig)
            
            # Show reconstructed vs original
            st.subheader("📈 重建數據與異常點")
            
            fig2 = go.Figure()
            
            # Add original data
            fig2.add_trace(go.Scatter(
                x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(kpi_data['VALUE']),
                mode='lines+markers',
                name='原始數據',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Add reconstructed data (trend + seasonal)
            reconstructed = decomp_result['trend'] + decomp_result['seasonal']
            fig2.add_trace(go.Scatter(
                x=to_plotly_list(kpi_data['REPORT_TIME']), y=to_plotly_list(reconstructed),
                mode='lines',
                name='重建數據 (趨勢+季節性)',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            # Add outliers
            if len(decomp_result['outlier_indices']) > 0:
                outlier_dates = kpi_data.iloc[decomp_result['outlier_indices']]['REPORT_TIME']
                outlier_values = kpi_data.iloc[decomp_result['outlier_indices']]['VALUE']
                
                fig2.add_trace(go.Scatter(
                    x=to_plotly_list(outlier_dates), y=to_plotly_list(outlier_values),
                    mode='markers',
                    name='異常點',
                    marker=dict(color='red', size=10, symbol='x')
                ))
            
            fig2.update_layout(
                title=f"{selected_fab} - {selected_kpi} - 季節性分解異常偵測",
                xaxis_title="報告時間",
                yaxis_title="數值",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig2)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("總數據點", len(kpi_data))
            
            with col2:
                st.metric("異常點數量", len(decomp_result['outlier_indices']))
            
            with col3:
                anomaly_rate = len(decomp_result['outlier_indices']) / len(kpi_data) * 100
                st.metric("異常率", f"{anomaly_rate:.2f}%")
            
            # Show outlier details
            if len(decomp_result['outlier_indices']) > 0:
                st.subheader("🔍 異常點詳細資訊")
                
                outlier_df = kpi_data.iloc[decomp_result['outlier_indices']].copy()
                outlier_df['趨勢值'] = decomp_result['trend'][decomp_result['outlier_indices']]
                outlier_df['季節性值'] = decomp_result['seasonal'][decomp_result['outlier_indices']]
                outlier_df['殘差值'] = decomp_result['resid'][decomp_result['outlier_indices']]
                outlier_df['重建值'] = decomp_result['trend'][decomp_result['outlier_indices']] + decomp_result['seasonal'][decomp_result['outlier_indices']]
                outlier_df = outlier_df.sort_values('殘差值', key=to_plotly_list(abs), ascending=False)
                
                st.dataframe(outlier_df[['REPORT_TIME', 'VALUE', '趨勢值', '季節性值', '殘差值', '重建值']])
            else:
                st.success("✅ 未發現異常點")
                
        except Exception as e:
            st.error(f"❌ 季節性分解過程中發生錯誤: {str(e)}")
            st.info("💡 建議：1) 確保數據有足夠的週期性 2) 調整季節週期參數 3) 檢查數據是否有缺失值")

def perform_seasonal_decomposition_fab(kpi_data, model, period, threshold):
    """適用於 FAB 資料結構的季節性分解"""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        st.error("❌ 需要安裝 statsmodels 庫: pip install statsmodels")
        return None
    
    data = kpi_data['VALUE'].values
    
    # Check if we have enough data points
    if len(data) < 2 * period:
        return None
    
    try:
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(data, model=model, period=period, extrapolate_trend='freq')
        
        # Extract components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        resid = decomposition.resid
        
        # Remove NaN values
        valid_mask = ~np.isnan(resid)
        resid_clean = resid[valid_mask]
        
        # Detect outliers in residuals
        residual_mean = np.mean(resid_clean)
        residual_std = np.std(resid_clean)
        
        outlier_mask = np.abs(resid - residual_mean) > threshold * residual_std
        outlier_mask = outlier_mask & valid_mask
        
        outlier_indices = np.where(outlier_mask)[0]
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'resid': resid,
            'outlier_indices': outlier_indices,
            'residual_mean': residual_mean,
            'residual_std': residual_std
        }
        
    except Exception as e:
        st.error(f"分解過程中發生錯誤: {str(e)}")
        return None

def perform_seasonal_decomposition(df, kpi_column, date_column, model, period, threshold):
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        st.error("❌ 需要安裝 statsmodels 庫: pip install statsmodels")
        return None
    
    data = df[kpi_column].values
    
    # Check if we have enough data points
    if len(data) < 2 * period:
        return None
    
    try:
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(data, model=model, period=period, extrapolate_trend='freq')
        
        # Extract components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        resid = decomposition.resid
        
        # Remove NaN values
        valid_mask = ~np.isnan(resid)
        resid_clean = resid[valid_mask]
        
        # Detect outliers in residuals
        residual_mean = np.mean(resid_clean)
        residual_std = np.std(resid_clean)
        
        outlier_mask = np.abs(resid - residual_mean) > threshold * residual_std
        outlier_mask = outlier_mask & valid_mask
        
        outlier_indices = np.where(outlier_mask)[0]
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'resid': resid,
            'outlier_indices': outlier_indices,
            'residual_mean': residual_mean,
            'residual_std': residual_std
        }
        
    except Exception as e:
        st.error(f"分解過程中發生錯誤: {str(e)}")
        return None

if __name__ == "__main__":
    main()