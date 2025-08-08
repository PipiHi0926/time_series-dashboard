import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple

def to_plotly_list(data):
    """將任何數據格式轉換為 Plotly 5.6.0 相容的 Python list"""
    if data is None:
        return []
    if hasattr(data, 'values'):
        return data.values.tolist()
    elif hasattr(data, 'tolist'):
        return data.tolist() 
    elif hasattr(data, '__iter__') and not isinstance(data, str):
        return list(data)
    else:
        return [data]

def batch_kpi_monitoring_page():
    """KPI 批量監控頁面"""
    st.header("📊 KPI 批量監控")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 可用 KPI: **{len(available_kpis)}** 個")
    
    st.subheader(f"🏭 {selected_fab} - 批量 KPI 異常監控")
    
    # 設定參數
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_method = st.selectbox(
            "偵測方法:",
            ["Z-Score", "IQR", "移動平均", "組合方法"]
        )
    
    with col2:
        if detection_method in ["Z-Score", "組合方法"]:
            threshold = st.slider("Z-Score 閾值:", 1.0, 5.0, 2.0, 0.1)
        elif detection_method == "IQR":
            threshold = st.slider("IQR 倍數:", 1.0, 3.0, 1.5, 0.1)
        else:
            threshold = st.slider("移動平均偏離%:", 5.0, 50.0, 15.0, 1.0)
    
    with col3:
        if detection_method in ["移動平均", "組合方法"]:
            window_size = st.slider("移動平均窗口:", 7, 60, 30, 1)
        else:
            window_size = 30
    
    # KPI 選擇
    selected_kpis = st.multiselect(
        "選擇要監控的 KPI:",
        options=available_kpis,
        default=available_kpis,
        help="選擇要進行批量監控的 KPI"
    )
    
    if not selected_kpis:
        st.warning("⚠️ 請至少選擇一個 KPI")
        return
    
    if st.button("🔍 執行批量監控"):
        # 執行批量監控
        monitoring_results = perform_batch_monitoring(
            fab_data, selected_kpis, detection_method, threshold, window_size
        )
        
        # 顯示總覽
        display_monitoring_overview(monitoring_results, selected_fab)
        
        # 顯示詳細結果
        display_detailed_results(monitoring_results, fab_data)
        
        # 顯示異常排名
        display_anomaly_ranking(monitoring_results)

def perform_batch_monitoring(fab_data: pd.DataFrame, kpis: List[str], 
                           method: str, threshold: float, window_size: int) -> Dict:
    """執行批量監控"""
    results = {}
    
    for kpi in kpis:
        kpi_data = fab_data[fab_data['KPI'] == kpi].copy()
        kpi_data = kpi_data.sort_values('REPORT_TIME')
        
        if len(kpi_data) < 10:  # 數據太少跳過
            continue
            
        values = kpi_data['VALUE'].values
        dates = kpi_data['REPORT_TIME'].values
        
        if method == "Z-Score":
            outliers, scores = detect_zscore_outliers(values, threshold)
        elif method == "IQR":
            outliers, scores = detect_iqr_outliers(values, threshold)
        elif method == "移動平均":
            outliers, scores = detect_ma_outliers(values, window_size, threshold)
        elif method == "組合方法":
            outliers, scores = detect_combined_outliers(values, threshold, window_size)
        
        results[kpi] = {
            'dates': dates,
            'values': values,
            'outliers': outliers,
            'scores': scores,
            'anomaly_count': len(outliers),
            'anomaly_rate': len(outliers) / len(values) * 100,
            'latest_value': values[-1] if len(values) > 0 else None,
            'trend': calculate_trend(values),
            'volatility': np.std(values)
        }
    
    return results

def detect_zscore_outliers(values: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Z-Score 異常偵測"""
    mean_val = np.mean(values)
    std_val = np.std(values)
    z_scores = np.abs((values - mean_val) / std_val)
    outliers = np.where(z_scores > threshold)[0]
    return outliers, z_scores

def detect_iqr_outliers(values: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """IQR 異常偵測"""
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = np.where((values < lower_bound) | (values > upper_bound))[0]
    scores = np.maximum((Q1 - values) / IQR, (values - Q3) / IQR)
    return outliers, scores

def detect_ma_outliers(values: np.ndarray, window_size: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """移動平均異常偵測"""
    ma = pd.Series(values).rolling(window=window_size).mean().values
    deviations = np.abs((values - ma) / ma) * 100
    deviations = np.nan_to_num(deviations, 0)
    outliers = np.where(deviations > threshold)[0]
    return outliers, deviations

def detect_combined_outliers(values: np.ndarray, z_threshold: float, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """組合方法異常偵測"""
    # Z-Score
    z_outliers, z_scores = detect_zscore_outliers(values, z_threshold)
    
    # 移動平均
    ma_outliers, ma_scores = detect_ma_outliers(values, window_size, 15.0)
    
    # 組合結果
    combined_outliers = np.union1d(z_outliers, ma_outliers)
    combined_scores = np.maximum(z_scores / z_threshold, ma_scores / 15.0)
    
    return combined_outliers, combined_scores

def calculate_trend(values: np.ndarray) -> str:
    """計算趨勢"""
    if len(values) < 2:
        return "無趨勢"
    
    # 使用線性回歸計算趨勢
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    
    if slope > 0.01:
        return "上升"
    elif slope < -0.01:
        return "下降"
    else:
        return "穩定"

def display_monitoring_overview(results: Dict, fab_name: str):
    """顯示監控總覽"""
    st.subheader("📈 監控總覽")
    
    # 計算總體統計
    total_kpis = len(results)
    total_anomalies = sum(result['anomaly_count'] for result in results.values())
    avg_anomaly_rate = np.mean([result['anomaly_rate'] for result in results.values()])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("監控 KPI 數量", total_kpis)
    
    with col2:
        st.metric("總異常點數", total_anomalies)
    
    with col3:
        st.metric("平均異常率", f"{avg_anomaly_rate:.2f}%")
    
    with col4:
        high_risk_kpis = sum(1 for result in results.values() if result['anomaly_rate'] > 5)
        st.metric("高風險 KPI", high_risk_kpis)
    
    # 異常率分佈圖
    kpi_names = list(results.keys())
    anomaly_rates = [results[kpi]['anomaly_rate'] for kpi in kpi_names]
    
    fig = go.Figure(data=go.Bar(
        x=to_plotly_list(kpi_names), y=to_plotly_list(anomaly_rates),
        marker_color=['red' if rate > 5 else 'orange' if rate > 2 else 'green' 
                     for rate in anomaly_rates],
        text=[f"{rate:.1f}%" for rate in anomaly_rates],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"{fab_name} - KPI 異常率分佈",
        xaxis_title="KPI",
        yaxis_title="異常率 (%)",
        height=400
    )
    
    st.plotly_chart(fig)

def display_detailed_results(results: Dict, fab_data: pd.DataFrame):
    """顯示詳細結果"""
    st.subheader("🔍 詳細監控結果")
    
    # 建立多個子圖
    kpi_names = list(results.keys())
    n_kpis = len(kpi_names)
    
    if n_kpis <= 4:
        rows, cols = n_kpis, 1
    elif n_kpis <= 8:
        rows, cols = 4, 2
    else:
        rows, cols = 4, 3
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=kpi_names[:rows*cols],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    for i, kpi in enumerate(kpi_names[:rows*cols]):
        row = i // cols + 1
        col = i % cols + 1
        
        result = results[kpi]
        
        # 添加原始數據
        fig.add_trace(
            go.Scatter(
                x=to_plotly_list(result['dates']), y=to_plotly_list(result['values']),
                mode='lines+markers',
                name=f'{kpi}',
                line=dict(width=1),
                marker=dict(size=3),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # 添加異常點
        if len(result['outliers']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=to_plotly_list(result['dates'][result['outliers']]), y=to_plotly_list(result['values'][result['outliers']]),
                    mode='markers',
                    name=f'{kpi} 異常',
                    marker=dict(color='red', size=6, symbol='x'),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=200 * rows,
        title_text="KPI 時序圖表與異常點",
        showlegend=False
    )
    
    st.plotly_chart(fig)

def display_anomaly_ranking(results: Dict):
    """顯示異常排名"""
    st.subheader("🏆 KPI 異常風險排名")
    
    # 建立排名資料
    ranking_data = []
    for kpi, result in results.items():
        ranking_data.append({
            'KPI': kpi,
            '異常點數': result['anomaly_count'],
            '異常率': f"{result['anomaly_rate']:.2f}%",
            '最新值': f"{result['latest_value']:.2f}" if result['latest_value'] else "N/A",
            '趨勢': result['trend'],
            '波動性': f"{result['volatility']:.2f}",
            '風險等級': get_risk_level(result['anomaly_rate'])
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values('異常點數', ascending=False)
    
    # 設定顏色映射
    def color_risk_level(val):
        if val == "高風險":
            return 'background-color: #ffebee'
        elif val == "中風險":
            return 'background-color: #fff3e0'
        else:
            return 'background-color: #e8f5e8'
    
    styled_df = ranking_df.style.applymap(color_risk_level, subset=['風險等級'])
    
    st.dataframe(styled_df)
    
    # 風險分佈餅圖
    risk_counts = ranking_df['風險等級'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=to_plotly_list(risk_counts.index),
        values=to_plotly_list(risk_counts.values),
        marker_colors=['#f44336', '#ff9800', '#4caf50'],
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title="KPI 風險等級分佈",
        height=400
    )
    
    st.plotly_chart(fig)

def get_risk_level(anomaly_rate: float) -> str:
    """根據異常率判斷風險等級"""
    if anomaly_rate > 5:
        return "高風險"
    elif anomaly_rate > 2:
        return "中風險"
    else:
        return "低風險"