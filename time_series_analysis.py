import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def time_series_analysis_page():
    """時序分析頁面"""
    st.header("📈 時序分析")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 當前 KPI: **{st.session_state.selected_kpi}**")
    
    st.subheader(f"🏭 {selected_fab} - 時序分析")
    
    # 選擇分析方法
    analysis_type = st.selectbox(
        "選擇分析方法:",
        ["趨勢分析", "週期性分析", "自相關分析", "變點檢測", "時序分解", "異常模式分析"]
    )
    
    # 選擇 KPI
    if analysis_type in ["趨勢分析", "週期性分析", "自相關分析", "變點檢測", "時序分解"]:
        selected_kpi = st.selectbox(
            "選擇要分析的 KPI:",
            options=available_kpis
        )
        
        if st.button("🔍 執行分析"):
            kpi_data = fab_data[fab_data['KPI'] == selected_kpi].copy()
            kpi_data = kpi_data.sort_values('REPORT_TIME')
            
            if len(kpi_data) == 0:
                st.error("❌ 所選 KPI 無資料")
                return
            
            if analysis_type == "趨勢分析":
                trend_analysis(kpi_data, selected_kpi, selected_fab)
            elif analysis_type == "週期性分析":
                periodicity_analysis(kpi_data, selected_kpi, selected_fab)
            elif analysis_type == "自相關分析":
                autocorrelation_analysis(kpi_data, selected_kpi, selected_fab)
            elif analysis_type == "變點檢測":
                changepoint_detection(kpi_data, selected_kpi, selected_fab)
            elif analysis_type == "時序分解":
                time_series_decomposition(kpi_data, selected_kpi, selected_fab)
    
    elif analysis_type == "異常模式分析":
        selected_kpis = st.multiselect(
            "選擇要分析的 KPI (最多8個):",
            options=available_kpis,
            default=available_kpis[:min(4, len(available_kpis))],
                    )
        
        if st.button("🔍 執行分析"):
            if len(selected_kpis) < 2:
                st.warning("⚠️ 異常模式分析至少需要選擇2個KPI")
                return
            
            anomaly_pattern_analysis(fab_data, selected_kpis, selected_fab)

def trend_analysis(kpi_data: pd.DataFrame, kpi_name: str, fab_name: str):
    """趨勢分析"""
    st.subheader("📊 趨勢分析結果")
    
    values = kpi_data['VALUE'].values
    dates = kpi_data['REPORT_TIME'].values
    
    # 線性趨勢
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    trend_line = slope * x + intercept
    
    # 趨勢統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        trend_direction = "上升" if slope > 0 else "下降" if slope < 0 else "穩定"
        st.metric("趨勢方向", trend_direction)
    
    with col2:
        st.metric("斜率", f"{slope:.4f}")
    
    with col3:
        st.metric("相關係數", f"{r_value:.3f}")
    
    with col4:
        significance = "顯著" if p_value < 0.05 else "不顯著"
        st.metric("趨勢顯著性", significance)
    
    # 趨勢圖
    fig = go.Figure()
    
    # 原始數據
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name='原始數據',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # 趨勢線
    fig.add_trace(go.Scatter(
        x=dates,
        y=trend_line,
        mode='lines',
        name=f'線性趨勢 (斜率={slope:.4f})',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # 移動平均
    window_sizes = [7, 30, 90]
    colors = ['green', 'orange', 'purple']
    
    for window, color in zip(window_sizes, colors):
        if len(values) >= window:
            ma = pd.Series(values).rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=dates,
                y=ma,
                mode='lines',
                name=f'{window}日移動平均',
                line=dict(color=color, width=1.5)
            ))
    
    fig.update_layout(
        title=f"{fab_name} - {kpi_name} 趨勢分析",
        xaxis_title="時間",
        yaxis_title="數值",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig)
    
    # 趨勢變化率分析
    st.subheader("📈 趨勢變化率分析")
    
    # 計算不同時間窗口的變化率
    periods = [7, 30, 90, 180]
    change_rates = []
    
    for period in periods:
        if len(values) >= period:
            recent_avg = np.mean(values[-period:])
            previous_avg = np.mean(values[-2*period:-period]) if len(values) >= 2*period else np.mean(values[:-period])
            change_rate = ((recent_avg - previous_avg) / previous_avg) * 100
            change_rates.append(change_rate)
        else:
            change_rates.append(None)
    
    change_df = pd.DataFrame({
        '時間窗口': [f'近{p}天' for p in periods],
        '變化率(%)': [f"{rate:.2f}%" if rate is not None else "N/A" for rate in change_rates]
    })
    
    st.dataframe(change_df)

def periodicity_analysis(kpi_data: pd.DataFrame, kpi_name: str, fab_name: str):
    """週期性分析"""
    st.subheader("🔄 週期性分析結果")
    
    values = kpi_data['VALUE'].values
    dates = kpi_data['REPORT_TIME'].values
    
    if len(values) < 14:
        st.warning("⚠️ 數據點太少，無法進行週期性分析")
        return
    
    # FFT 分析
    fft = np.fft.fft(values)
    freqs = np.fft.fftfreq(len(values))
    
    # 找出主要週期
    power = np.abs(fft) ** 2
    main_freq_idx = np.argsort(power[1:len(power)//2])[-5:] + 1  # 前5個主要頻率
    main_periods = [1/freqs[idx] for idx in main_freq_idx if freqs[idx] > 0]
    
    # 週期性指標
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**檢測到的主要週期:**")
        for i, period in enumerate(main_periods[:3]):
            if period <= len(values):
                st.write(f"{i+1}. {period:.1f} 天")
    
    with col2:
        # 週間模式分析
        kpi_data['day_of_week'] = pd.to_datetime(kpi_data['REPORT_TIME']).dt.day_name()
        weekly_stats = kpi_data.groupby('day_of_week')['VALUE'].agg(['mean', 'std']).round(2)
        st.write("**週間模式統計:**")
        st.dataframe(weekly_stats)
    
    # 視覺化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('原始時序', '頻譜分析', '週間模式', '自相關函數'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 原始時序
    fig.add_trace(
        go.Scatter(x=dates, y=values, mode='lines', name='原始數據'),
        row=1, col=1
    )
    
    # 頻譜
    fig.add_trace(
        go.Scatter(x=freqs[1:len(freqs)//2], y=power[1:len(power)//2], 
                   mode='lines', name='功率譜'),
        row=1, col=2
    )
    
    # 週間模式
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_means = [weekly_stats.loc[day, 'mean'] if day in weekly_stats.index else 0 for day in day_order]
    
    fig.add_trace(
        go.Bar(x=day_order, y=weekly_means, name='週間平均'),
        row=2, col=1
    )
    
    # 自相關
    autocorr = [np.corrcoef(values[:-i], values[i:])[0,1] for i in range(1, min(50, len(values)//2))]
    fig.add_trace(
        go.Scatter(x=list(range(1, len(autocorr)+1)), y=autocorr, 
                   mode='lines+markers', name='自相關'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig)

def autocorrelation_analysis(kpi_data: pd.DataFrame, kpi_name: str, fab_name: str):
    """自相關分析"""
    st.subheader("🔗 自相關分析結果")
    
    values = kpi_data['VALUE'].values
    
    if len(values) < 20:
        st.warning("⚠️ 數據點太少，無法進行自相關分析")
        return
    
    # 計算自相關和偏自相關
    max_lags = min(40, len(values) // 4)
    lags = range(1, max_lags + 1)
    
    # 自相關
    autocorr = [np.corrcoef(values[:-lag], values[lag:])[0,1] for lag in lags]
    
    # 找出顯著的自相關
    significant_lags = [lag for lag, corr in zip(lags, autocorr) if abs(corr) > 0.2]
    
    # 統計信息
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_autocorr = max(autocorr)
        max_lag = lags[autocorr.index(max_autocorr)]
        st.metric("最大自相關", f"{max_autocorr:.3f}", f"延遲{max_lag}天")
    
    with col2:
        st.metric("顯著自相關數量", len(significant_lags))
    
    with col3:
        persistence = sum(1 for corr in autocorr[:7] if corr > 0.1)  # 前7天的持續性
        st.metric("短期持續性", f"{persistence}/7天")
    
    # 視覺化
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('自相關函數', '顯著性檢驗'),
        vertical_spacing=0.1
    )
    
    # 自相關函數
    fig.add_trace(
        go.Scatter(x=list(lags), y=autocorr, 
                   mode='lines+markers', name='自相關'),
        row=1, col=1
    )
    
    # 添加置信區間
    confidence_level = 1.96 / np.sqrt(len(values))
    fig.add_hline(y=confidence_level, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-confidence_level, line_dash="dash", line_color="red", row=1, col=1)
    
    # 顯著性條形圖
    colors = ['red' if abs(corr) > confidence_level else 'blue' for corr in autocorr]
    fig.add_trace(
        go.Bar(x=list(lags), y=autocorr, marker_color=colors, name='顯著性'),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig)
    
    # 顯著滯後分析
    if significant_lags:
        st.subheader("📋 顯著自相關滯後")
        sig_df = pd.DataFrame({
            '滯後天數': significant_lags,
            '自相關係數': [autocorr[lag-1] for lag in significant_lags],
            '可能原因': [get_lag_interpretation(lag) for lag in significant_lags]
        })
        st.dataframe(sig_df)

def get_lag_interpretation(lag: int) -> str:
    """解釋滯後的可能原因"""
    if lag == 1:
        return "強烈的日間依賴性"
    elif 2 <= lag <= 3:
        return "短期記憶效應"
    elif 6 <= lag <= 8:
        return "可能的週週期性"
    elif 13 <= lag <= 15:
        return "雙週週期性"
    elif 28 <= lag <= 32:
        return "月週期性"
    elif 88 <= lag <= 92:
        return "季度週期性"
    else:
        return f"{lag}天週期性"

def changepoint_detection(kpi_data: pd.DataFrame, kpi_name: str, fab_name: str):
    """變點檢測"""
    st.subheader("⚡ 變點檢測結果")
    
    values = kpi_data['VALUE'].values
    dates = kpi_data['REPORT_TIME'].values
    
    if len(values) < 20:
        st.warning("⚠️ 數據點太少，無法進行變點檢測")
        return
    
    # 簡單的變點檢測算法（基於統計變化）
    changepoints = detect_changepoints(values)
    
    # 統計信息
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("檢測到變點數量", len(changepoints))
    
    with col2:
        if changepoints:
            last_change = max(changepoints)
            days_since = len(values) - last_change
            st.metric("距離最近變點", f"{days_since}天")
        else:
            st.metric("距離最近變點", "無變點")
    
    with col3:
        stability_score = calculate_stability_score(values, changepoints)
        st.metric("穩定性評分", f"{stability_score:.2f}")
    
    # 視覺化
    fig = go.Figure()
    
    # 原始數據
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name='原始數據',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # 標記變點
    if changepoints:
        for cp in changepoints:
            fig.add_vline(
                x=dates[cp],
                line_dash="dash",
                line_color="red",
                annotation_text=f"變點 {cp}"
            )
    
    # 添加分段平均線
    if changepoints:
        segments = [0] + changepoints + [len(values)]
        colors = ['green', 'orange', 'purple', 'brown', 'pink']
        
        for i in range(len(segments) - 1):
            start, end = segments[i], segments[i + 1]
            segment_mean = np.mean(values[start:end])
            
            fig.add_trace(go.Scatter(
                x=dates[start:end],
                y=[segment_mean] * (end - start),
                mode='lines',
                name=f'段 {i+1} 平均',
                line=dict(color=colors[i % len(colors)], width=3)
            ))
    
    fig.update_layout(
        title=f"{fab_name} - {kpi_name} 變點檢測",
        xaxis_title="時間",
        yaxis_title="數值",
        height=500
    )
    
    st.plotly_chart(fig)
    
    # 變點詳情
    if changepoints:
        st.subheader("📋 變點詳細信息")
        
        cp_details = []
        for i, cp in enumerate(changepoints):
            before_mean = np.mean(values[max(0, cp-10):cp])
            after_mean = np.mean(values[cp:min(len(values), cp+10)])
            change_magnitude = abs(after_mean - before_mean)
            change_direction = "上升" if after_mean > before_mean else "下降"
            
            cp_details.append({
                '變點位置': pd.to_datetime(dates[cp]).strftime('%Y-%m-%d'),
                '變化方向': change_direction,
                '變化幅度': f"{change_magnitude:.2f}",
                '變化前平均': f"{before_mean:.2f}",
                '變化後平均': f"{after_mean:.2f}"
            })
        
        st.dataframe(pd.DataFrame(cp_details))

def detect_changepoints(values: np.ndarray, min_size: int = 5) -> List[int]:
    """簡單的變點檢測算法"""
    changepoints = []
    n = len(values)
    
    for i in range(min_size, n - min_size):
        # 計算前後窗口的統計差異
        before = values[max(0, i-min_size):i]
        after = values[i:min(i+min_size, n)]
        
        if len(before) >= min_size and len(after) >= min_size:
            # 使用 t 檢驗檢測均值變化
            t_stat, p_value = stats.ttest_ind(before, after)
            
            if p_value < 0.01:  # 顯著性水準
                changepoints.append(i)
    
    # 合併相近的變點
    if changepoints:
        merged_changepoints = [changepoints[0]]
        for cp in changepoints[1:]:
            if cp - merged_changepoints[-1] > min_size:
                merged_changepoints.append(cp)
        return merged_changepoints
    
    return changepoints

def calculate_stability_score(values: np.ndarray, changepoints: List[int]) -> float:
    """計算穩定性評分 (0-1, 1表示最穩定)"""
    # 基於變點數量和波動性計算
    volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 1
    changepoint_penalty = len(changepoints) / len(values)
    
    stability = max(0, 1 - volatility - changepoint_penalty)
    return min(1, stability)

def time_series_decomposition(kpi_data: pd.DataFrame, kpi_name: str, fab_name: str):
    """時序分解"""
    st.subheader("🔄 時序分解結果")
    
    values = kpi_data['VALUE'].values
    dates = kpi_data['REPORT_TIME'].values
    
    if len(values) < 30:
        st.warning("⚠️ 數據點太少，無法進行時序分解")
        return
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # 執行分解
        decomposition = seasonal_decompose(values, model='additive', period=30, extrapolate_trend='freq')
        
        # 創建子圖
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('原始數據', '趨勢', '季節性', '殘差'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # 原始數據
        fig.add_trace(
            go.Scatter(x=dates, y=values, mode='lines', name='原始', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 趨勢
        fig.add_trace(
            go.Scatter(x=dates, y=decomposition.trend, mode='lines', name='趨勢', line=dict(color='green')),
            row=2, col=1
        )
        
        # 季節性
        fig.add_trace(
            go.Scatter(x=dates, y=decomposition.seasonal, mode='lines', name='季節性', line=dict(color='orange')),
            row=3, col=1
        )
        
        # 殘差
        fig.add_trace(
            go.Scatter(x=dates, y=decomposition.resid, mode='lines', name='殘差', line=dict(color='red')),
            row=4, col=1
        )
        
        fig.update_layout(height=800, showlegend=False, title_text=f"{fab_name} - {kpi_name} 時序分解")
        st.plotly_chart(fig)
        
        # 分解統計
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_strength = 1 - np.var(decomposition.resid) / np.var(values - decomposition.seasonal)
            st.metric("趨勢強度", f"{max(0, trend_strength):.3f}")
        
        with col2:
            seasonal_strength = 1 - np.var(decomposition.resid) / np.var(values - decomposition.trend)
            st.metric("季節性強度", f"{max(0, seasonal_strength):.3f}")
        
        with col3:
            residual_std = np.std(decomposition.resid)
            st.metric("殘差標準差", f"{residual_std:.3f}")
        
    except ImportError:
        st.error("❌ 需要安裝 statsmodels: pip install statsmodels")

def anomaly_pattern_analysis(fab_data: pd.DataFrame, selected_kpis: List[str], fab_name: str):
    """異常模式分析"""
    st.subheader("🔍 異常模式分析結果")
    
    # 準備數據
    pivot_data = fab_data[fab_data['KPI'].isin(selected_kpis)].pivot_table(
        index='REPORT_TIME', columns='KPI', values='VALUE', aggfunc='mean'
    ).fillna(method='ffill').fillna(method='bfill')
    
    if pivot_data.empty:
        st.error("❌ 無法建立數據矩陣")
        return
    
    # 標準化數據
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_data.values)
    
    # PCA 分析
    pca = PCA(n_components=min(3, len(selected_kpis)))
    pca_result = pca.fit_transform(scaled_data)
    
    # 異常檢測（基於馬氏距離）
    from scipy.spatial.distance import mahalanobis
    
    cov_matrix = np.cov(scaled_data.T)
    mean_vector = np.mean(scaled_data, axis=0)
    
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        mahalanobis_distances = [
            mahalanobis(row, mean_vector, inv_cov_matrix) for row in scaled_data
        ]
    except:
        # 如果協方差矩陣不可逆，使用歐氏距離
        mahalanobis_distances = [
            np.linalg.norm(row - mean_vector) for row in scaled_data
        ]
    
    # 識別異常點
    threshold = np.percentile(mahalanobis_distances, 95)  # 前5%作為異常
    anomaly_indices = np.where(np.array(mahalanobis_distances) > threshold)[0]
    
    # 結果展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("分析 KPI 數量", len(selected_kpis))
    
    with col2:
        st.metric("異常時間點", len(anomaly_indices))
    
    with col3:
        anomaly_rate = len(anomaly_indices) / len(pivot_data) * 100
        st.metric("異常比例", f"{anomaly_rate:.2f}%")
    
    # PCA 可視化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PCA 前兩個主成分', 'PCA 貢獻率', '異常分數時序', 'KPI 相關性熱圖'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # PCA 散點圖
    colors = ['red' if i in anomaly_indices else 'blue' for i in range(len(pca_result))]
    fig.add_trace(
        go.Scatter(x=pca_result[:, 0], y=pca_result[:, 1], 
                   mode='markers', marker=dict(color=colors),
                   name='數據點'),
        row=1, col=1
    )
    
    # PCA 貢獻率
    fig.add_trace(
        go.Bar(x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
               y=pca.explained_variance_ratio_, name='貢獻率'),
        row=1, col=2
    )
    
    # 異常分數時序
    fig.add_trace(
        go.Scatter(x=pivot_data.index, y=mahalanobis_distances,
                   mode='lines+markers', name='異常分數'),
        row=2, col=1
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=2, col=1)
    
    # 相關性熱圖
    corr_matrix = pivot_data.corr()
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                   colorscale='RdYlBu', zmid=0),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig)
    
    # 異常時間點詳情
    if len(anomaly_indices) > 0:
        st.subheader("📋 異常時間點詳情")
        
        anomaly_details = []
        for idx in anomaly_indices[:10]:  # 只顯示前10個
            date = pivot_data.index[idx]
            anomaly_score = mahalanobis_distances[idx]
            
            # 找出該時間點最異常的KPI
            row_data = pivot_data.iloc[idx]
            z_scores = (row_data - pivot_data.mean()) / pivot_data.std()
            most_anomalous_kpi = z_scores.abs().idxmax()
            
            anomaly_details.append({
                '時間': pd.to_datetime(date).strftime('%Y-%m-%d'),
                '異常分數': f"{anomaly_score:.3f}",
                '最異常KPI': most_anomalous_kpi,
                'Z-Score': f"{z_scores[most_anomalous_kpi]:.3f}"
            })
        
        st.dataframe(pd.DataFrame(anomaly_details))

# 新增其他頁面的佔位函數
def anomaly_trend_analysis_page():
    """異常趨勢分析頁面"""
    st.header("📊 異常趨勢分析")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        return
    
    st.info("🔧 異常趨勢分析功能開發中...")
    # 這裡可以實現更多功能，如：
    # - 異常頻率趨勢
    # - 異常類型分類
    # - 異常持續時間分析
    # - 異常影響範圍分析