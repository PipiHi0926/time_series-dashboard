import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def level_shift_detection_page():
    """Level Shift 檢測頁面"""
    st.header("🔄 Level Shift 檢測分析")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 當前 KPI: **{st.session_state.selected_kpi}**")
    
    st.subheader(f"🏭 {selected_fab} - Level Shift 檢測")
    
    # 參數設定
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_kpi = st.selectbox(
            "選擇 KPI:",
            options=available_kpis,
            index=available_kpis.index(st.session_state.selected_kpi) if st.session_state.selected_kpi in available_kpis else 0
        )
    
    with col2:
        comparison_window = st.slider("比較窗口大小 (天):", 3, 30, 7, 1,
                                    help="前後比較的天數窗口")
    
    with col3:
        significance_level = st.slider("顯著性水準:", 0.01, 0.10, 0.05, 0.01,
                                     help="統計檢驗的顯著性水準")
    
    with col4:
        min_shift_magnitude = st.number_input("最小變化幅度 (%):", 0.0, 50.0, 5.0, 0.5,
                                            help="視為Level Shift的最小變化百分比")
    
    if st.button("🔍 執行 Level Shift 檢測", type="primary"):
        kpi_data = fab_data[fab_data['KPI'] == selected_kpi].copy()
        kpi_data = kpi_data.sort_values('REPORT_TIME')
        
        if len(kpi_data) < comparison_window * 2:
            st.error(f"❌ 數據點不足，至少需要 {comparison_window * 2} 個數據點")
            return
        
        # 執行 Level Shift 檢測
        shift_results = detect_level_shifts(
            kpi_data, comparison_window, significance_level, min_shift_magnitude
        )
        
        # 顯示結果
        display_level_shift_results(shift_results, kpi_data, selected_kpi, selected_fab)

def trend_momentum_analysis_page():
    """趨勢動量分析頁面"""
    st.header("📈 趨勢動量分析")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 當前 KPI: **{st.session_state.selected_kpi}**")
    
    st.subheader(f"🏭 {selected_fab} - 趨勢動量分析")
    
    # 參數設定
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_kpi = st.selectbox(
            "選擇 KPI:",
            options=available_kpis,
            index=available_kpis.index(st.session_state.selected_kpi) if st.session_state.selected_kpi in available_kpis else 0
        )
    
    with col2:
        short_window = st.slider("短期窗口 (天):", 3, 14, 7, 1,
                               help="短期趨勢計算窗口")
    
    with col3:
        long_window = st.slider("長期窗口 (天):", 14, 90, 28, 1,
                              help="長期趨勢計算窗口")
    
    with col4:
        momentum_threshold = st.slider("動量閾值:", 0.1, 2.0, 0.5, 0.1,
                                     help="判斷顯著趨勢變化的閾值")
    
    # 分析選項
    analysis_options = st.multiselect(
        "選擇分析方法:",
        ["趨勢動量分析", "連續趨勢檢測", "加速度分析", "趨勢強度評估"],
        default=["趨勢動量分析", "連續趨勢檢測"]
    )
    
    if st.button("🔍 執行趨勢動量分析", type="primary"):
        kpi_data = fab_data[fab_data['KPI'] == selected_kpi].copy()
        kpi_data = kpi_data.sort_values('REPORT_TIME')
        
        if len(kpi_data) < long_window + 5:
            st.error(f"❌ 數據點不足，至少需要 {long_window + 5} 個數據點")
            return
        
        # 執行趨勢動量分析
        momentum_results = analyze_trend_momentum(
            kpi_data, short_window, long_window, momentum_threshold, analysis_options
        )
        
        # 顯示結果
        display_momentum_results(momentum_results, kpi_data, selected_kpi, selected_fab)

def detect_level_shifts(kpi_data: pd.DataFrame, window: int, 
                       significance: float, min_magnitude: float) -> Dict:
    """檢測 Level Shift"""
    values = kpi_data['VALUE'].values
    dates = kpi_data['REPORT_TIME'].values
    
    shifts = []
    shift_explanations = []
    
    for i in range(window, len(values) - window):
        # 前後窗口數據
        before = values[i-window:i]
        after = values[i:i+window]
        
        # 統計檢驗
        t_stat, p_value = stats.ttest_ind(before, after)
        
        # 計算變化幅度
        before_mean = np.mean(before)
        after_mean = np.mean(after)
        change_pct = abs((after_mean - before_mean) / before_mean) * 100
        
        # 判斷是否為顯著 Level Shift
        if p_value < significance and change_pct >= min_magnitude:
            shift_type = "上升" if after_mean > before_mean else "下降"
            
            shifts.append({
                'index': i,
                'date': dates[i],
                'before_mean': before_mean,
                'after_mean': after_mean,
                'change_pct': change_pct,
                'change_direction': shift_type,
                't_stat': t_stat,
                'p_value': p_value,
                'before_std': np.std(before),
                'after_std': np.std(after)
            })
            
            # 生成解釋
            explanation = generate_level_shift_explanation(
                before_mean, after_mean, change_pct, shift_type, p_value, window
            )
            shift_explanations.append(explanation)
    
    return {
        'shifts': shifts,
        'explanations': shift_explanations,
        'values': values,
        'dates': dates,
        'window': window
    }

def analyze_trend_momentum(kpi_data: pd.DataFrame, short_window: int, 
                         long_window: int, threshold: float, 
                         analysis_options: List[str]) -> Dict:
    """分析趨勢動量"""
    values = kpi_data['VALUE'].values
    dates = kpi_data['REPORT_TIME'].values
    
    results = {
        'values': values,
        'dates': dates,
        'short_window': short_window,
        'long_window': long_window
    }
    
    if "趨勢動量分析" in analysis_options:
        momentum_data = calculate_trend_momentum(values, short_window, long_window)
        results['momentum'] = momentum_data
    
    if "連續趨勢檢測" in analysis_options:
        continuous_trends = detect_continuous_trends(values, dates, short_window, threshold)
        results['continuous_trends'] = continuous_trends
    
    if "加速度分析" in analysis_options:
        acceleration_data = calculate_trend_acceleration(values, short_window)
        results['acceleration'] = acceleration_data
    
    if "趨勢強度評估" in analysis_options:
        strength_data = evaluate_trend_strength(values, short_window, long_window)
        results['strength'] = strength_data
    
    return results

def calculate_trend_momentum(values: np.ndarray, short_window: int, long_window: int) -> Dict:
    """計算趨勢動量"""
    short_trends = []
    long_trends = []
    momentum_signals = []
    
    # 計算短期和長期趨勢
    for i in range(long_window, len(values)):
        # 短期趨勢 (斜率)
        short_data = values[i-short_window:i]
        short_x = np.arange(len(short_data))
        short_slope = np.polyfit(short_x, short_data, 1)[0]
        short_trends.append(short_slope)
        
        # 長期趨勢 (斜率)
        long_data = values[i-long_window:i]
        long_x = np.arange(len(long_data))
        long_slope = np.polyfit(long_x, long_data, 1)[0]
        long_trends.append(long_slope)
        
        # 動量信號 (短期趨勢 - 長期趨勢)
        momentum = short_slope - long_slope
        momentum_signals.append(momentum)
    
    return {
        'short_trends': np.array(short_trends),
        'long_trends': np.array(long_trends),
        'momentum_signals': np.array(momentum_signals),
        'start_index': long_window
    }

def detect_continuous_trends(values: np.ndarray, dates: np.ndarray, 
                           window: int, threshold: float) -> Dict:
    """檢測連續趨勢"""
    trends = []
    continuous_periods = []
    
    # 計算滑動趨勢
    for i in range(window, len(values)):
        data = values[i-window:i]
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        trends.append({
            'index': i,
            'slope': slope,
            'r_value': r_value,
            'p_value': p_value,
            'trend_strength': abs(slope) * abs(r_value)
        })
    
    # 識別連續趨勢期間
    current_trend = None
    trend_start = None
    
    for i, trend in enumerate(trends):
        if abs(trend['trend_strength']) > threshold and trend['p_value'] < 0.05:
            trend_direction = "上升" if trend['slope'] > 0 else "下降"
            
            if current_trend == trend_direction:
                continue  # 延續當前趨勢
            else:
                # 結束前一個趨勢期間
                if current_trend is not None and trend_start is not None:
                    continuous_periods.append({
                        'start_index': trend_start,
                        'end_index': i - 1,
                        'start_date': dates[trend_start + window],
                        'end_date': dates[i - 1 + window],
                        'direction': current_trend,
                        'duration': i - trend_start,
                        'avg_slope': np.mean([t['slope'] for t in trends[trend_start:i]])
                    })
                
                # 開始新的趨勢期間
                current_trend = trend_direction
                trend_start = i
        else:
            # 結束當前趨勢期間
            if current_trend is not None and trend_start is not None:
                continuous_periods.append({
                    'start_index': trend_start,
                    'end_index': i - 1,
                    'start_date': dates[trend_start + window],
                    'end_date': dates[i - 1 + window],
                    'direction': current_trend,
                    'duration': i - trend_start,
                    'avg_slope': np.mean([t['slope'] for t in trends[trend_start:i]])
                })
            current_trend = None
            trend_start = None
    
    return {
        'trends': trends,
        'continuous_periods': continuous_periods,
        'window': window
    }

def calculate_trend_acceleration(values: np.ndarray, window: int) -> Dict:
    """計算趋势加速度"""
    accelerations = []
    velocities = []
    
    # 計算一階差分 (速度)
    for i in range(window, len(values) - 1):
        velocity = values[i] - values[i-1]
        velocities.append(velocity)
    
    # 計算二階差分 (加速度)
    for i in range(1, len(velocities)):
        acceleration = velocities[i] - velocities[i-1]
        accelerations.append(acceleration)
    
    return {
        'velocities': np.array(velocities),
        'accelerations': np.array(accelerations),
        'start_index': window + 1
    }

def evaluate_trend_strength(values: np.ndarray, short_window: int, long_window: int) -> Dict:
    """評估趨勢強度"""
    strengths = []
    consistencies = []
    
    for i in range(long_window, len(values)):
        # 短期趨勢強度
        short_data = values[i-short_window:i]
        short_x = np.arange(len(short_data))
        short_slope, _, short_r, short_p, _ = stats.linregress(short_x, short_data)
        short_strength = abs(short_slope) * abs(short_r) if short_p < 0.05 else 0
        
        # 長期趨勢強度
        long_data = values[i-long_window:i]
        long_x = np.arange(len(long_data))
        long_slope, _, long_r, long_p, _ = stats.linregress(long_x, long_data)
        long_strength = abs(long_slope) * abs(long_r) if long_p < 0.05 else 0
        
        # 趨勢一致性
        consistency = 1.0 if (short_slope * long_slope) > 0 else 0.0
        
        strengths.append({
            'short_strength': short_strength,
            'long_strength': long_strength,
            'combined_strength': (short_strength + long_strength) / 2
        })
        consistencies.append(consistency)
    
    return {
        'strengths': strengths,
        'consistencies': np.array(consistencies),
        'start_index': long_window
    }

def display_level_shift_results(results: Dict, kpi_data: pd.DataFrame, 
                              kpi_name: str, fab_name: str):
    """顯示 Level Shift 檢測結果"""
    shifts = results['shifts']
    explanations = results['explanations']
    values = results['values']
    dates = results['dates']
    
    # 統計摘要
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("檢測到 Level Shift", len(shifts))
    
    with col2:
        if shifts:
            avg_magnitude = np.mean([s['change_pct'] for s in shifts])
            st.metric("平均變化幅度", f"{avg_magnitude:.2f}%")
        else:
            st.metric("平均變化幅度", "N/A")
    
    with col3:
        if shifts:
            up_shifts = sum(1 for s in shifts if s['change_direction'] == '上升')
            st.metric("上升 Level Shift", up_shifts)
        else:
            st.metric("上升 Level Shift", 0)
    
    with col4:
        if shifts:
            down_shifts = sum(1 for s in shifts if s['change_direction'] == '下降')
            st.metric("下降 Level Shift", down_shifts)
        else:
            st.metric("下降 Level Shift", 0)
    
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
    
    # 標記 Level Shift 點
    if shifts:
        shift_dates = [s['date'] for s in shifts]
        shift_values = [values[s['index']] for s in shifts]
        shift_colors = ['red' if s['change_direction'] == '下降' else 'green' for s in shifts]
        
        fig.add_trace(go.Scatter(
            x=shift_dates,
            y=shift_values,
            mode='markers',
            name='Level Shift',
            marker=dict(
                color=shift_colors,
                size=12,
                symbol='star',
                line=dict(color='black', width=1)
            )
        ))
        
        # 添加標註
        for i, shift in enumerate(shifts):
            fig.add_annotation(
                x=shift['date'],
                y=values[shift['index']],
                text=f"{shift['change_direction']}<br>{shift['change_pct']:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor=shift_colors[i],
                bgcolor=shift_colors[i],
                bordercolor="white",
                font=dict(color="white", size=10)
            )
    
    fig.update_layout(
        title=f"{fab_name} - {kpi_name} Level Shift 檢測結果",
        xaxis_title="時間",
        yaxis_title="數值",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 詳細結果表格
    if shifts:
        st.subheader("📋 Level Shift 詳細信息")
        
        shift_details = []
        for i, (shift, explanation) in enumerate(zip(shifts, explanations)):
            shift_details.append({
                '時間': shift['date'].strftime('%Y-%m-%d'),
                '變化方向': shift['change_direction'],
                '變化幅度': f"{shift['change_pct']:.2f}%",
                '變化前均值': f"{shift['before_mean']:.2f}",
                '變化後均值': f"{shift['after_mean']:.2f}",
                'P值': f"{shift['p_value']:.4f}",
                '解釋說明': explanation
            })
        
        df = pd.DataFrame(shift_details)
        st.dataframe(df, use_container_width=True)

def display_momentum_results(results: Dict, kpi_data: pd.DataFrame, 
                           kpi_name: str, fab_name: str):
    """顯示趨勢動量分析結果"""
    values = results['values']
    dates = results['dates']
    
    # 創建子圖
    n_plots = len([k for k in results.keys() if k not in ['values', 'dates', 'short_window', 'long_window']])
    fig = make_subplots(
        rows=n_plots + 1, cols=1,
        subplot_titles=['原始時序數據'] + [k.replace('_', ' ').title() for k in results.keys() 
                                      if k not in ['values', 'dates', 'short_window', 'long_window']],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # 原始數據
    fig.add_trace(
        go.Scatter(x=dates, y=values, mode='lines+markers', name='原始數據'),
        row=1, col=1
    )
    
    current_row = 2
    
    # 趨勢動量分析
    if 'momentum' in results:
        momentum = results['momentum']
        start_idx = momentum['start_index']
        
        fig.add_trace(
            go.Scatter(x=dates[start_idx:], y=momentum['short_trends'], 
                      name=f'短期趨勢 ({results["short_window"]}天)', line=dict(color='blue')),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates[start_idx:], y=momentum['long_trends'], 
                      name=f'長期趨勢 ({results["long_window"]}天)', line=dict(color='red')),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates[start_idx:], y=momentum['momentum_signals'], 
                      name='動量信號', line=dict(color='green')),
            row=current_row, col=1
        )
        current_row += 1
    
    # 連續趨勢檢測
    if 'continuous_trends' in results:
        continuous = results['continuous_trends']
        trends = [t['trend_strength'] for t in continuous['trends']]
        trend_dates = dates[continuous['window']:]
        
        fig.add_trace(
            go.Scatter(x=trend_dates, y=trends, mode='lines', name='趨勢強度'),
            row=current_row, col=1
        )
        
        # 標記連續趨勢期間
        for period in continuous['continuous_periods']:
            fig.add_vrect(
                x0=period['start_date'], x1=period['end_date'],
                fillcolor="red" if period['direction'] == "下降" else "green",
                opacity=0.2, line_width=0,
                row=current_row, col=1
            )
        
        current_row += 1
    
    # 加速度分析
    if 'acceleration' in results:
        accel = results['acceleration']
        start_idx = accel['start_index']
        
        fig.add_trace(
            go.Scatter(x=dates[start_idx:start_idx+len(accel['accelerations'])], 
                      y=accel['accelerations'], mode='lines', name='趨勢加速度'),
            row=current_row, col=1
        )
        current_row += 1
    
    # 趨勢強度評估
    if 'strength' in results:
        strength = results['strength']
        start_idx = strength['start_index']
        combined_strengths = [s['combined_strength'] for s in strength['strengths']]
        
        fig.add_trace(
            go.Scatter(x=dates[start_idx:], y=combined_strengths, 
                      mode='lines', name='綜合趨勢強度'),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates[start_idx:], y=strength['consistencies'], 
                      mode='lines', name='趨勢一致性'),
            row=current_row, col=1
        )
        current_row += 1
    
    fig.update_layout(
        height=200 * (n_plots + 1),
        title_text=f"{fab_name} - {kpi_name} 趨勢動量分析",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 分析摘要
    display_momentum_summary(results, kpi_name)

def display_momentum_summary(results: Dict, kpi_name: str):
    """顯示動量分析摘要"""
    st.subheader("📊 分析摘要")
    
    # 連續趨勢統計
    if 'continuous_trends' in results:
        continuous = results['continuous_trends']
        periods = continuous['continuous_periods']
        
        if periods:
            st.write("**連續趨勢期間:**")
            for i, period in enumerate(periods):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"期間 {i+1}", period['direction'])
                with col2:
                    st.metric("持續天數", f"{period['duration']}天")
                with col3:
                    st.metric("平均斜率", f"{period['avg_slope']:.4f}")
                with col4:
                    start_str = period['start_date'].strftime('%m-%d')
                    end_str = period['end_date'].strftime('%m-%d')
                    st.metric("時間範圍", f"{start_str}~{end_str}")
                
                # 生成解釋
                explanation = generate_trend_explanation(period, kpi_name)
                st.info(f"💡 **解釋**: {explanation}")
                st.divider()

def generate_level_shift_explanation(before_mean: float, after_mean: float, 
                                   change_pct: float, direction: str, 
                                   p_value: float, window: int) -> str:
    """生成 Level Shift 解釋"""
    magnitude_desc = "顯著" if change_pct > 15 else "中等" if change_pct > 8 else "輕微"
    confidence = "高度" if p_value < 0.01 else "中等"
    
    explanation = f"在 {window} 天的比較窗口中，KPI 出現了 {magnitude_desc} 的 {direction} Level Shift。"
    explanation += f"變化前後的平均值從 {before_mean:.2f} 變為 {after_mean:.2f}，"
    explanation += f"變化幅度達 {change_pct:.2f}%，統計顯著性為 {confidence} 信心水準 (p={p_value:.4f})。"
    
    if change_pct > 20:
        explanation += " 這是一個非常顯著的水準變化，建議深入調查可能的根本原因。"
    elif change_pct > 10:
        explanation += " 這個變化值得關注，可能反映了製程或設備的變化。"
    else:
        explanation += " 這是一個相對較小但統計顯著的變化。"
    
    return explanation

def generate_trend_explanation(period: Dict, kpi_name: str) -> str:
    """生成趨勢解釋"""
    direction = period['direction']
    duration = period['duration']
    avg_slope = period['avg_slope']
    
    if duration >= 14:
        duration_desc = "長期"
    elif duration >= 7:
        duration_desc = "中期"
    else:
        duration_desc = "短期"
    
    slope_magnitude = abs(avg_slope)
    if slope_magnitude > 1.0:
        intensity = "強烈"
    elif slope_magnitude > 0.1:
        intensity = "明顯"
    else:
        intensity = "溫和"
    
    explanation = f"檢測到 {duration_desc} 的 {intensity} {direction} 趨勢，"
    explanation += f"持續了 {duration} 天，平均變化率為 {avg_slope:.4f}/天。"
    
    if direction == "上升":
        if kpi_name in ["Yield", "Equipment_Utilization", "Quality_Score"]:
            explanation += " 這是一個積極的趨勢，表明績效正在改善。"
        elif kpi_name in ["Defect_Rate", "Cycle_Time", "Cost_Per_Unit"]:
            explanation += " 這個上升趨勢需要關注，可能表明存在問題。"
    else:  # 下降
        if kpi_name in ["Yield", "Equipment_Utilization", "Quality_Score"]:
            explanation += " 這是一個需要關注的負面趨勢，建議調查原因。"
        elif kpi_name in ["Defect_Rate", "Cycle_Time", "Cost_Per_Unit"]:
            explanation += " 這是一個積極的下降趨勢，表明情況在改善。"
    
    if duration >= 21:
        explanation += " 長期趨勢的持續性表明這可能是系統性變化，而非隨機波動。"
    
    return explanation