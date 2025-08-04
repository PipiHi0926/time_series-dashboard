import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import normaltest, jarque_bera, kstest, anderson
import warnings
warnings.filterwarnings('ignore')

def descriptive_statistics_page():
    """敘述統計分析頁面"""
    st.header("📊 敘述統計分析")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 可用 KPI: **{len(available_kpis)}** 個")
    
    # 計算所有 KPI 的統計特性
    kpi_profiles = analyze_all_kpis(fab_data, available_kpis)
    
    # 顯示總覽
    display_kpi_overview(kpi_profiles, selected_fab)
    
    # 顯示詳細分析
    display_detailed_analysis(kpi_profiles, fab_data)
    
    # 方法論建議
    display_methodology_recommendations(kpi_profiles)

def analyze_all_kpis(fab_data: pd.DataFrame, kpis: List[str]) -> Dict:
    """分析所有 KPI 的統計特性"""
    profiles = {}
    
    for kpi in kpis:
        kpi_data = fab_data[fab_data['KPI'] == kpi]['VALUE'].dropna()
        
        if len(kpi_data) < 10:
            continue
            
        profile = analyze_single_kpi(kpi_data, kpi)
        profiles[kpi] = profile
    
    return profiles

def analyze_single_kpi(data: pd.Series, kpi_name: str) -> Dict:
    """分析單個 KPI 的統計特性"""
    profile = {
        'name': kpi_name,
        'data': data,
        'n_obs': len(data),
        'basic_stats': {},
        'distribution_tests': {},
        'data_characteristics': {},
        'tags': [],
        'recommended_methods': []
    }
    
    # 基礎統計量
    profile['basic_stats'] = calculate_basic_statistics(data)
    
    # 分布檢驗
    profile['distribution_tests'] = perform_distribution_tests(data)
    
    # 數據特性分析
    profile['data_characteristics'] = analyze_data_characteristics(data)
    
    # 生成標籤
    profile['tags'] = generate_kpi_tags(profile)
    
    # 方法論建議
    profile['recommended_methods'] = recommend_analysis_methods(profile)
    
    return profile

def calculate_basic_statistics(data: pd.Series) -> Dict:
    """計算基礎統計量"""
    return {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'var': data.var(),
        'min': data.min(),
        'max': data.max(),
        'q1': data.quantile(0.25),
        'q3': data.quantile(0.75),
        'iqr': data.quantile(0.75) - data.quantile(0.25),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'cv': data.std() / data.mean() if data.mean() != 0 else np.inf,
        'range': data.max() - data.min(),
        'mad': np.median(np.abs(data - data.median())),  # 中位數絕對偏差
        'percentiles': {
            '5th': data.quantile(0.05),
            '95th': data.quantile(0.95),
            '99th': data.quantile(0.99)
        }
    }

def perform_distribution_tests(data: pd.Series) -> Dict:
    """執行分布檢驗"""
    tests = {}
    
    # 常態性檢驗
    try:
        # Shapiro-Wilk 檢驗 (適合小樣本)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            tests['shapiro'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
        
        # D'Agostino-Pearson 檢驗
        dp_stat, dp_p = normaltest(data)
        tests['dagostino_pearson'] = {'statistic': dp_stat, 'p_value': dp_p}
        
        # Jarque-Bera 檢驗
        jb_stat, jb_p = jarque_bera(data)
        tests['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_p}
        
        # Kolmogorov-Smirnov 檢驗 (與標準常態分布比較)
        ks_stat, ks_p = kstest(stats.zscore(data), 'norm')
        tests['kolmogorov_smirnov'] = {'statistic': ks_stat, 'p_value': ks_p}
        
        # Anderson-Darling 檢驗
        ad_result = anderson(data, dist='norm')
        tests['anderson_darling'] = {
            'statistic': ad_result.statistic,
            'critical_values': ad_result.critical_values,
            'significance_levels': ad_result.significance_level
        }
        
    except Exception as e:
        tests['error'] = str(e)
    
    return tests

def analyze_data_characteristics(data: pd.Series) -> Dict:
    """分析數據特性"""
    characteristics = {}
    
    # 數據類型推斷
    characteristics['inferred_type'] = infer_data_type(data)
    
    # 零值分析
    zero_count = (data == 0).sum()
    characteristics['zero_proportion'] = zero_count / len(data)
    characteristics['has_zeros'] = zero_count > 0
    
    # 稀疏性分析
    characteristics['sparsity'] = calculate_sparsity(data)
    
    # 異常值分析
    characteristics['outliers'] = detect_outliers_iqr(data)
    
    # 趨勢分析
    characteristics['trend'] = analyze_trend(data)
    
    # 平穩性分析 (簡化版)
    characteristics['stationarity'] = analyze_stationarity(data)
    
    # 週期性分析
    characteristics['periodicity'] = analyze_periodicity(data)
    
    # 值域分析
    characteristics['range_analysis'] = analyze_value_range(data)
    
    return characteristics

def infer_data_type(data: pd.Series) -> str:
    """推斷數據類型"""
    unique_values = data.nunique()
    total_values = len(data)
    
    # 檢查是否為整數
    is_integer = all(data.dropna() == data.dropna().astype(int))
    
    # 檢查值域
    min_val, max_val = data.min(), data.max()
    
    if unique_values == 2 and set(data.unique()) <= {0, 1}:
        return 'binary'
    elif unique_values <= 10 and is_integer and min_val >= 0:
        return 'categorical_ordinal'
    elif is_integer and min_val >= 0 and max_val <= 100:
        return 'count_small'
    elif is_integer and min_val >= 0:
        return 'count_large'
    elif 0 <= min_val and max_val <= 1:
        return 'probability'
    elif 0 <= min_val and max_val <= 100:
        return 'percentage'
    elif min_val >= 0:
        return 'continuous_positive'
    else:
        return 'continuous_general'

def calculate_sparsity(data: pd.Series) -> Dict:
    """計算稀疏性指標"""
    total_count = len(data)
    non_zero_count = (data != 0).sum()
    zero_count = total_count - non_zero_count
    
    # 計算不同類型的稀疏性
    sparsity_metrics = {
        'zero_ratio': zero_count / total_count,
        'density': non_zero_count / total_count,
        'is_sparse': zero_count / total_count > 0.7,
        'consecutive_zeros': find_longest_consecutive_zeros(data),
        'zero_runs': count_zero_runs(data)
    }
    
    return sparsity_metrics

def find_longest_consecutive_zeros(data: pd.Series) -> int:
    """找出最長連續零值序列"""
    is_zero = data == 0
    consecutive_counts = []
    current_count = 0
    
    for val in is_zero:
        if val:
            current_count += 1
        else:
            if current_count > 0:
                consecutive_counts.append(current_count)
            current_count = 0
    
    if current_count > 0:
        consecutive_counts.append(current_count)
    
    return max(consecutive_counts) if consecutive_counts else 0

def count_zero_runs(data: pd.Series) -> int:
    """計算零值序列的數量"""
    is_zero = data == 0
    runs = 0
    in_run = False
    
    for val in is_zero:
        if val and not in_run:
            runs += 1
            in_run = True
        elif not val:
            in_run = False
    
    return runs

def detect_outliers_iqr(data: pd.Series) -> Dict:
    """使用 IQR 方法檢測異常值"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return {
        'count': len(outliers),
        'proportion': len(outliers) / len(data),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'extreme_low': (data < lower_bound).sum(),
        'extreme_high': (data > upper_bound).sum()
    }

def analyze_trend(data: pd.Series) -> Dict:
    """分析趨勢"""
    x = np.arange(len(data))
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat',
            'strength': abs(r_value)
        }
    except:
        return {'error': 'Unable to compute trend'}

def analyze_stationarity(data: pd.Series) -> Dict:
    """分析平穩性 (簡化版)"""
    # 分段檢查均值和方差的穩定性
    n = len(data)
    first_half = data[:n//2]
    second_half = data[n//2:]
    
    mean_diff = abs(second_half.mean() - first_half.mean()) / data.std()
    var_ratio = second_half.var() / first_half.var() if first_half.var() > 0 else 1
    
    return {
        'mean_shift': mean_diff,
        'variance_ratio': var_ratio,
        'likely_stationary': mean_diff < 0.5 and 0.5 < var_ratio < 2.0
    }

def analyze_periodicity(data: pd.Series) -> Dict:
    """分析週期性 (簡化版)"""
    if len(data) < 20:
        return {'insufficient_data': True}
    
    # 簡單的自相關分析
    max_lag = min(len(data) // 4, 50)
    autocorr = [data.autocorr(lag=i) for i in range(1, max_lag + 1)]
    
    # 找出顯著的自相關
    significant_lags = [i+1 for i, corr in enumerate(autocorr) if abs(corr) > 0.3]
    
    return {
        'max_autocorr': max(autocorr) if autocorr else 0,
        'significant_lags': significant_lags,
        'has_periodicity': len(significant_lags) > 0
    }

def analyze_value_range(data: pd.Series) -> Dict:
    """分析值域特性"""
    return {
        'is_bounded_below': data.min() >= 0,
        'is_bounded_above': data.max() <= 100,  # 假設百分比類型
        'is_percentage_like': 0 <= data.min() and data.max() <= 100,
        'is_probability_like': 0 <= data.min() and data.max() <= 1,
        'has_negative_values': (data < 0).any(),
        'value_concentration': calculate_value_concentration(data)
    }

def calculate_value_concentration(data: pd.Series) -> Dict:
    """計算值集中度"""
    value_counts = data.value_counts()
    total_count = len(data)
    
    # 計算不同集中度指標
    top_1_ratio = value_counts.iloc[0] / total_count if len(value_counts) > 0 else 0
    top_5_ratio = value_counts.head(5).sum() / total_count if len(value_counts) > 0 else 0
    
    return {
        'unique_ratio': len(value_counts) / total_count,
        'top_1_concentration': top_1_ratio,
        'top_5_concentration': top_5_ratio,
        'is_highly_concentrated': top_1_ratio > 0.5
    }

def generate_kpi_tags(profile: Dict) -> List[str]:
    """生成 KPI 標籤"""
    tags = []
    
    basic_stats = profile['basic_stats']
    characteristics = profile['data_characteristics']
    distribution_tests = profile['distribution_tests']
    
    # 數據類型標籤
    data_type = characteristics['inferred_type']
    tags.append(f"TYPE_{data_type.upper()}")
    
    # 常態性標籤
    is_normal = check_normality(distribution_tests)
    tags.append("NORMAL" if is_normal else "NON_NORMAL")
    
    # 稀疏性標籤
    if characteristics['sparsity']['is_sparse']:
        tags.append("SPARSE")
        tags.append(f"ZERO_HEAVY_{int(characteristics['sparsity']['zero_ratio']*100)}%")
    
    # 偏度標籤
    skewness = basic_stats['skewness']
    if abs(skewness) < 0.5:
        tags.append("SYMMETRIC")
    elif skewness > 0.5:
        tags.append("RIGHT_SKEWED")
    elif skewness < -0.5:
        tags.append("LEFT_SKEWED")
    
    # 峰度標籤
    kurtosis = basic_stats['kurtosis']
    if kurtosis > 1:
        tags.append("HEAVY_TAILED")
    elif kurtosis < -1:
        tags.append("LIGHT_TAILED")
    
    # 變異性標籤
    cv = basic_stats['cv']
    if cv < 0.1:
        tags.append("LOW_VARIABILITY")
    elif cv > 0.5:
        tags.append("HIGH_VARIABILITY")
    
    # 異常值標籤
    outlier_prop = characteristics['outliers']['proportion']
    if outlier_prop > 0.05:
        tags.append("OUTLIER_PRONE")
    
    # 趨勢標籤
    trend = characteristics['trend']
    if trend.get('is_significant', False):
        tags.append(f"TREND_{trend['direction'].upper()}")
    
    # 平穩性標籤
    if characteristics['stationarity']['likely_stationary']:
        tags.append("STATIONARY")
    else:
        tags.append("NON_STATIONARY")
    
    # 週期性標籤
    if characteristics['periodicity'].get('has_periodicity', False):
        tags.append("PERIODIC")
    
    return tags

def check_normality(distribution_tests: Dict) -> bool:
    """檢查是否符合常態分布"""
    normal_tests = 0
    total_tests = 0
    
    for test_name, result in distribution_tests.items():
        if test_name in ['shapiro', 'dagostino_pearson', 'jarque_bera', 'kolmogorov_smirnov']:
            total_tests += 1
            if result.get('p_value', 0) > 0.05:  # 不拒絕常態分布假設
                normal_tests += 1
    
    return normal_tests / total_tests > 0.5 if total_tests > 0 else False

def recommend_analysis_methods(profile: Dict) -> List[str]:
    """推薦分析方法"""
    recommendations = []
    tags = profile['tags']
    characteristics = profile['data_characteristics']
    
    # 基於數據類型推薦
    if 'TYPE_BINARY' in tags:
        recommendations.extend(['邏輯回歸', '卡方檢驗', '二項分布分析'])
    elif 'TYPE_COUNT_SMALL' in tags or 'TYPE_COUNT_LARGE' in tags:
        recommendations.extend(['泊松回歸', '負二項回歸', '計數模型'])
    elif 'SPARSE' in tags:
        recommendations.extend(['稀疏事件分析', '零膨脹模型', '生存分析'])
    
    # 基於分布特性推薦
    if 'NORMAL' in tags:
        recommendations.extend(['Z-Score檢測', '基於常態分布的控制圖', 'T檢驗'])
    else:
        recommendations.extend(['非參數檢驗', 'IQR異常檢測', '中位數基礎分析'])
    
    # 基於時序特性推薦
    if 'TREND_INCREASING' in tags or 'TREND_DECREASING' in tags:
        recommendations.extend(['趨勢分析', '回歸分析', '預測模型'])
    
    if 'PERIODIC' in tags:
        recommendations.extend(['季節性分解', 'ARIMA模型', 'FFT分析'])
    
    if 'NON_STATIONARY' in tags:
        recommendations.extend(['差分分析', 'ARIMA模型', '結構變點檢測'])
    
    # 基於變異性推薦
    if 'HIGH_VARIABILITY' in tags:
        recommendations.extend(['穩健統計方法', 'Modified Z-Score', '百分位數方法'])
    
    if 'OUTLIER_PRONE' in tags:
        recommendations.extend(['異常值檢測', '穩健回歸', '截尾統計'])
    
    # 移除重複並限制數量
    recommendations = list(set(recommendations))[:8]
    
    return recommendations

def display_kpi_overview(profiles: Dict, fab_name: str):
    """顯示 KPI 總覽"""
    st.subheader(f"📈 {fab_name} KPI 總覽")
    
    if not profiles:
        st.warning("無足夠數據進行分析")
        return
    
    # 統計摘要
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("KPI 總數", len(profiles))
    
    with col2:
        normal_count = sum(1 for p in profiles.values() if 'NORMAL' in p['tags'])
        st.metric("常態分布 KPI", normal_count)
    
    with col3:
        sparse_count = sum(1 for p in profiles.values() if 'SPARSE' in p['tags'])
        st.metric("稀疏 KPI", sparse_count)
    
    with col4:
        trend_count = sum(1 for p in profiles.values() 
                         if any('TREND_' in tag for tag in p['tags']))
        st.metric("有趨勢 KPI", trend_count)
    
    # KPI 分類圖表
    display_kpi_classification_charts(profiles)

def display_kpi_classification_charts(profiles: Dict):
    """顯示 KPI 分類圖表"""
    # 數據類型分布
    type_counts = {}
    for profile in profiles.values():
        data_type = profile['data_characteristics']['inferred_type']
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
    
    # 常態性分布
    normality_counts = {'常態': 0, '非常態': 0}
    for profile in profiles.values():
        if 'NORMAL' in profile['tags']:
            normality_counts['常態'] += 1
        else:
            normality_counts['非常態'] += 1
    
    # 創建圖表
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('數據類型分布', '常態性分布', '稀疏性分布'),
        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "bar"}]]
    )
    
    # 數據類型餅圖
    fig.add_trace(
        go.Pie(labels=list(type_counts.keys()), values=list(type_counts.values()), name="數據類型"),
        row=1, col=1
    )
    
    # 常態性餅圖
    fig.add_trace(
        go.Pie(labels=list(normality_counts.keys()), values=list(normality_counts.values()), name="常態性"),
        row=1, col=2
    )
    
    # 稀疏性條形圖
    sparse_data = []
    for kpi_name, profile in profiles.items():
        zero_ratio = profile['data_characteristics']['sparsity']['zero_ratio']
        sparse_data.append({'KPI': kpi_name, 'Zero_Ratio': zero_ratio})
    
    sparse_df = pd.DataFrame(sparse_data)
    
    fig.add_trace(
        go.Bar(x=sparse_df['KPI'], y=sparse_df['Zero_Ratio'], name="零值比例"),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_detailed_analysis(profiles: Dict, fab_data: pd.DataFrame):
    """顯示詳細分析"""
    st.subheader("🔍 詳細 KPI 分析")
    
    # KPI 選擇
    selected_kpi = st.selectbox(
        "選擇要詳細分析的 KPI:",
        options=list(profiles.keys()),
        key="detailed_analysis_kpi"
    )
    
    if selected_kpi not in profiles:
        return
    
    profile = profiles[selected_kpi]
    
    # 顯示基本信息
    display_kpi_basic_info(profile)
    
    # 顯示統計測試結果
    display_statistical_tests(profile)
    
    # 顯示視覺化
    display_kpi_visualization(profile, fab_data, selected_kpi)

def display_kpi_basic_info(profile: Dict):
    """顯示 KPI 基本信息"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 基礎統計量**")
        stats_df = pd.DataFrame([
            ['觀測數', f"{profile['n_obs']:,}"],
            ['平均值', f"{profile['basic_stats']['mean']:.4f}"],
            ['中位數', f"{profile['basic_stats']['median']:.4f}"],
            ['標準差', f"{profile['basic_stats']['std']:.4f}"],
            ['變異係數', f"{profile['basic_stats']['cv']:.4f}"],
            ['偏度', f"{profile['basic_stats']['skewness']:.4f}"],
            ['峰度', f"{profile['basic_stats']['kurtosis']:.4f}"]
        ], columns=['統計量', '值'])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**🏷️ KPI 標籤**")
        tags_text = ""
        for tag in profile['tags']:
            color = get_tag_color(tag)
            tags_text += f'<span style="background-color: {color}; padding: 2px 6px; margin: 2px; border-radius: 3px; font-size: 12px;">{tag}</span>'
        
        st.markdown(tags_text, unsafe_allow_html=True)
        
        st.write("**💡 建議方法**")
        for method in profile['recommended_methods']:
            st.write(f"• {method}")

def get_tag_color(tag: str) -> str:
    """取得標籤顏色"""
    if 'TYPE_' in tag:
        return '#e3f2fd'
    elif tag in ['NORMAL', 'STATIONARY']:
        return '#e8f5e8'
    elif tag in ['NON_NORMAL', 'NON_STATIONARY']:
        return '#fff3e0'
    elif 'SPARSE' in tag or 'ZERO_' in tag:
        return '#fce4ec'
    elif 'TREND_' in tag:
        return '#f3e5f5'
    else:
        return '#f5f5f5'

def display_statistical_tests(profile: Dict):
    """顯示統計檢驗結果"""
    st.write("**🧪 統計檢驗結果**")
    
    tests = profile['distribution_tests']
    if 'error' in tests:
        st.error(f"統計檢驗錯誤: {tests['error']}")
        return
    
    test_results = []
    for test_name, result in tests.items():
        if test_name == 'anderson_darling':
            continue  # Anderson-Darling 結果較複雜，暫時跳過
        
        p_value = result.get('p_value', None)
        if p_value is not None:
            significance = "顯著" if p_value < 0.05 else "不顯著"
            interpretation = get_test_interpretation(test_name, p_value < 0.05)
            
            test_results.append({
                '檢驗方法': format_test_name(test_name),
                'P值': f"{p_value:.6f}",
                '顯著性': significance,
                '解釋': interpretation
            })
    
    if test_results:
        st.dataframe(pd.DataFrame(test_results), use_container_width=True, hide_index=True)

def format_test_name(test_name: str) -> str:
    """格式化檢驗名稱"""
    name_map = {
        'shapiro': 'Shapiro-Wilk',
        'dagostino_pearson': "D'Agostino-Pearson",
        'jarque_bera': 'Jarque-Bera',
        'kolmogorov_smirnov': 'Kolmogorov-Smirnov'
    }
    return name_map.get(test_name, test_name)

def get_test_interpretation(test_name: str, is_significant: bool) -> str:
    """取得檢驗結果解釋"""
    if test_name in ['shapiro', 'dagostino_pearson', 'jarque_bera', 'kolmogorov_smirnov']:
        if is_significant:
            return "拒絕常態分布假設"
        else:
            return "不拒絕常態分布假設"
    return ""

def display_kpi_visualization(profile: Dict, fab_data: pd.DataFrame, kpi_name: str):
    """顯示 KPI 視覺化"""
    st.write("**📈 KPI 視覺化分析**")
    
    # 取得 KPI 數據
    kpi_data = fab_data[fab_data['KPI'] == kpi_name].copy()
    kpi_data = kpi_data.sort_values('REPORT_TIME')
    
    # 創建子圖
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('時序圖', '分布直方圖', 'Q-Q圖', '箱型圖'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 時序圖
    fig.add_trace(
        go.Scatter(x=kpi_data['REPORT_TIME'], y=kpi_data['VALUE'], 
                  mode='lines+markers', name='原始數據', line=dict(width=1)),
        row=1, col=1
    )
    
    # 分布直方圖
    fig.add_trace(
        go.Histogram(x=profile['data'], nbinsx=30, name='分布'),
        row=1, col=2
    )
    
    # Q-Q 圖 (與常態分布比較)
    data_sorted = np.sort(profile['data'])
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data_sorted)))
    
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=data_sorted, 
                  mode='markers', name='Q-Q點'),
        row=2, col=1
    )
    
    # 添加 Q-Q 參考線
    min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', name='理論線', line=dict(dash='dash')),
        row=2, col=1
    )
    
    # 箱型圖
    fig.add_trace(
        go.Box(y=profile['data'], name='箱型圖'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_methodology_recommendations(profiles: Dict):
    """顯示方法論建議"""
    st.subheader("💡 方法論建議")
    
    # 統計所有推薦方法
    method_counts = {}
    for profile in profiles.values():
        for method in profile['recommended_methods']:
            method_counts[method] = method_counts.get(method, 0) + 1
    
    # 按適用的 KPI 數量排序
    sorted_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
    
    st.write("**🎯 針對當前 FAB 的推薦分析方法 (按適用性排序):**")
    
    for method, count in sorted_methods:
        percentage = (count / len(profiles)) * 100
        st.write(f"• **{method}** - 適用於 {count} 個 KPI ({percentage:.1f}%)")
        
        # 顯示適用的 KPI
        applicable_kpis = [kpi for kpi, profile in profiles.items() 
                          if method in profile['recommended_methods']]
        
        with st.expander(f"查看適用 {method} 的 KPI"):
            for kpi in applicable_kpis:
                kpi_tags = profiles[kpi]['tags']
                st.write(f"  - **{kpi}**: {', '.join(kpi_tags[:3])}...")

    # 特殊建議
    st.write("**⚠️ 特殊注意事項:**")
    
    sparse_kpis = [kpi for kpi, profile in profiles.items() if 'SPARSE' in profile['tags']]
    if sparse_kpis:
        st.warning(f"稀疏數據 KPI ({', '.join(sparse_kpis)}): 建議使用專門的稀疏數據分析方法")
    
    binary_kpis = [kpi for kpi, profile in profiles.items() if 'TYPE_BINARY' in profile['tags']]
    if binary_kpis:
        st.info(f"二元數據 KPI ({', '.join(binary_kpis)}): 避免使用連續數據分析方法")
    
    non_normal_kpis = [kpi for kpi, profile in profiles.items() if 'NON_NORMAL' in profile['tags']]
    if len(non_normal_kpis) > len(profiles) * 0.7:
        st.warning("大部分 KPI 不符合常態分布，建議優先使用非參數方法")