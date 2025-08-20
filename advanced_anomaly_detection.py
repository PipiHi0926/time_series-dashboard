import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from matplotlib_utils import render_matplotlib_figure, create_isolation_forest_plot, create_dbscan_plot, create_anomaly_plot
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def isolation_forest_analysis_page():
    """Isolation Forest 異常檢測頁面"""
    st.header("🌲 Isolation Forest 異常檢測")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 當前 KPI: **{st.session_state.selected_kpi}**")
    
    st.subheader("⚙️ 模型參數設定")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_kpi = st.selectbox(
            "選擇 KPI:",
            options=available_kpis,
            index=available_kpis.index(st.session_state.selected_kpi) if st.session_state.selected_kpi in available_kpis else 0
        )
    
    with col2:
        contamination = st.slider(
            "污染率 (異常比例):",
            min_value=0.01, max_value=0.3, value=0.1, step=0.01,
            help="預期異常點佔總數據的比例"
        )
    
    with col3:
        n_estimators = st.slider(
            "樹的數量:",
            min_value=50, max_value=300, value=100, step=10,
            help="隨機森林中樹的數量，更多的樹通常提供更好的性能"
        )
    
    with col4:
        max_features = st.slider(
            "最大特徵數:",
            min_value=1, max_value=10, value=1, step=1,
            help="每棵樹使用的最大特徵數"
        )
    
    # 時間窗口特徵
    st.subheader("📊 特徵工程設定")
    col1, col2 = st.columns(2)
    
    with col1:
        use_time_features = st.checkbox("使用時間特徵", value=True, help="包括小時、星期等時間特徵")
        window_size = st.slider("滾動窗口大小:", 3, 30, 7, help="用於計算滾動統計特徵的窗口大小")
    
    with col2:
        use_lag_features = st.checkbox("使用滯後特徵", value=True, help="包括前幾期的數值作為特徵")
        n_lags = st.slider("滯後期數:", 1, 10, 3, help="包括多少個滯後期的特徵")
    
    if st.button("🔍 執行 Isolation Forest 分析", type="primary"):
        kpi_data = fab_data[fab_data['KPI'] == selected_kpi].copy()
        kpi_data = kpi_data.sort_values('REPORT_TIME')
        
        if len(kpi_data) < 20:
            st.error("❌ 數據點不足，至少需要20個數據點")
            return
        
        # 執行 Isolation Forest 分析
        results = perform_isolation_forest_analysis(
            kpi_data, contamination, n_estimators, max_features,
            use_time_features, use_lag_features, window_size, n_lags
        )
        
        # 顯示結果
        display_isolation_forest_results(results, selected_kpi, selected_fab)
        
        # 顯示特徵重要性分析
        display_feature_importance_analysis(results)

def dbscan_analysis_page():
    """DBSCAN 聚類異常檢測頁面"""
    st.header("🎯 DBSCAN 聚類異常檢測")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 當前 KPI: **{st.session_state.selected_kpi}**")
    
    st.subheader("⚙️ 聚類參數設定")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_kpi = st.selectbox(
            "選擇 KPI:",
            options=available_kpis,
            index=available_kpis.index(st.session_state.selected_kpi) if st.session_state.selected_kpi in available_kpis else 0
        )
    
    with col2:
        eps = st.slider(
            "eps (鄰域半徑):",
            min_value=0.1, max_value=5.0, value=0.5, step=0.1,
            help="定義鄰域的半徑，較小的值會產生更多的小聚類"
        )
    
    with col3:
        min_samples = st.slider(
            "最小樣本數:",
            min_value=3, max_value=20, value=5, step=1,
            help="形成聚類所需的最小樣本數"
        )
    
    with col4:
        metric = st.selectbox(
            "距離度量:",
            ["euclidean", "manhattan", "cosine"],
            help="計算點之間距離的方法"
        )
    
    # 特徵選擇
    st.subheader("📊 特徵選擇")
    col1, col2 = st.columns(2)
    
    with col1:
        use_value_features = st.checkbox("使用數值特徵", value=True)
        use_statistical_features = st.checkbox("使用統計特徵", value=True, help="包括滾動均值、標準差等")
    
    with col2:
        use_time_features = st.checkbox("使用時間特徵", value=False, help="包括小時、日期等")
        window_size = st.slider("統計窗口大小:", 3, 30, 7)
    
    if st.button("🔍 執行 DBSCAN 分析", type="primary"):
        kpi_data = fab_data[fab_data['KPI'] == selected_kpi].copy()
        kpi_data = kpi_data.sort_values('REPORT_TIME')
        
        if len(kpi_data) < min_samples * 2:
            st.error(f"❌ 數據點不足，至少需要 {min_samples * 2} 個數據點")
            return
        
        # 執行 DBSCAN 分析
        results = perform_dbscan_analysis(
            kpi_data, eps, min_samples, metric,
            use_value_features, use_statistical_features, 
            use_time_features, window_size
        )
        
        # 顯示結果
        display_dbscan_results(results, selected_kpi, selected_fab)
        
        # 顯示聚類分析
        display_cluster_analysis(results)

def ensemble_anomaly_detection_page():
    """集成異常檢測頁面"""
    st.header("🎪 集成異常檢測")
    
    if st.session_state.fab_data is None:
        st.warning("⚠️ 請先上傳數據並選擇 FAB")
        st.info("💡 請先從左側選單選擇「KPI 快速分析」載入數據")
        return
    
    fab_data = st.session_state.fab_data
    selected_fab = st.session_state.selected_fab
    available_kpis = st.session_state.available_kpis
    
    # 顯示當前選擇
    st.info(f"🏭 當前 FAB: **{selected_fab}** | 📊 當前 KPI: **{st.session_state.selected_kpi}**")
    
    st.subheader("🎯 選擇檢測方法")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_kpi = st.selectbox(
            "選擇 KPI:",
            options=available_kpis,
            index=available_kpis.index(st.session_state.selected_kpi) if st.session_state.selected_kpi in available_kpis else 0
        )
        
        methods = st.multiselect(
            "選擇檢測方法:",
            ["Z-Score", "IQR", "Isolation Forest", "DBSCAN", "LOF", "One-Class SVM"],
            default=["Z-Score", "IQR", "Isolation Forest"],
            help="選擇多種方法進行集成檢測"
        )
    
    with col2:
        ensemble_strategy = st.selectbox(
            "集成策略:",
            ["Voting (投票)", "Union (並集)", "Intersection (交集)", "Weighted (加權)"],
            help="如何結合多種方法的結果"
        )
        
        threshold_percentile = st.slider(
            "異常閾值百分位:",
            min_value=90, max_value=99, value=95,
            help="異常點的百分位閾值"
        )
    
    if len(methods) < 2:
        st.warning("⚠️ 請至少選擇兩種檢測方法")
        return
    
    if st.button("🔍 執行集成異常檢測", type="primary"):
        kpi_data = fab_data[fab_data['KPI'] == selected_kpi].copy()
        kpi_data = kpi_data.sort_values('REPORT_TIME')
        
        if len(kpi_data) < 20:
            st.error("❌ 數據點不足，至少需要20個數據點")
            return
        
        # 執行集成異常檢測
        results = perform_ensemble_anomaly_detection(
            kpi_data, methods, ensemble_strategy, threshold_percentile
        )
        
        # 顯示結果
        display_ensemble_results(results, selected_kpi, selected_fab, methods)

def perform_isolation_forest_analysis(kpi_data: pd.DataFrame, contamination: float,
                                     n_estimators: int, max_features: int,
                                     use_time_features: bool, use_lag_features: bool,
                                     window_size: int, n_lags: int) -> Dict:
    """執行 Isolation Forest 分析"""
    
    # 特徵工程
    features_df = create_features_for_isolation_forest(
        kpi_data, use_time_features, use_lag_features, window_size, n_lags
    )
    
    # 標準化特徵
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)
    
    # 訓練 Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    predictions = iso_forest.fit_predict(features_scaled)
    anomaly_scores = iso_forest.decision_function(features_scaled)
    
    # 找出異常點
    anomaly_mask = predictions == -1
    anomaly_indices = np.where(anomaly_mask)[0]
    
    return {
        'kpi_data': kpi_data,
        'features_df': features_df,
        'features_scaled': features_scaled,
        'predictions': predictions,
        'anomaly_scores': anomaly_scores,
        'anomaly_indices': anomaly_indices,
        'model': iso_forest,
        'scaler': scaler,
        'contamination': contamination
    }

def perform_dbscan_analysis(kpi_data: pd.DataFrame, eps: float, min_samples: int,
                          metric: str, use_value_features: bool, 
                          use_statistical_features: bool, use_time_features: bool,
                          window_size: int) -> Dict:
    """執行 DBSCAN 分析"""
    
    # 特徵工程
    features_df = create_features_for_dbscan(
        kpi_data, use_value_features, use_statistical_features, 
        use_time_features, window_size
    )
    
    # 標準化特徵
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)
    
    # 執行 DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = dbscan.fit_predict(features_scaled)
    
    # 識別噪聲點（異常點）
    anomaly_mask = labels == -1
    anomaly_indices = np.where(anomaly_mask)[0]
    
    return {
        'kpi_data': kpi_data,
        'features_df': features_df,
        'features_scaled': features_scaled,
        'labels': labels,
        'anomaly_indices': anomaly_indices,
        'model': dbscan,
        'scaler': scaler,
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
    }

def perform_ensemble_anomaly_detection(kpi_data: pd.DataFrame, methods: List[str],
                                     ensemble_strategy: str, threshold_percentile: int) -> Dict:
    """執行集成異常檢測"""
    
    values = kpi_data['VALUE'].values
    results = {}
    anomaly_scores_dict = {}
    
    # 對每種方法執行檢測
    for method in methods:
        if method == "Z-Score":
            mean_val = np.mean(values)
            std_val = np.std(values)
            scores = np.abs((values - mean_val) / std_val)
            threshold = 2.0
            anomalies = np.where(scores > threshold)[0]
            
        elif method == "IQR":
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            scores = np.maximum((Q1 - values) / IQR, (values - Q3) / IQR)
            threshold = 1.5
            anomalies = np.where(scores > threshold)[0]
            
        elif method == "Isolation Forest":
            # 簡化的 Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            features = values.reshape(-1, 1)
            predictions = iso_forest.fit_predict(features)
            scores = -iso_forest.decision_function(features)  # 轉換為正的異常分數
            anomalies = np.where(predictions == -1)[0]
            
        elif method == "LOF":
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            predictions = lof.fit_predict(values.reshape(-1, 1))
            scores = -lof.negative_outlier_factor_
            anomalies = np.where(predictions == -1)[0]
            
        elif method == "One-Class SVM":
            svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
            predictions = svm.fit_predict(values.reshape(-1, 1))
            scores = -svm.decision_function(values.reshape(-1, 1)).flatten()
            anomalies = np.where(predictions == -1)[0]
            
        else:
            continue
        
        results[method] = {
            'anomalies': anomalies,
            'scores': scores
        }
        anomaly_scores_dict[method] = scores
    
    # 集成策略
    ensemble_anomalies = combine_anomaly_results(
        results, ensemble_strategy, len(values), threshold_percentile
    )
    
    return {
        'kpi_data': kpi_data,
        'individual_results': results,
        'ensemble_anomalies': ensemble_anomalies,
        'anomaly_scores_dict': anomaly_scores_dict,
        'ensemble_strategy': ensemble_strategy
    }

def create_features_for_isolation_forest(kpi_data: pd.DataFrame, use_time_features: bool,
                                        use_lag_features: bool, window_size: int, n_lags: int) -> pd.DataFrame:
    """為 Isolation Forest 創建特徵"""
    features = pd.DataFrame()
    
    # 基礎特徵
    features['value'] = kpi_data['VALUE']
    
    # 滾動統計特徵
    features['rolling_mean'] = kpi_data['VALUE'].rolling(window_size).mean()
    features['rolling_std'] = kpi_data['VALUE'].rolling(window_size).std()
    features['rolling_min'] = kpi_data['VALUE'].rolling(window_size).min()
    features['rolling_max'] = kpi_data['VALUE'].rolling(window_size).max()
    
    # 滯後特徵
    if use_lag_features:
        for lag in range(1, n_lags + 1):
            features[f'lag_{lag}'] = kpi_data['VALUE'].shift(lag)
    
    # 時間特徵
    if use_time_features:
        kpi_data['REPORT_TIME'] = pd.to_datetime(kpi_data['REPORT_TIME'])
        features['hour'] = kpi_data['REPORT_TIME'].dt.hour
        features['day_of_week'] = kpi_data['REPORT_TIME'].dt.dayofweek
        features['day_of_month'] = kpi_data['REPORT_TIME'].dt.day
        features['month'] = kpi_data['REPORT_TIME'].dt.month
    
    # 變化特徵
    features['diff'] = kpi_data['VALUE'].diff()
    features['pct_change'] = kpi_data['VALUE'].pct_change()
    
    # 填補缺失值
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features

def create_features_for_dbscan(kpi_data: pd.DataFrame, use_value_features: bool,
                              use_statistical_features: bool, use_time_features: bool,
                              window_size: int) -> pd.DataFrame:
    """為 DBSCAN 創建特徵"""
    features = pd.DataFrame()
    
    if use_value_features:
        features['value'] = kpi_data['VALUE']
        features['value_normalized'] = (kpi_data['VALUE'] - kpi_data['VALUE'].mean()) / kpi_data['VALUE'].std()
    
    if use_statistical_features:
        features['rolling_mean'] = kpi_data['VALUE'].rolling(window_size).mean()
        features['rolling_std'] = kpi_data['VALUE'].rolling(window_size).std()
        features['rolling_skew'] = kpi_data['VALUE'].rolling(window_size).skew()
        features['z_score'] = (kpi_data['VALUE'] - kpi_data['VALUE'].rolling(window_size).mean()) / kpi_data['VALUE'].rolling(window_size).std()
    
    if use_time_features:
        kpi_data['REPORT_TIME'] = pd.to_datetime(kpi_data['REPORT_TIME'])
        features['hour'] = kpi_data['REPORT_TIME'].dt.hour
        features['day_of_week'] = kpi_data['REPORT_TIME'].dt.dayofweek
    
    # 填補缺失值
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return features

def combine_anomaly_results(results: Dict, strategy: str, n_samples: int, threshold_percentile: int) -> np.ndarray:
    """結合多種異常檢測方法的結果"""
    
    if strategy == "Union (並集)":
        # 任何方法檢測到的異常都算異常
        all_anomalies = set()
        for method_result in results.values():
            all_anomalies.update(method_result['anomalies'])
        return np.array(list(all_anomalies))
    
    elif strategy == "Intersection (交集)":
        # 所有方法都檢測到的異常才算異常
        if not results:
            return np.array([])
        
        common_anomalies = set(results[list(results.keys())[0]]['anomalies'])
        for method_result in list(results.values())[1:]:
            common_anomalies = common_anomalies.intersection(set(method_result['anomalies']))
        return np.array(list(common_anomalies))
    
    elif strategy == "Voting (投票)":
        # 多數投票
        vote_counts = np.zeros(n_samples)
        for method_result in results.values():
            vote_counts[method_result['anomalies']] += 1
        
        threshold_votes = len(results) // 2 + 1
        return np.where(vote_counts >= threshold_votes)[0]
    
    elif strategy == "Weighted (加權)":
        # 基於異常分數的加權平均
        weighted_scores = np.zeros(n_samples)
        total_weight = 0
        
        for method, method_result in results.items():
            # 簡單的權重分配
            weight = 1.0
            normalized_scores = (method_result['scores'] - np.min(method_result['scores'])) / (np.max(method_result['scores']) - np.min(method_result['scores']) + 1e-10)
            weighted_scores += weight * normalized_scores
            total_weight += weight
        
        weighted_scores /= total_weight
        threshold = np.percentile(weighted_scores, threshold_percentile)
        return np.where(weighted_scores > threshold)[0]
    
    return np.array([])

def display_isolation_forest_results(results: Dict, kpi_name: str, fab_name: str):
    """顯示 Isolation Forest 結果"""
    
    # 統計摘要
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("數據點總數", len(results['kpi_data']))
    
    with col2:
        st.metric("異常點數量", len(results['anomaly_indices']))
    
    with col3:
        anomaly_rate = len(results['anomaly_indices']) / len(results['kpi_data']) * 100
        st.metric("異常率", f"{anomaly_rate:.2f}%")
    
    with col4:
        st.metric("特徵數量", results['features_df'].shape[1])
    
    # 視覺化
    kpi_data = results['kpi_data']
    dates = pd.to_datetime(kpi_data['REPORT_TIME'])
    values = kpi_data['VALUE'].values
    
    fig = create_isolation_forest_plot(
        dates=dates,
        values=values,
        anomaly_scores=results['anomaly_scores'],
        predictions=results['predictions'],
        contamination=results['contamination'],
        title=f"{fab_name} - {kpi_name}"
    )
    
    render_matplotlib_figure(fig)

def display_dbscan_results(results: Dict, kpi_name: str, fab_name: str):
    """顯示 DBSCAN 結果"""
    
    # 統計摘要
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("數據點總數", len(results['kpi_data']))
    
    with col2:
        st.metric("聚類數量", results['n_clusters'])
    
    with col3:
        st.metric("噪聲點數量", len(results['anomaly_indices']))
    
    with col4:
        noise_rate = len(results['anomaly_indices']) / len(results['kpi_data']) * 100
        st.metric("噪聲率", f"{noise_rate:.2f}%")
    
    # 視覺化
    kpi_data = results['kpi_data']
    dates = pd.to_datetime(kpi_data['REPORT_TIME'])
    values = kpi_data['VALUE'].values
    
    fig = create_dbscan_plot(
        dates=dates,
        values=values,
        labels=results['labels'],
        eps=results['eps'],
        min_samples=results['min_samples'],
        title=f"{fab_name} - {kpi_name}"
    )
    
    render_matplotlib_figure(fig)

def display_ensemble_results(results: Dict, kpi_name: str, fab_name: str, methods: List[str]):
    """顯示集成異常檢測結果"""
    
    # 統計摘要
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("檢測方法數", len(methods))
    
    with col2:
        st.metric("數據點總數", len(results['kpi_data']))
    
    with col3:
        st.metric("集成異常點", len(results['ensemble_anomalies']))
    
    with col4:
        ensemble_rate = len(results['ensemble_anomalies']) / len(results['kpi_data']) * 100
        st.metric("集成異常率", f"{ensemble_rate:.2f}%")
    
    # 比較視覺化
    kpi_data = results['kpi_data']
    dates = pd.to_datetime(kpi_data['REPORT_TIME'])
    values = kpi_data['VALUE'].values
    
    # 創建比較圖表
    methods_results = {}
    for method, result in results['individual_results'].items():
        methods_results[method] = {
            'anomalies': result['anomalies'],
            'scores': result['scores']
        }
    
    from matplotlib_utils import create_comparison_plot
    fig = create_comparison_plot(
        dates=dates,
        values=values,
        methods_results=methods_results,
        title=f"{fab_name} - {kpi_name}"
    )
    
    render_matplotlib_figure(fig)
    
    # 方法一致性分析
    st.subheader("📊 方法一致性分析")
    
    consistency_data = []
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                anomalies1 = set(results['individual_results'][method1]['anomalies'])
                anomalies2 = set(results['individual_results'][method2]['anomalies'])
                
                intersection = len(anomalies1.intersection(anomalies2))
                union = len(anomalies1.union(anomalies2))
                jaccard = intersection / union if union > 0 else 0
                
                consistency_data.append({
                    '方法1': method1,
                    '方法2': method2,
                    '交集數量': intersection,
                    '並集數量': union,
                    'Jaccard相似度': f"{jaccard:.3f}"
                })
    
    if consistency_data:
        st.dataframe(pd.DataFrame(consistency_data))

def display_feature_importance_analysis(results: Dict):
    """顯示特徵重要性分析"""
    st.subheader("📈 特徵分析")
    
    feature_names = results['features_df'].columns.tolist()
    
    # 顯示特徵統計
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**特徵統計摘要:**")
        st.dataframe(results['features_df'].describe())
    
    with col2:
        st.write("**異常點的特徵特性:**")
        if len(results['anomaly_indices']) > 0:
            anomaly_features = results['features_df'].iloc[results['anomaly_indices']]
            normal_features = results['features_df'].iloc[~results['features_df'].index.isin(results['anomaly_indices'])]
            
            comparison_stats = []
            for col in feature_names:
                comparison_stats.append({
                    '特徵': col,
                    '異常點均值': f"{anomaly_features[col].mean():.3f}",
                    '正常點均值': f"{normal_features[col].mean():.3f}",
                    '差異': f"{abs(anomaly_features[col].mean() - normal_features[col].mean()):.3f}"
                })
            
            st.dataframe(pd.DataFrame(comparison_stats))

def display_cluster_analysis(results: Dict):
    """顯示聚類分析"""
    st.subheader("🎯 聚類分析詳情")
    
    labels = results['labels']
    unique_labels = np.unique(labels)
    
    cluster_stats = []
    for label in unique_labels:
        mask = labels == label
        cluster_size = np.sum(mask)
        
        if label == -1:
            cluster_stats.append({
                '聚類ID': '噪聲點',
                '點數量': cluster_size,
                '百分比': f"{cluster_size / len(labels) * 100:.1f}%",
                '類型': '異常'
            })
        else:
            cluster_stats.append({
                '聚類ID': f'聚類 {label}',
                '點數量': cluster_size,
                '百分比': f"{cluster_size / len(labels) * 100:.1f}%",
                '類型': '正常'
            })
    
    cluster_df = pd.DataFrame(cluster_stats)
    cluster_df = cluster_df.sort_values('點數量', ascending=False)
    st.dataframe(cluster_df)