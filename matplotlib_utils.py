import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_anomaly_plot(dates: np.ndarray, values: np.ndarray, 
                        anomalies: np.ndarray, title: str,
                        method: str = "Z-Score", scores: Optional[np.ndarray] = None,
                        threshold: float = 2.0, show_threshold: bool = True,
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """創建異常檢測圖表
    
    Parameters:
    -----------
    dates : np.ndarray - 時間序列
    values : np.ndarray - 數值序列
    anomalies : np.ndarray - 異常點索引
    title : str - 圖表標題
    method : str - 檢測方法名稱
    scores : np.ndarray - 異常分數
    threshold : float - 閾值
    show_threshold : bool - 是否顯示閾值線
    figsize : tuple - 圖表大小
    
    Returns:
    --------
    plt.Figure - matplotlib figure對象
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # 轉換日期格式
    if isinstance(dates[0], (np.datetime64, pd.Timestamp)):
        dates = pd.to_datetime(dates)
    
    # 主圖 - 時間序列與異常點
    ax1.plot(dates, values, 'b-', linewidth=1.5, label='原始數據', alpha=0.7)
    
    # 標記異常點
    if len(anomalies) > 0:
        ax1.scatter(dates[anomalies], values[anomalies], 
                   color='red', s=50, zorder=5, label=f'異常點 (n={len(anomalies)})')
        
        # 為每個異常點添加垂直線和陰影區域
        for idx in anomalies:
            ax1.axvline(x=dates[idx], color='red', alpha=0.2, linestyle='--')
            
            # 計算異常程度並顯示
            if scores is not None and idx < len(scores):
                severity = scores[idx]
                if severity > threshold * 1.5:
                    ax1.annotate(f'{severity:.1f}', 
                               xy=(dates[idx], values[idx]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color='darkred')
    
    # 添加均值線和標準差區域
    mean_val = np.mean(values)
    std_val = np.std(values)
    ax1.axhline(y=mean_val, color='green', linestyle='--', alpha=0.5, label=f'均值: {mean_val:.2f}')
    ax1.fill_between(dates, mean_val - std_val, mean_val + std_val, 
                     alpha=0.2, color='green', label=f'±1σ 區間')
    
    if show_threshold and method == "Z-Score":
        ax1.fill_between(dates, mean_val - threshold*std_val, mean_val + threshold*std_val, 
                        alpha=0.1, color='orange', label=f'±{threshold}σ 閾值')
    
    ax1.set_title(f'{title} - {method} 異常檢測', fontsize=14, fontweight='bold')
    ax1.set_ylabel('數值', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 格式化x軸日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//20)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 子圖 - 異常分數
    if scores is not None:
        ax2.plot(dates, scores, 'g-', linewidth=1, label='異常分數')
        ax2.fill_between(dates, 0, scores, alpha=0.3, color='green')
        
        # 標記超過閾值的點
        if show_threshold:
            ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'閾值: {threshold}')
            exceed_threshold = scores > threshold
            ax2.fill_between(dates, 0, scores, where=exceed_threshold, 
                           alpha=0.5, color='red', label='超過閾值')
        
        # 標記異常點位置
        if len(anomalies) > 0:
            ax2.scatter(dates[anomalies], scores[anomalies], 
                       color='red', s=30, zorder=5)
    
    ax2.set_xlabel('時間', fontsize=12)
    ax2.set_ylabel('異常分數', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//20)))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def create_zscore_analysis_plot(dates: np.ndarray, values: np.ndarray,
                               z_scores: np.ndarray, threshold: float,
                               outliers: np.ndarray, title: str) -> plt.Figure:
    """創建Z-Score分析圖表"""
    fig = plt.figure(figsize=(14, 8))
    
    # 創建網格布局
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # 主時間序列圖
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, values, 'b-', linewidth=1.5, label='原始數據')
    
    # 計算統計量
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # 繪製統計區間
    ax1.axhline(y=mean_val, color='green', linestyle='--', label=f'μ = {mean_val:.2f}')
    ax1.fill_between(dates, mean_val - std_val, mean_val + std_val, 
                     alpha=0.2, color='green', label='μ ± σ')
    ax1.fill_between(dates, mean_val - 2*std_val, mean_val + 2*std_val, 
                     alpha=0.1, color='yellow', label='μ ± 2σ')
    ax1.fill_between(dates, mean_val - threshold*std_val, mean_val + threshold*std_val, 
                     alpha=0.05, color='red', label=f'μ ± {threshold}σ (閾值)')
    
    # 標記異常點
    if len(outliers) > 0:
        ax1.scatter(dates[outliers], values[outliers], 
                   color='red', s=100, zorder=5, marker='x', linewidths=2,
                   label=f'異常點 ({len(outliers)}個, {len(outliers)/len(values)*100:.1f}%)')
    
    ax1.set_title(f'{title} - Z-Score 異常檢測分析', fontsize=14, fontweight='bold')
    ax1.set_ylabel('數值', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Z-Score 時間序列
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(dates, z_scores, 'g-', linewidth=1, label='Z-Score')
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'閾值 = {threshold}')
    ax2.axhline(y=-threshold, color='red', linestyle='--')
    ax2.fill_between(dates, -threshold, threshold, alpha=0.1, color='green', label='正常範圍')
    
    # 標記異常Z-Score
    if len(outliers) > 0:
        ax2.scatter(dates[outliers], z_scores[outliers], 
                   color='red', s=50, zorder=5)
    
    ax2.set_ylabel('Z-Score', fontsize=12)
    ax2.set_xlabel('時間', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Z-Score 分布直方圖
    ax3 = fig.add_subplot(gs[2, 0])
    n, bins, patches = ax3.hist(z_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    
    # 標記異常區域
    for i, patch in enumerate(patches):
        if abs(bins[i]) > threshold:
            patch.set_facecolor('red')
            patch.set_alpha(0.7)
    
    ax3.axvline(x=threshold, color='red', linestyle='--', label=f'閾值 = ±{threshold}')
    ax3.axvline(x=-threshold, color='red', linestyle='--')
    ax3.set_xlabel('Z-Score', fontsize=12)
    ax3.set_ylabel('頻率', fontsize=12)
    ax3.set_title('Z-Score 分布', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 異常統計信息
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    # 計算統計信息
    stats_text = f"""異常檢測統計：
    
總數據點：{len(values)}
異常點數：{len(outliers)}
異常率：{len(outliers)/len(values)*100:.2f}%

統計參數：
均值 (μ)：{mean_val:.2f}
標準差 (σ)：{std_val:.2f}
閾值：±{threshold}σ

異常點分布：
超過 +{threshold}σ：{np.sum(z_scores > threshold)}
低於 -{threshold}σ：{np.sum(z_scores < -threshold)}
最大 Z-Score：{np.max(np.abs(z_scores)):.2f}"""
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_iqr_analysis_plot(dates: np.ndarray, values: np.ndarray,
                            outliers: np.ndarray, iqr_multiplier: float,
                            title: str) -> plt.Figure:
    """創建IQR分析圖表"""
    fig = plt.figure(figsize=(14, 8))
    
    # 計算IQR統計量
    Q1 = np.percentile(values, 25)
    Q2 = np.percentile(values, 50)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    # 創建子圖
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # 主時間序列圖
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, values, 'b-', linewidth=1.5, label='原始數據', alpha=0.7)
    
    # 繪製IQR區間
    ax1.axhline(y=Q2, color='green', linestyle='-', label=f'中位數 Q2 = {Q2:.2f}')
    ax1.axhline(y=Q1, color='orange', linestyle='--', label=f'Q1 = {Q1:.2f}')
    ax1.axhline(y=Q3, color='orange', linestyle='--', label=f'Q3 = {Q3:.2f}')
    ax1.fill_between(dates, Q1, Q3, alpha=0.2, color='yellow', label='IQR區間')
    
    # 繪製異常判定邊界
    ax1.axhline(y=lower_bound, color='red', linestyle=':', label=f'下界 = {lower_bound:.2f}')
    ax1.axhline(y=upper_bound, color='red', linestyle=':', label=f'上界 = {upper_bound:.2f}')
    ax1.fill_between(dates, lower_bound, upper_bound, alpha=0.1, color='green', label='正常範圍')
    
    # 標記異常點
    if len(outliers) > 0:
        ax1.scatter(dates[outliers], values[outliers], 
                   color='red', s=100, zorder=5, marker='x', linewidths=2,
                   label=f'異常點 ({len(outliers)}個)')
        
        # 為嚴重異常點添加標註
        for idx in outliers:
            if values[idx] > upper_bound + IQR or values[idx] < lower_bound - IQR:
                ax1.annotate(f'{values[idx]:.1f}', 
                           xy=(dates[idx], values[idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='darkred',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_title(f'{title} - IQR 異常檢測分析', fontsize=14, fontweight='bold')
    ax1.set_ylabel('數值', fontsize=12)
    ax1.legend(loc='best', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 箱線圖
    ax2 = fig.add_subplot(gs[1, 0])
    box = ax2.boxplot(values, vert=True, patch_artist=True, 
                      widths=0.7, showfliers=True)
    
    # 設置箱線圖顏色
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][0].set_alpha(0.7)
    box['medians'][0].set_color('red')
    box['medians'][0].set_linewidth(2)
    
    # 添加異常點
    if len(outliers) > 0:
        ax2.scatter(np.ones(len(outliers)), values[outliers], 
                   color='red', s=50, zorder=5, marker='x')
    
    ax2.set_ylabel('數值', fontsize=12)
    ax2.set_title('數據分布箱線圖', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 統計信息
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # 計算額外統計信息
    outliers_above = np.sum(values[outliers] > upper_bound) if len(outliers) > 0 else 0
    outliers_below = np.sum(values[outliers] < lower_bound) if len(outliers) > 0 else 0
    
    stats_text = f"""IQR 異常檢測統計：
    
數據點總數：{len(values)}
異常點數量：{len(outliers)}
異常率：{len(outliers)/len(values)*100:.2f}%

四分位數：
Q1 (25%)：{Q1:.2f}
Q2 (50%)：{Q2:.2f}
Q3 (75%)：{Q3:.2f}
IQR：{IQR:.2f}

異常邊界：
倍數：{iqr_multiplier}×IQR
下界：{lower_bound:.2f}
上界：{upper_bound:.2f}

異常分布：
高於上界：{outliers_above}
低於下界：{outliers_below}"""
    
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    return fig

def create_isolation_forest_plot(dates: np.ndarray, values: np.ndarray,
                                anomaly_scores: np.ndarray, predictions: np.ndarray,
                                contamination: float, title: str) -> plt.Figure:
    """創建Isolation Forest分析圖表"""
    fig = plt.figure(figsize=(14, 10))
    
    # 找出異常點
    anomalies = np.where(predictions == -1)[0]
    normal = np.where(predictions == 1)[0]
    
    # 創建子圖布局
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # 主時間序列圖
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, values, 'b-', linewidth=1.5, label='原始數據', alpha=0.6)
    
    # 標記異常點和正常點
    if len(normal) > 0:
        ax1.scatter(dates[normal], values[normal], 
                   color='green', s=20, alpha=0.5, label='正常點')
    if len(anomalies) > 0:
        ax1.scatter(dates[anomalies], values[anomalies], 
                   color='red', s=60, marker='x', linewidths=2,
                   label=f'異常點 ({len(anomalies)}個, {len(anomalies)/len(values)*100:.1f}%)')
    
    ax1.set_title(f'{title} - Isolation Forest 異常檢測', fontsize=14, fontweight='bold')
    ax1.set_ylabel('數值', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 異常分數時間序列
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(dates, anomaly_scores, 'g-', linewidth=1, label='異常分數')
    ax2.fill_between(dates, anomaly_scores.min(), anomaly_scores, 
                     alpha=0.3, color='green')
    
    # 標記異常點的分數
    if len(anomalies) > 0:
        ax2.scatter(dates[anomalies], anomaly_scores[anomalies], 
                   color='red', s=50, zorder=5, label='異常點分數')
    
    # 添加分數閾值線
    threshold = np.percentile(anomaly_scores, contamination * 100)
    ax2.axhline(y=threshold, color='red', linestyle='--', 
               label=f'閾值 ({contamination*100:.1f}%分位數)')
    
    ax2.set_ylabel('異常分數', fontsize=12)
    ax2.set_xlabel('時間', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 異常分數分布直方圖
    ax3 = fig.add_subplot(gs[2, 0])
    n, bins, patches = ax3.hist(anomaly_scores, bins=30, alpha=0.7, 
                                color='blue', edgecolor='black')
    
    # 標記異常區域
    ax3.axvline(x=threshold, color='red', linestyle='--', 
               label=f'閾值 = {threshold:.3f}')
    
    # 為異常分數區域著色
    for i, patch in enumerate(patches):
        if bins[i] < threshold:
            patch.set_facecolor('red')
            patch.set_alpha(0.7)
    
    ax3.set_xlabel('異常分數', fontsize=12)
    ax3.set_ylabel('頻率', fontsize=12)
    ax3.set_title('異常分數分布', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 統計信息和參數
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    # 計算統計信息
    mean_score = np.mean(anomaly_scores)
    std_score = np.std(anomaly_scores)
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)
    
    stats_text = f"""Isolation Forest 統計：
    
模型參數：
污染率：{contamination*100:.1f}%
預期異常數：{int(len(values) * contamination)}
實際異常數：{len(anomalies)}

異常分數統計：
平均分數：{mean_score:.4f}
標準差：{std_score:.4f}
最小分數：{min_score:.4f}
最大分數：{max_score:.4f}
閾值：{threshold:.4f}

檢測結果：
正常點：{len(normal)} ({len(normal)/len(values)*100:.1f}%)
異常點：{len(anomalies)} ({len(anomalies)/len(values)*100:.1f}%)"""
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_dbscan_plot(dates: np.ndarray, values: np.ndarray,
                      labels: np.ndarray, eps: float, min_samples: int,
                      title: str) -> plt.Figure:
    """創建DBSCAN聚類分析圖表"""
    fig = plt.figure(figsize=(14, 10))
    
    # 識別聚類和噪聲點
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    noise_points = np.where(labels == -1)[0]
    
    # 創建顏色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters + 1))
    
    # 創建子圖布局
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # 主時間序列圖
    ax1 = fig.add_subplot(gs[0, :])
    
    # 繪製每個聚類
    for i, label in enumerate(unique_labels):
        if label == -1:
            # 噪聲點（異常點）
            mask = labels == label
            ax1.scatter(dates[mask], values[mask], 
                       color='red', s=60, marker='x', linewidths=2,
                       label=f'異常點 ({np.sum(mask)}個)', zorder=5)
        else:
            # 正常聚類
            mask = labels == label
            ax1.plot(dates[mask], values[mask], 'o', 
                    color=colors[i], markersize=6, alpha=0.7,
                    label=f'聚類 {label} ({np.sum(mask)}個)')
    
    # 連接時間序列
    ax1.plot(dates, values, 'k-', linewidth=0.5, alpha=0.3)
    
    ax1.set_title(f'{title} - DBSCAN 異常檢測', fontsize=14, fontweight='bold')
    ax1.set_ylabel('數值', fontsize=12)
    ax1.legend(loc='best', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 滑動窗口密度圖
    ax2 = fig.add_subplot(gs[1, :])
    window_size = max(min_samples * 2, 10)
    densities = []
    
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2)
        window_values = values[start:end]
        density = len(window_values) / (np.ptp(window_values) + 1e-10)
        densities.append(density)
    
    densities = np.array(densities)
    ax2.plot(dates, densities, 'b-', linewidth=1, label='局部密度')
    ax2.fill_between(dates, 0, densities, alpha=0.3, color='blue')
    
    # 標記異常點位置
    if len(noise_points) > 0:
        ax2.scatter(dates[noise_points], densities[noise_points], 
                   color='red', s=30, zorder=5, label='異常點位置')
    
    ax2.set_ylabel('局部密度', fontsize=12)
    ax2.set_xlabel('時間', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 聚類大小分布
    ax3 = fig.add_subplot(gs[2, 0])
    cluster_sizes = []
    cluster_names = []
    
    for label in unique_labels:
        if label != -1:
            size = np.sum(labels == label)
            cluster_sizes.append(size)
            cluster_names.append(f'聚類{label}')
    
    if len(noise_points) > 0:
        cluster_sizes.append(len(noise_points))
        cluster_names.append('異常點')
    
    bars = ax3.bar(cluster_names, cluster_sizes, 
                   color=['green']*len(cluster_names[:-1]) + ['red'])
    
    # 添加數值標籤
    for bar, size in zip(bars, cluster_sizes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{size}', ha='center', va='bottom', fontsize=10)
    
    ax3.set_xlabel('聚類', fontsize=12)
    ax3.set_ylabel('點數', fontsize=12)
    ax3.set_title('聚類分布', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 統計信息
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    # 計算統計信息
    anomaly_rate = len(noise_points) / len(values) * 100
    
    stats_text = f"""DBSCAN 聚類統計：
    
算法參數：
eps (鄰域半徑)：{eps:.3f}
min_samples (最小點數)：{min_samples}

聚類結果：
聚類數量：{n_clusters}
異常點數：{len(noise_points)}
異常率：{anomaly_rate:.2f}%

數據統計：
總數據點：{len(values)}
最大聚類：{max(cluster_sizes[:-1]) if n_clusters > 0 else 0}個點
最小聚類：{min(cluster_sizes[:-1]) if n_clusters > 0 else 0}個點
平均聚類大小：{np.mean(cluster_sizes[:-1]) if n_clusters > 0 else 0:.1f}"""
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_comparison_plot(dates: np.ndarray, values: np.ndarray,
                         methods_results: Dict[str, Dict],
                         title: str) -> plt.Figure:
    """創建多方法比較圖表
    
    Parameters:
    -----------
    methods_results : dict
        格式: {'方法名': {'anomalies': array, 'scores': array, 'threshold': float}}
    """
    n_methods = len(methods_results)
    fig = plt.figure(figsize=(14, 4 * n_methods))
    
    for i, (method_name, result) in enumerate(methods_results.items()):
        ax = plt.subplot(n_methods, 1, i + 1)
        
        # 繪製原始數據
        ax.plot(dates, values, 'b-', linewidth=1, alpha=0.6, label='原始數據')
        
        # 標記異常點
        anomalies = result.get('anomalies', np.array([]))
        if len(anomalies) > 0:
            ax.scatter(dates[anomalies], values[anomalies], 
                      color='red', s=50, marker='x', linewidths=2,
                      label=f'異常點 ({len(anomalies)}個)')
        
        # 添加方法特定的信息
        scores = result.get('scores')
        if scores is not None:
            ax2 = ax.twinx()
            ax2.plot(dates, scores, 'g-', linewidth=0.5, alpha=0.5)
            ax2.set_ylabel('異常分數', color='g', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='g')
        
        ax.set_title(f'{method_name} - 檢測到 {len(anomalies)} 個異常點', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('數值', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if i == n_methods - 1:
            ax.set_xlabel('時間', fontsize=10)
    
    plt.suptitle(f'{title} - 多方法異常檢測比較', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def render_matplotlib_figure(fig: plt.Figure):
    """在Streamlit中渲染matplotlib圖表"""
    st.pyplot(fig)
    plt.close(fig)  # 釋放內存