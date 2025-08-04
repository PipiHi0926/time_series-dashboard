import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RealisticFABDataGenerator:
    """生成真實的FAB KPI數據"""
    
    def __init__(self, start_date: str = "2023-01-01", days: int = 365):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.days = days
        self.dates = [self.start_date + timedelta(days=i) for i in range(days)]
        
        # FAB 特性定義
        self.fab_profiles = {
            'FAB12A': {
                'maturity': 'mature',
                'technology': '28nm',
                'stability_factor': 0.95,
                'process_complexity': 'high'
            },
            'FAB14B': {
                'maturity': 'ramping',
                'technology': '16nm', 
                'stability_factor': 0.85,
                'process_complexity': 'very_high'
            },
            'FAB18C': {
                'maturity': 'development',
                'technology': '7nm',
                'stability_factor': 0.75,
                'process_complexity': 'extreme'
            },
            'FAB22D': {
                'maturity': 'legacy',
                'technology': '65nm',
                'stability_factor': 0.98,
                'process_complexity': 'medium'
            }
        }
        
        # KPI 類型定義
        self.kpi_types = {
            # 連續型 KPI
            'Yield': {
                'type': 'continuous_percentage',
                'base_range': (85, 95),
                'noise_level': 'low',
                'trend_sensitivity': 'high',
                'seasonal_pattern': 'weekly'
            },
            'Throughput': {
                'type': 'continuous_count',
                'base_range': (500, 1200),
                'noise_level': 'medium',
                'trend_sensitivity': 'medium',
                'seasonal_pattern': 'daily'
            },
            'Cycle_Time': {
                'type': 'continuous_duration',
                'base_range': (120, 180),
                'noise_level': 'medium',
                'trend_sensitivity': 'high',
                'seasonal_pattern': 'none'
            },
            'Equipment_Utilization': {
                'type': 'continuous_percentage',
                'base_range': (75, 95),
                'noise_level': 'low',
                'trend_sensitivity': 'medium',
                'seasonal_pattern': 'weekly'
            },
            'Cost_Per_Unit': {
                'type': 'continuous_currency',
                'base_range': (15, 25),
                'noise_level': 'high',
                'trend_sensitivity': 'low',
                'seasonal_pattern': 'monthly'
            },
            
            # 計數型 KPI (泊松分布)
            'Defect_Count': {
                'type': 'count_poisson',
                'lambda_range': (2, 8),
                'noise_level': 'high',
                'trend_sensitivity': 'high',
                'seasonal_pattern': 'none'
            },
            'Critical_Alerts': {
                'type': 'count_poisson',
                'lambda_range': (0.5, 3),
                'noise_level': 'very_high',
                'trend_sensitivity': 'high',
                'seasonal_pattern': 'none'
            },
            'Maintenance_Events': {
                'type': 'count_poisson',
                'lambda_range': (1, 4),
                'noise_level': 'medium',
                'trend_sensitivity': 'medium',
                'seasonal_pattern': 'weekly'
            },
            
            # 二元型 KPI
            'Line_Down_Flag': {
                'type': 'binary',
                'probability_range': (0.02, 0.08),
                'noise_level': 'high',
                'trend_sensitivity': 'very_high',
                'seasonal_pattern': 'none'
            },
            'Quality_Pass_Flag': {
                'type': 'binary',
                'probability_range': (0.92, 0.98),
                'noise_level': 'low',
                'trend_sensitivity': 'medium',
                'seasonal_pattern': 'none'
            },
            
            # 稀疏事件 KPI
            'Equipment_Failure': {
                'type': 'sparse_event',
                'event_rate': 0.01,  # 每天1%機率
                'noise_level': 'very_high',
                'trend_sensitivity': 'very_high',
                'seasonal_pattern': 'none'
            },
            'Process_Excursion': {
                'type': 'sparse_event',
                'event_rate': 0.005,  # 每天0.5%機率
                'noise_level': 'very_high',
                'trend_sensitivity': 'very_high',
                'seasonal_pattern': 'none'
            }
        }
    
    def generate_realistic_fab_data(self) -> pd.DataFrame:
        """生成真實的 FAB 數據"""
        all_data = []
        
        for fab_name, fab_profile in self.fab_profiles.items():
            for kpi_name, kpi_config in self.kpi_types.items():
                kpi_data = self._generate_kpi_timeseries(fab_name, fab_profile, kpi_name, kpi_config)
                all_data.extend(kpi_data)
        
        df = pd.DataFrame(all_data)
        return df.sort_values(['FAB', 'REPORT_TIME', 'KPI']).reset_index(drop=True)
    
    def _generate_kpi_timeseries(self, fab_name: str, fab_profile: Dict, 
                               kpi_name: str, kpi_config: Dict) -> List[Dict]:
        """為特定 FAB 和 KPI 生成時序數據"""
        data = []
        
        # 基礎參數
        stability = fab_profile['stability_factor']
        complexity_factor = self._get_complexity_factor(fab_profile['process_complexity'])
        
        if kpi_config['type'] == 'continuous_percentage':
            values = self._generate_continuous_percentage(kpi_config, stability, complexity_factor)
        elif kpi_config['type'] == 'continuous_count':
            values = self._generate_continuous_count(kpi_config, stability, complexity_factor)
        elif kpi_config['type'] == 'continuous_duration':
            values = self._generate_continuous_duration(kpi_config, stability, complexity_factor)
        elif kpi_config['type'] == 'continuous_currency':
            values = self._generate_continuous_currency(kpi_config, stability, complexity_factor)
        elif kpi_config['type'] == 'count_poisson':
            values = self._generate_poisson_counts(kpi_config, stability, complexity_factor)
        elif kpi_config['type'] == 'binary':
            values = self._generate_binary_events(kpi_config, stability, complexity_factor)
        elif kpi_config['type'] == 'sparse_event':
            values = self._generate_sparse_events(kpi_config, stability, complexity_factor)
        else:
            values = [0] * self.days
        
        # 添加製程事件
        values = self._add_process_events(values, fab_profile, kpi_config)
        
        # 組裝數據
        for i, (date, value) in enumerate(zip(self.dates, values)):
            data.append({
                'FAB': fab_name,
                'KPI': kpi_name,
                'REPORT_TIME': date,
                'VALUE': value
            })
        
        return data
    
    def _get_complexity_factor(self, complexity: str) -> float:
        """根據製程複雜度調整變異係數"""
        factors = {
            'medium': 1.0,
            'high': 1.2,
            'very_high': 1.5,
            'extreme': 2.0
        }
        return factors.get(complexity, 1.0)
    
    def _generate_continuous_percentage(self, config: Dict, stability: float, complexity: float) -> List[float]:
        """生成連續百分比型 KPI"""
        base_min, base_max = config['base_range']
        base_mean = (base_min + base_max) / 2
        base_std = (base_max - base_min) / 6 * complexity
        
        # 基礎趨勢
        trend = self._generate_trend_component(stability)
        
        # 季節性模式
        seasonal = self._generate_seasonal_component(config['seasonal_pattern'])
        
        # 噪聲
        noise_level = self._get_noise_level(config['noise_level']) * complexity
        noise = np.random.normal(0, base_std * noise_level, self.days)
        
        # 組合並限制範圍
        values = base_mean + trend + seasonal + noise
        values = np.clip(values, base_min - 5, base_max + 5)
        
        return values.tolist()
    
    def _generate_continuous_count(self, config: Dict, stability: float, complexity: float) -> List[float]:
        """生成連續計數型 KPI"""
        base_min, base_max = config['base_range']
        base_mean = (base_min + base_max) / 2
        base_std = (base_max - base_min) / 4 * complexity
        
        trend = self._generate_trend_component(stability) * base_mean * 0.1
        seasonal = self._generate_seasonal_component(config['seasonal_pattern']) * base_mean * 0.05
        
        noise_level = self._get_noise_level(config['noise_level']) * complexity
        noise = np.random.normal(0, base_std * noise_level, self.days)
        
        values = base_mean + trend + seasonal + noise
        values = np.maximum(values, 0)  # 確保非負
        
        return values.tolist()
    
    def _generate_continuous_duration(self, config: Dict, stability: float, complexity: float) -> List[float]:
        """生成連續時間型 KPI (如週期時間)"""
        base_min, base_max = config['base_range']
        base_mean = (base_min + base_max) / 2
        
        # 時間型 KPI 通常呈右偏分布
        shape = 2.0 / complexity  # 複雜度越高越右偏
        scale = base_mean / shape
        
        values = np.random.gamma(shape, scale, self.days)
        
        # 添加製程改善趨勢
        improvement_trend = -np.linspace(0, base_mean * 0.1 * stability, self.days)
        values += improvement_trend
        
        # 添加週期性維護影響
        maintenance_cycle = np.sin(2 * np.pi * np.arange(self.days) / 28) * base_mean * 0.05
        values += maintenance_cycle
        
        values = np.maximum(values, base_min * 0.8)
        
        return values.tolist()
    
    def _generate_continuous_currency(self, config: Dict, stability: float, complexity: float) -> List[float]:
        """生成連續貨幣型 KPI"""
        base_min, base_max = config['base_range']
        base_mean = (base_min + base_max) / 2
        base_std = (base_max - base_min) / 6
        
        # 成本通常有通膨趨勢
        inflation_trend = np.linspace(0, base_mean * 0.05, self.days)
        
        # 學習曲線效應 (成本隨時間降低)
        learning_curve = -np.log(1 + np.arange(self.days) / 365) * base_mean * 0.02 * stability
        
        # 季節性採購週期
        seasonal = self._generate_seasonal_component(config['seasonal_pattern']) * base_mean * 0.03
        
        noise_level = self._get_noise_level(config['noise_level']) * complexity
        noise = np.random.normal(0, base_std * noise_level, self.days)
        
        values = base_mean + inflation_trend + learning_curve + seasonal + noise
        values = np.maximum(values, base_min * 0.5)
        
        return values.tolist()
    
    def _generate_poisson_counts(self, config: Dict, stability: float, complexity: float) -> List[int]:
        """生成泊松分布計數型 KPI"""
        lambda_min, lambda_max = config['lambda_range']
        base_lambda = (lambda_min + lambda_max) / 2
        
        # 時變 lambda 參數
        trend = self._generate_trend_component(stability) * base_lambda * 0.2
        seasonal = self._generate_seasonal_component(config['seasonal_pattern']) * base_lambda * 0.1
        
        # 確保 lambda > 0
        lambdas = np.maximum(base_lambda + trend + seasonal, 0.1)
        
        # 生成泊松分布數據
        values = [np.random.poisson(lam) for lam in lambdas]
        
        return values
    
    def _generate_binary_events(self, config: Dict, stability: float, complexity: float) -> List[int]:
        """生成二元事件型 KPI"""
        prob_min, prob_max = config['probability_range']
        base_prob = (prob_min + prob_max) / 2
        
        # 時變概率
        trend = self._generate_trend_component(stability) * base_prob * 0.3
        seasonal = self._generate_seasonal_component(config['seasonal_pattern']) * base_prob * 0.1
        
        # 確保概率在 [0, 1] 範圍內
        probs = np.clip(base_prob + trend + seasonal, 0.001, 0.999)
        
        # 生成二元數據
        values = [np.random.binomial(1, p) for p in probs]
        
        return values
    
    def _generate_sparse_events(self, config: Dict, stability: float, complexity: float) -> List[int]:
        """生成稀疏事件型 KPI"""
        base_rate = config['event_rate']
        
        # 考慮製程成熟度影響事件頻率
        adjusted_rate = base_rate * (2 - stability) * complexity
        
        # 生成事件
        values = [1 if np.random.random() < adjusted_rate else 0 for _ in range(self.days)]
        
        # 添加群聚效應 (事件往往連續發生)
        for i in range(1, len(values)):
            if values[i-1] == 1 and np.random.random() < 0.3:
                values[i] = 1
        
        return values
    
    def _generate_trend_component(self, stability: float) -> np.ndarray:
        """生成趨勢成分"""
        # 基礎線性趨勢
        linear_trend = np.linspace(0, 1, self.days) * (1 - stability) * 2
        
        # 添加隨機遊走
        random_walk = np.cumsum(np.random.normal(0, 0.01, self.days)) * (1 - stability)
        
        # 製程改善階段變化
        improvement_phases = self._add_improvement_phases()
        
        return linear_trend + random_walk + improvement_phases
    
    def _generate_seasonal_component(self, pattern: str) -> np.ndarray:
        """生成季節性成分"""
        if pattern == 'none':
            return np.zeros(self.days)
        elif pattern == 'daily':
            # 工作日效應
            weekdays = np.array([(self.start_date + timedelta(days=i)).weekday() for i in range(self.days)])
            weekend_effect = np.where(weekdays >= 5, -0.2, 0.1)
            return weekend_effect
        elif pattern == 'weekly':
            # 週循環
            return 0.5 * np.sin(2 * np.pi * np.arange(self.days) / 7)
        elif pattern == 'monthly':
            # 月循環
            return 0.3 * np.sin(2 * np.pi * np.arange(self.days) / 30)
        else:
            return np.zeros(self.days)
    
    def _get_noise_level(self, level: str) -> float:
        """取得噪聲水準"""
        levels = {
            'very_low': 0.01,
            'low': 0.05,
            'medium': 0.1,
            'high': 0.2,
            'very_high': 0.5
        }
        return levels.get(level, 0.1)
    
    def _add_improvement_phases(self) -> np.ndarray:
        """添加製程改善階段"""
        phases = np.zeros(self.days)
        
        # 隨機添加 2-4 個改善階段
        n_phases = np.random.randint(2, 5)
        phase_starts = sorted(np.random.choice(range(30, self.days-30), n_phases, replace=False))
        
        for start in phase_starts:
            # 改善持續 10-30 天
            duration = np.random.randint(10, 31)
            end = min(start + duration, self.days)
            
            # 改善幅度
            improvement = np.random.uniform(0.1, 0.3)
            phases[start:end] += improvement
        
        return phases
    
    def _add_process_events(self, values: List[float], fab_profile: Dict, kpi_config: Dict) -> List[float]:
        """添加製程事件影響"""
        values = np.array(values, dtype=float)  # 確保為 float 類型
        
        # 設備故障事件
        if np.random.random() < 0.1:  # 10% 機率發生設備故障
            failure_day = np.random.randint(0, self.days)
            impact_duration = np.random.randint(1, 5)
            impact_magnitude = np.random.uniform(0.1, 0.5)
            
            end_day = min(failure_day + impact_duration, self.days)
            
            if kpi_config['type'] in ['continuous_percentage', 'binary']:
                values[failure_day:end_day] *= (1 - impact_magnitude)
            else:
                values[failure_day:end_day] *= (1 + impact_magnitude)
        
        # 製程改善事件
        if np.random.random() < 0.15:  # 15% 機率發生製程改善
            improvement_day = np.random.randint(self.days // 2, self.days)
            improvement_magnitude = np.random.uniform(0.05, 0.2)
            
            if kpi_config['type'] in ['continuous_percentage', 'binary']:
                values[improvement_day:] *= (1 + improvement_magnitude)
            else:
                values[improvement_day:] *= (1 - improvement_magnitude)
        
        return values.tolist()

def generate_realistic_fab_sample_data() -> pd.DataFrame:
    """生成真實的 FAB 範例數據"""
    generator = RealisticFABDataGenerator(start_date="2023-01-01", days=365)
    return generator.generate_realistic_fab_data()

if __name__ == "__main__":
    # 測試數據生成
    df = generate_realistic_fab_sample_data()
    print(f"Generated {len(df)} data points")
    print(f"FABs: {df['FAB'].unique()}")
    print(f"KPIs: {df['KPI'].unique()}")
    print("\nSample data:")
    print(df.head(10))