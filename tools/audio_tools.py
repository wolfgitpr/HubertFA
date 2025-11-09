import librosa
import pathlib
import numpy as np
import pathlib  
from scipy.stats import median_abs_deviation 

def wav_total_length(wavs):
    #return hours
    total_length = 0.0
    if isinstance(wavs, list):
        for file in wavs:
            wave_seconds = librosa.get_duration(filename=str(file))
            total_length += wave_seconds / 3600.
        return total_length
    
    else:
        wav_path = pathlib.Path(wavs)
        if wav_path.is_file() and wav_path.suffix == '.wav':
            return librosa.get_duration(filename=str(wav_path)) / 3600.
        elif wav_path.is_dir():
            for ch in wav_path.iterdir():
                if ch.is_file() and ch.suffix == '.wav':
                    total_length += wav_total_length(ch)
            return total_length

def remove_outliers_per_position(data_series, threshold=1.5):
    """
    使用中位数绝对偏差(MAD)方法去除离群值
    参数:
        data_series -- 每个位置的时间戳列表
        threshold -- MAD阈值
    返回: 处理后的平均值列表
    """
    processed_values = []
    for position_values in data_series:
        if not position_values:
            processed_values.append(0.0)  # 空位置填0
            continue
            
        med = np.median(position_values)  
        mad_val = median_abs_deviation(position_values)
        
        # 处理MAD=0的情况（所有值相同）
        if mad_val == 0:  
            processed_values.append(med)
            print(f"MAD=0警告: 所有值相同，直接使用中位数 {med:.8f}")
            continue
            
        # 计算修正Z-score
        z_scores = np.abs((np.array(position_values) - med) / (mad_val * 1.4826))
        
        # 分离有效值和离群值
        retained_values = []
        filtered_out = []
        for x, z in zip(position_values, z_scores):
            if z <= threshold:
                retained_values.append(x)
            else:
                filtered_out.append(x)
        
        print(f"中位数 {med:.8f}")
        #print(f"  保留值: {retained_values} (数量 {len(retained_values)})")
        print(f"  过滤值: {filtered_out} (数量 {len(filtered_out)})")
        
        if len(retained_values) > 0:
            final_value = np.mean(retained_values)
        else:
            final_value = med
            
        processed_values.append(final_value)

    return processed_values