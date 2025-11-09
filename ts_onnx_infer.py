# 作者：白烁（https://space.bilibili.com/179281251）
import os
import pathlib
import numpy as np
import librosa
import onnxruntime as ort
import yaml
from tqdm import tqdm
import sys
from pathlib import Path
from tools.align_word import Word, WordList
import networks.g2p
from tools.alignment_decoder import AlignmentDecoder
from tools.config_utils import check_configs
from tools.export_tool import Exporter
from tools.post_processing import post_processing
from tools.audio_tools import remove_outliers_per_position
import click

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_onnx(session, input_dict):
    output_names = [output.name for output in session.get_outputs()]
    return dict(zip(output_names, session.run(output_names, input_dict)))

def create_session(onnx_path):
    providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), options, providers=providers)

def reconstruct_words_with_processed_phonemes(processed_data, ph_idx_to_word_idx, word_seq, ph_seq):
    """
    将处理后的音素还原到单词结构并更新时间戳
    参数:
        processed_data: 处理后的音素字典 {i: Phoneme}
        ph_idx_to_word_idx: 音素到单词的映射列表
        word_seq: 单词序列列表
        ph_seq: 音素序列列表
    返回:
        words: 更新后的单词列表
    """
    words = WordList()
    current_word = None
    pending_phonemes = []
    j = 0
    k = 0
    # 按原始顺序遍历所有音素索引
    for i, ph_idx in enumerate(ph_idx_to_word_idx):
        # 获取原始音素文本
        original_text = ph_seq[i]
        if original_text == 'SP':

            if current_word:
                current_word.end = processed_phoneme.end
                for phoneme in pending_phonemes:
                    current_word.append_phoneme(phoneme)
                words.append(current_word)
            current_word = Word(None, None, None)
            pending_phonemes = []
            continue
        # 获取处理后的音素
        processed_phoneme = processed_data.get(j)
        if k == ph_idx:
            current_word.start = processed_phoneme.start
            current_word.text = word_seq[k]
            k += 1
        j += 1
        if not processed_phoneme:
            continue
        pending_phonemes.append(processed_phoneme)
    # 特殊处理最后一个单词的时间精度
    if words:
        last_word = words[-1]
        last_phoneme = last_word.phonemes[-1]
        last_word.end = last_phoneme.end

    return words

@click.command()
@click.option("--onnx_folder", "-of", required=True, type=pathlib.Path, help="Path to ONNX models")
@click.option("--folder", "-f", default="segments", type=str, help="Input folder path")
@click.option("--g2p", "-g", default="Dictionary", type=str, help="G2P class name")
@click.option("--non_speech_phonemes", "-np", default="AP", type=str, help="non speech phonemes, exp. AP,EP")
@click.option("--save_confidence", "-sc", is_flag=True, help="Save confidence.csv")
@click.option("--language", "-l", default="zh", help="Dictionary language")
@click.option("--dictionary", "-d", type=pathlib.Path, help="Custom dictionary path")
def hfa_infer(onnx_folder,
          folder, 
          g2p, 
          non_speech_phonemes, 
          save_confidence, 
          language, 
          dictionary):
    onnx_folder = pathlib.Path(onnx_folder)
    check_configs(onnx_folder)
    with open(onnx_folder / 'VERSION', 'r', encoding='utf-8') as f:
        assert int(f.readline().strip()) >= 3, f"onnx model version must be greater than 3."

    vocab = load_yaml(onnx_folder / "vocab.yaml")
    non_speech_phonemes = [ph.strip() for ph in non_speech_phonemes.split(",") if ph.strip()]

    if "Dictionary" in g2p:
        if dictionary is None:
            dictionary = onnx_folder / vocab["dictionaries"].get(language, "")
        assert os.path.exists(dictionary), f"{pathlib.Path(dictionary).absolute()} does not exist"

    assert set(non_speech_phonemes).issubset(set(vocab['non_speech_phonemes'])), \
        f"The non_speech_phonemes contain elements that are not included in the vocab."

    if not g2p.endswith("G2P"):
        g2p += "G2P"
    g2p_class = getattr(networks.g2p, g2p)
    grapheme_to_phoneme = g2p_class(
        **{"language": language, "dictionary": dictionary, "non_speech_phonemes": non_speech_phonemes})
    dataset = grapheme_to_phoneme.get_dataset(pathlib.Path(folder).rglob("*.wav"))

    config = load_yaml(onnx_folder / 'config.yaml')
    vocab = load_yaml(onnx_folder / 'vocab.yaml')
    language_prefix = vocab.get("language_prefix", True)
    mel_cfg = config['melspec_config']

    # Create ONNX sessions
    model = create_session(onnx_folder / 'model.onnx')

    # Process dataset
    decoder = AlignmentDecoder(vocab, ["None", *non_speech_phonemes], mel_cfg["sample_rate"], mel_cfg["hop_size"])
    predictions = []
    ignored_phonemes = vocab['silent_phonemes'] + vocab['non_speech_phonemes']

    for i in tqdm(range(len(dataset)), desc="Processing", unit="it"):
        wav_path, ph_seq, word_seq, ph_idx_to_word_idx, language, non_speech_phonemes = dataset[i]
        ph_seq = [f"{language}/{ph}" if ph not in ignored_phonemes and language_prefix else ph for ph in ph_seq]

        # Load and resample audio
        original_wav, sr = librosa.load(wav_path, sr=mel_cfg['sample_rate'], mono=True)
        wav_length = len(original_wav) / mel_cfg['sample_rate']

        avg_result = True
        if avg_result:
            data = {}
            blank_basetime = 0.5
            # 添加空白音频片段（0-5秒均匀分布，共10次）

            for j in range(10):
                # 计算当前空白长度（0.0, 0.5, 1.0, ..., 4.5秒）
                blank_duration = j * blank_basetime
                # 创建空白音频片段
                blank_samples = int(blank_duration * mel_cfg['sample_rate'])
                blank_wave = np.zeros(blank_samples, dtype=original_wav.dtype)
                # 拼接空白片段和原始音频
                processed_wav = np.concatenate([blank_wave, original_wav])
                processed_length = len(processed_wav) / mel_cfg['sample_rate']
            
                results = run_onnx(model, {'waveform': [processed_wav]})
                words, confidence = decoder.decode(
                    results['ph_frame_logits'],
                    results['ph_edge_logits'],
                    results['cvnt_logits'],
                    processed_length, ph_seq, word_seq, ph_idx_to_word_idx,
                    non_speech_phonemes=non_speech_phonemes
                )
                words = [word for word in words if word.text not in ['AP', 'SP', 'EP']]
                data[j] = [phoneme for word in words for phoneme in word.phonemes]

            n_phonemes = len(data[0])
            processed_data = {}
            for i in range(n_phonemes):
                # 收集同一位置的所有start/end时间
                start_times = [data[j][i].start - j*blank_basetime for j in range(10)]
                end_times = [data[j][i].end - j*blank_basetime for j in range(10)]
                # 处理离群值并计算平均值
                processed_starts = remove_outliers_per_position([start_times])[0]
                processed_ends = remove_outliers_per_position([end_times])[0]
                # 保留原始文本信息
                sample_phoneme = data[0][i]
                # 创建处理后的音素对象
                sample_phoneme.start = processed_starts
                sample_phoneme.end = processed_ends
                processed_data[i] = sample_phoneme
            words = reconstruct_words_with_processed_phonemes(processed_data, ph_idx_to_word_idx, word_seq, ph_seq)
        else:
            results = run_onnx(model, {'waveform': [original_wav]})
            words, confidence = decoder.decode(
                results['ph_frame_logits'],
                results['ph_edge_logits'],
                results['cvnt_logits'],
                wav_length, ph_seq, word_seq, ph_idx_to_word_idx,
                non_speech_phonemes=non_speech_phonemes
            )
        words.clear_language_prefix()
        predictions.append((wav_path, wav_length, words, confidence))

    predictions, log = post_processing(predictions)
    if log:
        print("error:", "\n".join(log))

    Exporter(predictions=predictions).export(['textgrid', 'confidence'] if save_confidence else ['textgrid'])
    print("Output files are saved to the same folder as the input wav files.")


if __name__ == '__main__':
    hfa_infer()
