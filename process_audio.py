# process_audio.py
import librosa
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import torch
import os
import pickle

nltk.download('punkt')

def load_audio_file(file_path, sr=22050):
    # 加载音频文件
    # 'sr' 是采样率，22050 是常用值
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def audio_to_melspectrogram(audio, sr=22050, n_mels=128, hop_length=512):
    # 将音频转换为梅尔频谱图
    melspec = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    return melspec


def process_audio_files(file_paths, sr=22050, n_mels=128, hop_length=512):
    processed_audios = []
    valid_indices = []  # 用于存储有效音频的索引

    for index, file_path in enumerate(file_paths):
        try:
            # 尝试加载和预处理音频文件
            audio = load_audio_file(file_path, sr=sr)
            melspec = audio_to_melspectrogram(audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
            processed_audios.append(melspec)
            valid_indices.append(index)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return processed_audios, valid_indices



# 简单的文本预处理函数
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 基本标记化
    tokens = word_tokenize(text)
    return tokens

def main():
    # 加载 .csv 文件
    csv_file = 'musiccaps-public.csv'
    df = pd.read_csv(csv_file)

    # 提取字段作为文本数据

    captions = df['caption']
    aspect_list = df['aspect_list']

    # 应用预处理
    preprocessed_captions = [preprocess_text(text) for text in captions]
    preprocessed_aspects = [preprocess_text(text) for text in aspect_list]

    #  并行结合
    combined_data = list(zip(preprocessed_captions, preprocessed_aspects))



    # 提取音频文件名
    audio_filenames = df['ytid'].tolist()
    # 构建音频文件的完整路径
    audio_file_paths = [f'./downloaded_audios/{filename}.wav' for filename in audio_filenames]
    
    # 初始化计数器
    existing_files_count = 0

    # 检查每个音频文件是否存在
    for file_path in audio_file_paths:
        if os.path.exists(file_path):
            existing_files_count += 1

    print(f"Total audio files in CSV: {len(audio_filenames)}")
    print(f"Existing audio files: {existing_files_count}")
    
    
    # 预处理音频数据，并获取有效音频文件的索引
    audio_data, valid_audio_indices = process_audio_files(audio_file_paths)

    # 根据有效音频文件的索引来同步文本数据
    synced_text_data = [combined_data[i] for i in valid_audio_indices]


    # 确保长度相同, audio_data 和 combined_data 是一一对应的
    # assert len(audio_data) == len(synced_text_data)
    
    # 保存处理后的数据
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    with open('processed_data/audio_data.pkl', 'wb') as f:
        pickle.dump(audio_data, f)

    with open('processed_data/text_data.pkl', 'wb') as f:
        pickle.dump(synced_text_data, f)

if __name__ == "__main__":
    main()

