import torch
from audiolm_pytorch import HubertWithKmeans
from audiolm_pytorch import SemanticTransformer, SemanticTransformerTrainer
from audiolm_pytorch import CoarseTransformer, CoarseTransformerTrainer
from audiolm_pytorch import SoundStream, FineTransformer, FineTransformerTrainer
from audiolm_pytorch import AudioLMSoundStream, AudioLM, MusicLMSoundStream
from musiclm_pytorch import MuLaNEmbedQuantizer
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
import gc  # 导入垃圾回收模块
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import librosa
import numpy as np
import pickle


nltk.download('punkt')

# 公共变量
checkpoint_path = 'hubert_base_ls960.pt'
kmeans_path = 'hubert_base_ls960_L9_km500.bin'

audio_output_dir = './downloaded_audios'
batch_size = 1
data_max_length = 320 * 32
num_train_steps = 1_000_000

audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 128,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
)

text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
)

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
)

# setup the quantizer with the namespaced conditioning embeddings, unique per quantizer as well as namespace (per transformer)

quantizer = MuLaNEmbedQuantizer(
    mulan = mulan,                          # pass in trained mulan from above
    conditioning_dims = (1024, 1024, 1024), # say all three transformers have model dimensions of 1024
    namespaces = ('semantic', 'coarse', 'fine')
)



# 函数：训练 FineTransformer
def train_fine_transformer(audio_data, combined_data):
    soundstream = MusicLMSoundStream()

    fine_transformer = FineTransformer(
            num_coarse_quantizers = 4,
            num_fine_quantizers = 8,
            codebook_size = 1024,
            dim = 1024,
            depth = 6,
            audio_text_condition = True
                    ).cuda()
    
    # 确保 Trainer 接收文本数据作为输入
    trainer = FineTransformerTrainer(
        transformer=fine_transformer,
        codec=soundstream, 
        # audio_data=audio_data, # 没有这个输入
        # text_data=combined_data, # 没有这个输入
        folder=audio_output_dir, 
        batch_size=batch_size, 
        data_max_length=data_max_length, 
        num_train_steps=num_train_steps,
        audio_conditioner = quantizer
        )    
    
    trainer.train()
    torch.save(fine_transformer.state_dict(), 'fine_transformer.pth')
    print("save fine_transformer.pth")    
    del fine_transformer, trainer, soundstream
    gc.collect()




# 加载处理后的数据
with open('processed_data/audio_data.pkl', 'rb') as f:
    audio_data = pickle.load(f)

with open('processed_data/text_data.pkl', 'rb') as f:
    text_data = pickle.load(f)

# 使用 audio_data 和 text_data 进行训练
train_fine_transformer(audio_data, text_data)
