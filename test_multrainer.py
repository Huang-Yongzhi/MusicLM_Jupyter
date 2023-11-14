import torch
import multiprocessing
from audiolm_pytorch import HubertWithKmeans
from audiolm_pytorch import SemanticTransformer, SemanticTransformerTrainer
from audiolm_pytorch import CoarseTransformer, CoarseTransformerTrainer
from audiolm_pytorch import FineTransformer, FineTransformerTrainer
from musiclm_pytorch import MuLaNEmbedQuantizer, MusicLMSoundStream
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
import gc  # 导入垃圾回收模块

# 公共变量
checkpoint_path = 'hubert_base_ls960.pt'
kmeans_path = 'hubert_base_ls960_L9_km500.bin'
audio_output_dir = './downloaded_audios'
batch_size = 1
data_max_length = 320 * 32
num_train_steps = 1_000_000

# 函数：训练 SemanticTransformer
def train_semantic_transformer(): 
    # 初始化 wav2vec 模型
    wav2vec = HubertWithKmeans(
        checkpoint_path=checkpoint_path, 
        kmeans_path=kmeans_path
    )   

    # 初始化 SemanticTransformer 模型
    if torch.cuda.is_available():
        semantic_transformer = SemanticTransformer(
            num_semantic_tokens=wav2vec.codebook_size, 
            dim=1024, 
            depth=6, 
            audio_text_condition=True
        ).cuda()
    else:
        semantic_transformer = SemanticTransformer(
            num_semantic_tokens=wav2vec.codebook_size, 
            dim=1024, 
            depth=6, 
            audio_text_condition=True
        )
    
    # 初始化 SemanticTransformerTrainer
    trainer = SemanticTransformerTrainer(
        transformer=semantic_transformer, 
        wav2vec=wav2vec, 
        audio_conditioner=quantizer, 
        folder=audio_output_dir, 
        batch_size=batch_size, 
        data_max_length=data_max_length, 
        num_train_steps=num_train_steps
    )
    
    # 训练 SemanticTransformer 模型
    trainer.train()
    torch.save(semantic_transformer.state_dict(), 'semantic_transformer.pth')
    print("save semantic_transformer.pth")
    
    # 删除 trainer 和调用垃圾回收
    del trainer, semantic_transformer, wav2vec, mulan, audio_transformer, text_transformer, quantizer
    gc.collect()

# 函数：训练 CoarseTransformer
def train_coarse_transformer():  
    # 初始化 wav2vec 模型
    wav2vec = HubertWithKmeans(
        checkpoint_path=checkpoint_path, 
        kmeans_path=kmeans_path
    )
    
    # 初始化音频流
    soundstream = MusicLMSoundStream()

    # 初始化 CoarseTransformer 模型
    if torch.cuda.is_available():
        coarse_transformer = CoarseTransformer(
            num_semantic_tokens=wav2vec.codebook_size, 
            codebook_size=1024, 
            num_coarse_quantizers=4, 
            dim=1024, 
            depth=6, 
            audio_text_condition=True
        ).cuda()
    else: 
        coarse_transformer = CoarseTransformer(
            num_semantic_tokens=wav2vec.codebook_size, 
            codebook_size=1024, 
            num_coarse_quantizers=4, 
            dim=1024, 
            depth=6, 
            audio_text_condition=True
        )
    
    # 初始化 CoarseTransformerTrainer
    trainer = CoarseTransformerTrainer(
        transformer=coarse_transformer, 
        codec=soundstream, 
        wav2vec=wav2vec, 
        audio_conditioner=quantizer, 
        folder=audio_output_dir, 
        batch_size=batch_size, 
        data_max_length=data_max_length, 
        num_train_steps=num_train_steps
    )
    
    # 训练 CoarseTransformer 模型
    trainer.train()
    torch.save(coarse_transformer.state_dict(), 'coarse_transformer.pth')
    print("save coarse_transformer.pth")
    
    # 删除 trainer 和调用垃圾回收
    del trainer, coarse_transformer, wav2vec, soundstream, quantizer
    gc.collect()

# 函数：训练 FineTransformer
def train_fine_transformer():
    # 初始化音频流
    soundstream = MusicLMSoundStream()

    # 初始化 FineTransformer 模型
    if torch.cuda.is_available():
        fine_transformer = FineTransformer(
            num_coarse_quantizers=4,
            num_fine_quantizers=8,
            codebook_size=1024,
            dim=1024,
            depth=6,
            audio_text_condition=True
        ).cuda()
    else:
        fine_transformer = FineTransformer(
            num_coarse_quantizers=4,
            num_fine_quantizers=8,
            codebook_size=1024,
            dim=1024,
            depth=6,
            audio_text_condition=True
        )
    
    # 确保 Trainer 接收文本数据作为输入
    trainer = FineTransformerTrainer(
        transformer=fine_transformer,
        codec=soundstream, 
        folder=audio_output_dir, 
        batch_size=batch_size, 
        data_max_length=data_max_length, 
        num_train_steps=num_train_steps,
        audio_conditioner=quantizer
    )    
    
    # 训练 FineTransformer 模型
    trainer.train()
    torch.save(fine_transformer.state_dict(), 'fine_transformer.pth')
    print("save fine_transformer.pth")    
    
    # 删除 trainer 和调用垃圾回收
    del trainer, fine_transformer, soundstream, quantizer
    gc.collect()

if __name__ == '__main__':
    # 设置多处理
    multiprocessing.set_start_method('spawn')

    # 创建多个进程来并行训练
    processes = []
    processes.append(multiprocessing.Process(target=train_semantic_transformer))
    processes.append(multiprocessing.Process(target=train_coarse_transformer))
    processes.append(multiprocessing.Process(target=train_fine_transformer))

    # 启动并等待所有进程完成
    for process in processes:
        process.start()

    for process in processes:
        process.join()
