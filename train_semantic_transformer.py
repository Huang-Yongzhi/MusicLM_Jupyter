import torch
from audiolm_pytorch import HubertWithKmeans
from audiolm_pytorch import SemanticTransformer, SemanticTransformerTrainer
from audiolm_pytorch import CoarseTransformer, CoarseTransformerTrainer
from audiolm_pytorch import SoundStream, FineTransformer, FineTransformerTrainer
from audiolm_pytorch import AudioLMSoundStream, AudioLM
import gc  # 导入垃圾回收模块
from musiclm_pytorch import MuLaNEmbedQuantizer
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer

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

# now say you want the conditioning embeddings for semantic transformer

wavs = torch.randn(2, 1024)
conds = quantizer(wavs = wavs, namespace = 'semantic') # (2, 8, 1024) - 8 is number of quantizers

# 公共变量
checkpoint_path = 'hubert_base_ls960.pt'
kmeans_path = 'hubert_base_ls960_L9_km500.bin'

audio_output_dir = './downloaded_audios'
batch_size = 1
data_max_length = 320 * 32
num_train_steps = 1_000

# 函数：训练 SemanticTransformer
def train_semantic_transformer(): 
    wav2vec = HubertWithKmeans(
        checkpoint_path=checkpoint_path, 
        kmeans_path=kmeans_path
        )   # 每个函数中重新创建 wav2vec，后面会删掉
    
    
    semantic_transformer = SemanticTransformer(
        num_semantic_tokens=wav2vec.codebook_size, 
        dim=1024, 
        depth=6, 
        audio_text_condition=True
        ).cuda()
    
    trainer = SemanticTransformerTrainer(
        transformer=semantic_transformer, 
        wav2vec=wav2vec, 
        audio_conditioner=quantizer, 
        folder=audio_output_dir, 
        batch_size=batch_size, 
        data_max_length=data_max_length, 
        num_train_steps=num_train_steps
        )
    
    trainer.train()
    torch.save(semantic_transformer.state_dict(), 'semantic_transformer.pth')
    print("save semantic_transformer.pth")
    del semantic_transformer, trainer, wav2vec
    gc.collect()  # 执行垃圾回收



# 依次训练每个模型
train_semantic_transformer()
