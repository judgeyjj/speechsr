import torch

cache = torch.load("caption_cache.pt")
sample_path = "/data01/audio_group/m24_yuanjiajun/mdct_sr/data/wav48/train/p227/p227_002.wav"
print(cache[sample_path])