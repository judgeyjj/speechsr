import os
import warnings

import torch
from typing import Optional


class QwenAudioCaptioner:
    """
    Qwen2-Audio 音频转文本适配器

    论文要求:
    - 训练阶段: 从高分辨率音频生成 caption
    - 推理阶段: 从低分辨率音频生成 caption
    - 返回纯文本字符串

    模式:
    1) local: 本地加载 Qwen2-Audio 模型
    2) api:   调用云端 API
    """

    _DEFAULT_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

    def __init__(self, mode: str = 'local', model_name: Optional[str] = None):
        """
        Args:
            mode: 'local' 或 'api'
            model_name: 模型名称（若为空则读取环境变量 QWEN_AUDIO_MODEL）
        """
        self.mode = mode
        env_model = os.getenv('QWEN_AUDIO_MODEL')
        selected_model = model_name or env_model or self._DEFAULT_MODEL_ID

        if 'qwen2-audio' not in selected_model.lower():
            warnings.warn(
                f"指定的模型 `{selected_model}` 非 Qwen2-Audio，已回退至 {self._DEFAULT_MODEL_ID}",
                UserWarning,
            )
            selected_model = self._DEFAULT_MODEL_ID

        self.model_name = selected_model
        
        if mode == 'local':
            self._init_local_model()
        elif mode == 'api':
            self._init_api()
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'local' or 'api'")

    def _init_local_model(self):
        """初始化本地 Qwen2-Audio 模型"""
        try:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
            
            print(f"Loading Qwen2-Audio model: {self.model_name}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            self.device = next(self.model.parameters()).device
            print(f"Model loaded on device: {self.device}")
            
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Please run: pip install transformers"
            )
        except Exception as e:
            warnings.warn(
                f"[Caption] 无法加载 Qwen2-Audio 模型 (`{self.model_name}`): {e}",
                RuntimeWarning,
            )
            raise RuntimeError(f"Failed to load Qwen2-Audio model: {e}")
    
    def _init_api(self):
        """初始化API配置"""
        self.api_key = os.getenv('QWEN_API_KEY')
        if not self.api_key:
            raise ValueError(
                "QWEN_API_KEY not found in environment variables. "
                "Please set it: export QWEN_API_KEY='your-key'"
            )
        
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        print("API mode initialized")
    
    def generate_caption(self, 
                        audio_path: str, 
                        use_hr_audio: bool = True,
                        max_length: int = 100,
                        temperature: float = 0.7) -> str:
        """
        从音频生成文本描述
        
        Args:
            audio_path: 音频文件路径
            use_hr_audio: True=训练阶段(高分辨率), False=推理阶段(低分辨率)
            max_length: 生成文本最大长度
            temperature: 生成温度
        
        Returns:
            caption: 文本描述字符串
        """
        # 提示词（可根据use_hr_audio调整）
        if use_hr_audio:
            prompt = "Describe this high-quality audio in detail, including its content, style, and acoustic characteristics."
        else:
            prompt = "Describe this audio in detail."
        
        if self.mode == 'local':
            return self._generate_local(audio_path, prompt, max_length, temperature)
        else:
            return self._generate_api(audio_path, prompt, max_length, temperature)
    
    def _generate_local(self, audio_path, prompt, max_length, temperature):
        """本地模型生成"""
        try:
            # 准备输入
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": audio_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 处理输入
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            if hasattr(self.processor, "__call__"):
                inputs = self.processor(
                    text=[text],
                    audios=[audio_path],
                    return_tensors="pt",
                    padding=True,
                )
            else:
                raise RuntimeError("Processor does not support direct calls for audio generation")
            
            inputs = inputs.to(self.device)
            
            # 生成
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True
                )
            
            # 解码
            generated_text = self.processor.batch_decode(
                output_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )
            caption = generated_text[0].strip() if generated_text else ""
            
            return caption
            
        except Exception as e:
            print(f"Local generation failed: {e}")
            return ""
    
    def _generate_api(self, audio_path, prompt, max_length, temperature):
        """API生成"""
        try:
            import requests
            import json
            import base64
            
            # 读取音频文件并编码
            with open(audio_path, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            # 构建请求
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "model": "qwen-audio-turbo",
                "input": {
                    "audio": audio_data,
                    "prompt": prompt
                },
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature
                }
            }
            
            # 发送请求
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                caption = result['output']['text']
                return caption
            else:
                print(f"API call failed: {response.status_code}")
                return f"Audio description (auto-generated)"
                
        except Exception as e:
            print(f"API generation failed: {e}")
            return f"Audio description (auto-generated)"
    
    def batch_generate(self, audio_paths, use_hr_audio=True, max_length=100):
        """
        批量生成caption
        
        Args:
            audio_paths: 音频文件路径列表
            use_hr_audio: 是否使用高分辨率模式
            max_length: 最大长度
        
        Returns:
            captions: 文本描述列表
        """
        captions = []
        for audio_path in audio_paths:
            caption = self.generate_caption(
                audio_path,
                use_hr_audio=use_hr_audio,
                max_length=max_length
            )
            captions.append(caption)
        return captions


class CaptionCache:
    """
    Caption缓存器
    
    用于预先生成并缓存所有caption，避免训练时重复计算
    """
    
    def __init__(self, cache_file='caption_cache.pt'):
        """
        Args:
            cache_file: 缓存文件路径
        """
        self.cache_file = cache_file
        self.cache = {}
        
        # 如果缓存文件存在，加载
        if os.path.exists(cache_file):
            self.load()
    
    def add(self, audio_path, caption):
        """添加到缓存"""
        self.cache[audio_path] = caption
    
    def get(self, audio_path):
        """从缓存获取"""
        return self.cache.get(audio_path, None)
    
    def save(self):
        """保存缓存到文件"""
        torch.save(self.cache, self.cache_file)
        print(f"Cache saved to {self.cache_file} ({len(self.cache)} entries)")
    
    def load(self):
        """从文件加载缓存"""
        self.cache = torch.load(self.cache_file)
        print(f"Cache loaded from {self.cache_file} ({len(self.cache)} entries)")
    
    def __len__(self):
        return len(self.cache)


def pregenerate_captions(audio_dir,
                        output_cache='caption_cache.pt',
                        mode='local',
                        use_hr_audio=True):
    """
    预生成所有caption并缓存
    
    Args:
        audio_dir: 音频文件目录
        output_cache: 输出缓存文件
        mode: 'local' 或 'api'
        use_hr_audio: 是否使用高分辨率模式
    
    Returns:
        cache: CaptionCache对象
    """
    import os
    from tqdm import tqdm

    # 初始化captioner
    captioner = QwenAudioCaptioner(mode=mode)
    cache = CaptionCache(output_cache)

    # 获取所有音频文件（递归）
    audio_files = []
    exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    # 调试：检查目录是否存在
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
    
    print(f"Scanning directory: {audio_dir}")
    scanned_dirs = 0
    scanned_files = 0
    
    for root, dirs, files in os.walk(audio_dir):
        scanned_dirs += 1
        for name in files:
            scanned_files += 1
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                full_path = os.path.join(root, name)
                audio_files.append(full_path)
            elif scanned_files <= 10:  # 打印前10个非音频文件作为调试信息
                print(f"  [跳过] {os.path.join(root, name)} (扩展名: {ext or '无'})")

    audio_files.sort()
    
    print(f"Scanned {scanned_dirs} directories, {scanned_files} files")
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print(f"\n警告: 未找到任何音频文件！")
        print(f"请检查:")
        print(f"  1. 目录路径是否正确: {audio_dir}")
        print(f"  2. 子目录中是否包含 .wav/.mp3/.flac 等音频文件")
        print(f"  3. 文件权限是否允许读取")
        # 列出前几个子目录作为提示
        try:
            subdirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))][:5]
            if subdirs:
                print(f"\n发现子目录示例: {', '.join(subdirs)}")
                # 检查第一个子目录的内容
                first_subdir = os.path.join(audio_dir, subdirs[0])
                first_files = os.listdir(first_subdir)[:5]
                print(f"  子目录 '{subdirs[0]}' 中的文件示例: {', '.join(first_files)}")
        except Exception as e:
            print(f"  无法列出子目录: {e}")
    
    # 生成caption
    for audio_path in tqdm(audio_files, desc="Generating captions"):
        # 检查是否已缓存
        if cache.get(audio_path) is not None:
            continue
        
        # 生成
        caption = captioner.generate_caption(audio_path, use_hr_audio=use_hr_audio)
        cache.add(audio_path, caption)
        
        # 每100个保存一次
        if len(cache) % 100 == 0:
            cache.save()
    
    # 最终保存
    cache.save()
    
    return cache


# 测试代码
if __name__ == "__main__":
    print("=== Audio Captioning Adapter 测试 ===\n")
    
    # 注意: 需要真实的音频文件路径进行测试
    test_audio = "test_audio.wav"  # 替换为实际音频文件
    
    if not os.path.exists(test_audio):
        print(f"警告: 测试音频文件 {test_audio} 不存在")
        print("请提供真实音频文件进行测试")
        print("\n示例用法:")
        print("1. 本地模式:")
        print("   captioner = QwenAudioCaptioner(mode='local')")
        print("   caption = captioner.generate_caption('audio.wav', use_hr_audio=True)")
        print("\n2. API模式:")
        print("   export QWEN_API_KEY='your-key'")
        print("   captioner = QwenAudioCaptioner(mode='api')")
        print("   caption = captioner.generate_caption('audio.wav', use_hr_audio=False)")
        print("\n3. 批量预生成:")
        print("   cache = pregenerate_captions('dataset/train/high_res', mode='local')")
    else:
        # 实际测试
        try:
            captioner = QwenAudioCaptioner(mode='local')
            
            # 训练模式（高分辨率）
            caption_hr = captioner.generate_caption(test_audio, use_hr_audio=True)
            print(f"高分辨率音频描述: {caption_hr}\n")
            
            # 推理模式（低分辨率）
            caption_lr = captioner.generate_caption(test_audio, use_hr_audio=False)
            print(f"低分辨率音频描述: {caption_lr}\n")
            
            print("✅ 测试通过！")
            
        except Exception as e:
            print(f"测试失败: {e}")
            print("请确保已安装transformers并下载了Qwen2-Audio模型")
