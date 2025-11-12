"""
SAGA-SR评估脚本

功能:
- 批量评估测试集
- 计算LSD、SI-SDR等指标
- 生成评估报告
- 保存生成的音频样本
"""

import os
import json
import torch
import torchaudio
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np

from inference_saga_sr import SAGASRInference
from metrics import evaluate_audio_quality, compute_lsd, compute_si_sdr


class SAGASREvaluator:
    """
    SAGA-SR评估器
    
    用于批量评估测试集并生成详细报告
    """
    
    def __init__(self, 
                 model_checkpoint_path,
                 model_config_path,
                 device='cuda'):
        """
        Args:
            model_checkpoint_path: 训练好的模型权重
            model_config_path: 模型配置文件
            device: 设备
        """
        self.device = device
        
        # 创建推理器
        self.inferencer = SAGASRInference(
            model_checkpoint_path=model_checkpoint_path,
            model_config_path=model_config_path,
            device=device,
            use_caption=False  # 评估时不使用caption
        )
        
        print(f"Evaluator initialized on {device}")
    
    def evaluate_dataset(self,
                        test_audio_dir,
                        output_dir,
                        num_samples=None,
                        save_audio=True,
                        target_rolloff=16000.0,
                        num_steps=100):
        """
        评估整个测试集
        
        Args:
            test_audio_dir: 测试音频目录（高分辨率）
            output_dir: 输出目录
            num_samples: 评估样本数量（None=全部）
            save_audio: 是否保存生成的音频
            target_rolloff: 目标roll-off频率
            num_steps: 采样步数
        
        Returns:
            results: 评估结果字典
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        if save_audio:
            audio_output_dir = os.path.join(output_dir, 'generated_audio')
            os.makedirs(audio_output_dir, exist_ok=True)
        
        # 获取所有测试文件
        audio_files = []
        for ext in ['*.wav', '*.flac', '*.mp3']:
            audio_files.extend(Path(test_audio_dir).glob(ext))
        
        audio_files = sorted(audio_files)
        
        if num_samples is not None:
            audio_files = audio_files[:num_samples]
        
        print(f"Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            raise ValueError(f"No audio files found in {test_audio_dir}")
        
        # 评估每个文件
        results = {
            'metrics': [],
            'files': [],
            'summary': {}
        }
        
        for audio_file in tqdm(audio_files, desc="Evaluating"):
            try:
                # 加载高分辨率音频（ground truth）
                hr_audio, sr = torchaudio.load(str(audio_file))
                
                # 生成低分辨率输入（模拟）
                # 这里简化处理，实际应该用dataset.py中的低通滤波
                from dataset import SAGASRDataset
                dataset = SAGASRDataset(audio_dir=test_audio_dir, compute_rolloff=True)
                
                # 临时创建低分辨率版本
                lr_audio = dataset._apply_lowpass_filter(hr_audio)
                lr_audio_path = os.path.join(output_dir, f'temp_lr_{audio_file.name}')
                torchaudio.save(lr_audio_path, lr_audio, sr)
                
                # 推理生成高分辨率音频
                pred_hr_audio = self.inferencer.upsample(
                    input_audio_path=lr_audio_path,
                    output_audio_path=None,  # 不保存，直接返回
                    target_rolloff=target_rolloff,
                    num_steps=num_steps
                )
                
                # 删除临时文件
                os.remove(lr_audio_path)
                
                # 确保维度匹配
                if pred_hr_audio.dim() == 3:  # [B, C, T]
                    pred_hr_audio = pred_hr_audio[0]  # 取第一个batch
                if pred_hr_audio.dim() == 2:  # [C, T]
                    pred_hr_audio = pred_hr_audio.mean(0)  # 转单声道
                
                if hr_audio.dim() == 2:
                    hr_audio = hr_audio.mean(0)
                
                # 确保长度一致
                min_len = min(len(pred_hr_audio), len(hr_audio))
                pred_hr_audio = pred_hr_audio[:min_len]
                hr_audio = hr_audio[:min_len]
                
                # 计算评估指标
                metrics = evaluate_audio_quality(pred_hr_audio, hr_audio, sr=sr)
                metrics['file'] = audio_file.name
                
                results['metrics'].append(metrics)
                results['files'].append(str(audio_file))
                
                # 保存生成的音频
                if save_audio:
                    output_path = os.path.join(audio_output_dir, f'pred_{audio_file.name}')
                    torchaudio.save(output_path, pred_hr_audio.unsqueeze(0).cpu(), sr)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        # 计算汇总统计
        if len(results['metrics']) > 0:
            lsd_values = [m['lsd'] for m in results['metrics'] if not np.isnan(m['lsd'])]
            si_sdr_values = [m['si_sdr'] for m in results['metrics'] if not np.isnan(m['si_sdr'])]
            mse_values = [m['mse'] for m in results['metrics'] if not np.isnan(m['mse'])]
            
            results['summary'] = {
                'num_samples': len(results['metrics']),
                'lsd_mean': float(np.mean(lsd_values)) if lsd_values else float('nan'),
                'lsd_std': float(np.std(lsd_values)) if lsd_values else float('nan'),
                'lsd_median': float(np.median(lsd_values)) if lsd_values else float('nan'),
                'si_sdr_mean': float(np.mean(si_sdr_values)) if si_sdr_values else float('nan'),
                'si_sdr_std': float(np.std(si_sdr_values)) if si_sdr_values else float('nan'),
                'mse_mean': float(np.mean(mse_values)) if mse_values else float('nan'),
            }
        
        # 保存结果
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Evaluation Summary ===")
        print(f"Samples evaluated: {results['summary'].get('num_samples', 0)}")
        print(f"LSD (dB): {results['summary'].get('lsd_mean', 'N/A'):.4f} ± {results['summary'].get('lsd_std', 'N/A'):.4f}")
        print(f"SI-SDR (dB): {results['summary'].get('si_sdr_mean', 'N/A'):.2f} ± {results['summary'].get('si_sdr_std', 'N/A'):.2f}")
        print(f"Results saved to: {results_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='SAGA-SR Evaluation')
    
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config JSON')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test audio files (high-res)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None=all)')
    parser.add_argument('--save_audio', action='store_true',
                       help='Save generated audio files')
    parser.add_argument('--target_rolloff', type=float, default=16000.0,
                       help='Target roll-off frequency (Hz)')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of sampling steps')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = SAGASREvaluator(
        model_checkpoint_path=args.model_checkpoint,
        model_config_path=args.model_config,
        device=args.device
    )
    
    # 运行评估
    results = evaluator.evaluate_dataset(
        test_audio_dir=args.test_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        save_audio=args.save_audio,
        target_rolloff=args.target_rolloff,
        num_steps=args.num_steps
    )
    
    print("\n=== Evaluation Complete ===")
    print(f"Check {args.output_dir} for detailed results")


if __name__ == "__main__":
    main()
