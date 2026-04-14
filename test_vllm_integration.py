#!/usr/bin/env python3
"""
本地 vLLM 集成测试脚本

用法:
  python test_vllm_integration.py  # 默认测试 base 模型 (port 8000)
  python test_vllm_integration.py --model instruct  # 测试 instruct 模型 (port 8001)
"""

import argparse
import json
import sys
from generator import RAGGenerator
from config import VLLM_LOCAL_CONFIG, LOCAL_GEN_MODEL_CHOICES

def test_vllm_connection(model_name: str):
    """测试连接到本地 vLLM 服务"""
    print(f"\n{'='*60}")
    print(f"测试本地 vLLM 集成: {model_name}")
    print(f"{'='*60}\n")
    
    # 获取 vLLM 配置
    vllm_cfg = VLLM_LOCAL_CONFIG[model_name]
    print(f"[*] vLLM 配置:")
    print(f"    模型: {model_name}")
    print(f"    API endpoint: {vllm_cfg['base_url']}")
    print(f"    API key: {'***' if vllm_cfg['api_key'] else 'EMPTY'}\n")
    
    try:
        # 创建生成器
        print("[*] 初始化生成器...")
        generator = RAGGenerator(
            model_name=model_name,
            use_api=True,
            api_key=vllm_cfg["api_key"],
            api_base_url=vllm_cfg["base_url"],
            max_tokens=100,
            temperature=0.1
        )
        print("    ✓ 生成器初始化成功\n")
        
        # 测试单个查询
        print("[*] 测试单个查询...")
        question = "What is 2 + 2?"
        contexts = ["2 + 2 = 4"]
        
        print(f"    问题: {question}")
        print(f"    上下文: {contexts[0]}\n")
        
        answer = generator.generate(question, contexts)
        print(f"    [✓] 生成的答案: {answer}\n")
        
        # 测试多个查询
        print("[*] 测试多个查询...")
        test_cases = [
            ("What is the capital of France?", ["The capital of France is Paris."]),
            ("Who wrote Romeo and Juliet?", ["Shakespeare wrote Romeo and Juliet."]),
        ]
        
        results = []
        for q, ctx in test_cases:
            try:
                a = generator.generate(q, ctx)
                results.append({
                    "question": q,
                    "context": ctx[0][:50] + "..." if len(ctx[0]) > 50 else ctx[0],
                    "answer": a,
                    "status": "success"
                })
                print(f"    ✓ Q: {q[:40]}... => A: {a[:50]}...")
            except Exception as e:
                results.append({
                    "question": q,
                    "status": "error",
                    "error": str(e)
                })
                print(f"    ✗ Q: {q[:40]}... => Error: {str(e)[:50]}...")
        
        print(f"\n[✓] 测试完成！{len([r for r in results if r['status'] == 'success'])}/{len(results)} 成功\n")
        
        return True
        
    except Exception as e:
        print(f"\n[✗] 错误: {e}")
        print("\n提示:")
        print(f"  1. 确保 vLLM 服务在运行: python -m vllm.entrypoints.openai.api_server ...")
        print(f"  2. 检查 API endpoint 是否正确: {vllm_cfg['base_url']}")
        print(f"  3. 运行 launch_vllm_local.sh 来自动启动服务")
        return False

def main():
    parser = argparse.ArgumentParser(description="本地 vLLM 集成测试")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        choices=LOCAL_GEN_MODEL_CHOICES,
        help="要测试的模型"
    )
    args = parser.parse_args()
    
    success = test_vllm_connection(args.model)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
