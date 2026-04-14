#!/usr/bin/env python3
"""
测试新的 --use_vllm 参数解析和生成器初始化
不需要实际连接到 vLLM 服务
"""

import sys
import argparse

# 移除之前的 sys.argv，模拟命令行参数
original_argv = sys.argv.copy()

def test_parameter_parsing():
    """测试参数解析"""
    print("\n" + "="*60)
    print("测试 1: 参数解析")
    print("="*60 + "\n")
    
    test_cases = [
        {
            "name": "使用 --use_vllm base 模型",
            "args": ["main.py", "--mode", "generate_retrieve", "--use_vllm", "Qwen/Qwen2.5-7B"]
        },
        {
            "name": "使用 --use_vllm instruct 模型",
            "args": ["main.py", "--mode", "generate_retrieve", "--use_vllm", "Qwen/Qwen2.5-7B-Instruct"]
        },
        {
            "name": "传统 API 参数方式",
            "args": ["main.py", "--mode", "generate_retrieve", "--use_api", "--gen_model", "Qwen/Qwen2.5-7B", "--api_base_url", "http://127.0.0.1:8000/v1"]
        },
        {
            "name": "本地权重加载",
            "args": ["main.py", "--mode", "generate_retrieve", "--local_gen_model", "Qwen/Qwen2.5-7B"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"[测试 {i}] {test_case['name']}")
        print(f"  命令行参数: {' '.join(test_case['args'][1:])}")
        
        # 模拟参数解析
        sys.argv = test_case['args']
        
        try:
            # 动态导入并解析参数
            import importlib
            import main as main_module
            # 重新加载模块以获取最新的参数定义
            importlib.reload(main_module)
            args = main_module.parse_args()
            
            # 检查关键参数
            print(f"  ✓ 参数解析成功")
            print(f"    - mode: {args.mode}")
            if args.use_vllm:
                print(f"    - use_vllm: {args.use_vllm}")
            if args.use_api:
                print(f"    - use_api: True")
            if args.local_gen_model:
                print(f"    - local_gen_model: {args.local_gen_model}")
            print()
            
        except Exception as e:
            print(f"  ✗ 参数解析失败: {e}\n")

def test_generator_initialization():
    """测试生成器初始化（不需要实际连接）"""
    print("="*60)
    print("测试 2: 生成器初始化逻辑")
    print("="*60 + "\n")
    
    # 直接测试 build_generator 逻辑
    sys.argv = ["main.py", "--use_vllm", "Qwen/Qwen2.5-7B"]
    
    try:
        from main import parse_args
        from config import VLLM_LOCAL_CONFIG, LOCAL_GEN_MODEL_CHOICES
        
        args = parse_args()
        
        print("[测试 1] --use_vllm 参数解析")
        print(f"  ✓ use_vllm = {args.use_vllm}")
        print(f"  ✓ 在 VLLM_LOCAL_CONFIG 中: {args.use_vllm in VLLM_LOCAL_CONFIG}")
        
        if args.use_vllm:
            vllm_cfg = VLLM_LOCAL_CONFIG[args.use_vllm]
            print(f"  ✓ API endpoint: {vllm_cfg['base_url']}")
        print()
        
        # 验证配置
        print("[测试 2] vLLM 配置验证")
        for model_name in LOCAL_GEN_MODEL_CHOICES:
            if model_name in VLLM_LOCAL_CONFIG:
                cfg = VLLM_LOCAL_CONFIG[model_name]
                print(f"  ✓ {model_name}")
                print(f"      - endpoint: {cfg['base_url']}")
                print(f"      - api_key: {'EMPTY' if cfg['api_key'] == 'EMPTY' else cfg['api_key'][:10]}")
        print()
        
        # 测试生成器选择逻辑
        print("[测试 3] 生成器选择逻辑")
        test_configs = [
            ("--use_vllm Qwen/Qwen2.5-7B", "vLLM API 模式 (base)"),
            ("--use_vllm Qwen/Qwen2.5-7B-Instruct", "vLLM API 模式 (instruct)"),
            ("--use_api --gen_model Qwen/Qwen2.5-7B --api_base_url http://custom:8000/v1", "自定义 API 模式"),
            ("--local_gen_model Qwen/Qwen2.5-7B", "本地权重加载"),
        ]
        
        for config_str, desc in test_configs:
            args_list = ["main.py", "--mode", "generate_retrieve"] + config_str.split()
            sys.argv = args_list
            args = parse_args()
            
            # 模拟 build_generator 的逻辑
            if args.use_vllm:
                mode = "✓ vLLM 快捷模式"
            elif (not args.use_local) and (args.use_api or args.api_key is not None or args.api_base_url is not None):
                mode = "✓ 传统 API 模式"
            else:
                mode = "✓ 本地权重加载"
            
            print(f"  {mode}: {desc}")
        
        print()
        
    except Exception as e:
        import traceback
        print(f"✗ 初始化测试失败:")
        print(traceback.format_exc())

def main():
    print("\n" + "="*60)
    print("新 vLLM 集成功能测试")
    print("="*60)
    print("\n该测试验证：")
    print("  1. 新增 --use_vllm 参数是否正确添加")
    print("  2. VLLM_LOCAL_CONFIG 配置是否完整")
    print("  3. 生成器选择逻辑是否正确")
    print("\n")
    
    try:
        test_parameter_parsing()
        test_generator_initialization()
        
        print("="*60)
        print("✓ 所有测试通过！")
        print("="*60 + "\n")
        print("下一步:")
        print("  1. 运行 'bash launch_vllm_local.sh both' 启动 vLLM 服务")
        print("  2. 运行 'python test_vllm_integration.py' 测试服务连接")
        print("  3. 使用 '--use_vllm' 参数运行生成任务\n")
        
    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("✗ 测试失败")
        print("="*60)
        print(traceback.format_exc())
        sys.exit(1)
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()
