#!/usr/bin/env python
# coding: utf-8

"""
快速测试TFLite模型的脚本
用于验证转换后的模型是否可以正常加载和推理
"""

import os
import sys
import numpy as np
import tensorflow as tf
import time
import re


def load_input_from_tmplog(tmplog_path):
    """从tmplog.txt文件中加载输入数据"""
    if not os.path.exists(tmplog_path):
        print(f"⚠️  Warning: tmplog file not found at {tmplog_path}")
        return None
    
    with open(tmplog_path, 'r') as f:
        lines = f.readlines()
    
    # 解析第一行的完整输入数组
    if len(lines) < 1:
        print("⚠️  Warning: tmplog file is empty")
        return None
    
    first_line = lines[0].strip()
    if first_line.startswith("Full input array:"):
        # 从第二行或同一行提取数据
        data_line = first_line.replace("Full input array:", "").strip()
        if not data_line and len(lines) > 1:
            data_line = lines[1].strip()
    else:
        data_line = first_line
    
    # 提取所有浮点数
    # 匹配格式如 +0.581000, -0.085986 等
    numbers = re.findall(r'[+-]?\d+\.\d+', data_line)
    
    if not numbers:
        print("⚠️  Warning: Could not parse numbers from tmplog file")
        return None
    
    # 转换为numpy数组
    input_array = np.array([float(num) for num in numbers], dtype=np.float32)
    
    print(f"✅ Loaded input data from {tmplog_path}")
    print(f"   Input dimension: {len(input_array)}")
    print(f"   First 5 values: {input_array[:5]}")
    
    return input_array


def test_tflite_model(model_path, num_tests=10, fixed_input=None):
    """测试TFLite模型"""
    
    print("=" * 60)
    print("TFLite Model Test")
    print("=" * 60)
    print(f"Model path: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return False
    
    file_size = os.path.getsize(model_path) / 1024  # KB
    print(f"Model size: {file_size:.2f} KB")
    
    # 加载TFLite模型
    print("\nLoading TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # 获取输入输出信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nModel Information:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")
    
    input_shape = input_details[0]['shape']
    input_dim = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
    
    output_shape = output_details[0]['shape']
    output_dim = output_shape[-1] if len(output_shape) > 1 else output_shape[0]
    
    # 准备测试输入
    if fixed_input is not None:
        if len(fixed_input) != input_dim:
            print(f"⚠️  Warning: Fixed input dimension ({len(fixed_input)}) doesn't match model input dimension ({input_dim})")
            print(f"   Adjusting input...")
            if len(fixed_input) > input_dim:
                fixed_input = fixed_input[:input_dim]
            else:
                # 填充零
                fixed_input = np.pad(fixed_input, (0, input_dim - len(fixed_input)), mode='constant')
        test_input_base = fixed_input.reshape(1, input_dim)
        print(f"✅ Using fixed input from tmplog (dimension: {input_dim})")
    else:
        test_input_base = None
        print(f"⚠️  Using random input (dimension: {input_dim})")
    
    # 测试推理
    print(f"\nRunning {num_tests} inference tests...")
    
    inference_times = []
    outputs = []
    
    for i in range(num_tests):
        # 使用固定输入或生成随机输入
        if test_input_base is not None:
            test_input = test_input_base
        else:
            test_input = np.random.randn(1, input_dim).astype(np.float32)
        
        # 设置输入
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # 推理
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        
        # 获取输出
        output = interpreter.get_tensor(output_details[0]['index'])
        
        inference_time = (end_time - start_time) * 1000  # ms
        inference_times.append(inference_time)
        outputs.append(output)
        
        # 打印第一次测试的详细信息
        if i == 0:
            print(f"\n  Test 1:")
            print(f"    Input (first 5 elements): {test_input[0][:5]}")
            print(f"    Output: {output[0]}")
            print(f"    Inference time: {inference_time:.4f} ms")
    
    # 统计信息
    print(f"\nPerformance Statistics ({num_tests} runs):")
    print(f"  Average inference time: {np.mean(inference_times):.4f} ms")
    print(f"  Min inference time: {np.min(inference_times):.4f} ms")
    print(f"  Max inference time: {np.max(inference_times):.4f} ms")
    print(f"  Std deviation: {np.std(inference_times):.4f} ms")
    
    # 验证输出范围（应该在[-1, 1]之间，因为使用了tanh激活）
    all_outputs = np.concatenate(outputs)
    print(f"\nOutput Statistics:")
    print(f"  Min value: {np.min(all_outputs):.6f}")
    print(f"  Max value: {np.max(all_outputs):.6f}")
    print(f"  Mean value: {np.mean(all_outputs):.6f}")
    print(f"  Std deviation: {np.std(all_outputs):.6f}")
    
    # 检查输出是否在合理范围内
    if np.min(all_outputs) >= -1.1 and np.max(all_outputs) <= 1.1:
        print("  ✅ Output values are within expected range [-1, 1]")
    else:
        print("  ⚠️  Warning: Some output values are outside expected range [-1, 1]")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    
    return True


def compare_with_original(tflite_path, pkl_path, fixed_input=None):
    """比较TFLite模型和原始Flax模型的输出"""
    
    print("\n" + "=" * 60)
    print("Comparing TFLite model with original Flax model")
    print("=" * 60)
    
    try:
        import pickle
        import jax
        import jax.numpy as jnp
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        from aquila.modules.mlp import MLP
    except ImportError as e:
        print(f"⚠️  Skipping comparison (missing dependencies): {e}")
        return
    
    # 加载Flax模型
    print(f"\nLoading Flax model from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    flax_params = data['params']
    input_dim = data.get('input_dimension', 220)
    action_dim = 4
    
    flax_model = MLP([input_dim, 128, 128, action_dim], initial_scale=0.2)
    
    # 加载TFLite模型
    print(f"Loading TFLite model from: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 准备测试输入
    if fixed_input is not None:
        if len(fixed_input) != input_dim:
            print(f"⚠️  Warning: Fixed input dimension ({len(fixed_input)}) doesn't match model input dimension ({input_dim})")
            if len(fixed_input) > input_dim:
                fixed_input = fixed_input[:input_dim]
            else:
                fixed_input = np.pad(fixed_input, (0, input_dim - len(fixed_input)), mode='constant')
        test_input_base = fixed_input.reshape(1, input_dim)
        print(f"\n✅ Using fixed input from tmplog for comparison")
        num_comparisons = 1  # 只需要比较一次，因为输入是固定的
    else:
        test_input_base = None
        print("\nGenerating random test inputs...")
        num_comparisons = 5
    
    max_diffs = []
    mean_diffs = []
    
    for i in range(num_comparisons):
        if test_input_base is not None:
            test_input = test_input_base
        else:
            test_input = np.random.randn(1, input_dim).astype(np.float32)
        
        # TFLite推理
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Flax推理
        flax_input = jnp.array(test_input)
        flax_output = flax_model.apply(flax_params, flax_input)
        flax_output_np = np.array(flax_output)
        
        # 计算差异
        diff = np.abs(tflite_output - flax_output_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)
        
        print(f"\n  Comparison {i+1}:")
        print(f"    TFLite output: {tflite_output[0]}")
        print(f"    Flax output:   {flax_output_np[0]}")
        print(f"    Max difference: {max_diff:.8f}")
        print(f"    Mean difference: {mean_diff:.8f}")
    
    # 统计
    print("\n" + "-" * 60)
    print(f"Overall Statistics ({num_comparisons} comparisons):")
    print(f"  Average max difference: {np.mean(max_diffs):.8f}")
    print(f"  Average mean difference: {np.mean(mean_diffs):.8f}")
    print(f"  Largest max difference: {np.max(max_diffs):.8f}")
    
    if np.max(max_diffs) < 1e-5:
        print("\n✅ Models are equivalent (difference < 1e-5)")
    elif np.max(max_diffs) < 1e-4:
        print("\n✅ Models are very similar (difference < 1e-4)")
    elif np.max(max_diffs) < 1e-3:
        print("\n⚠️  Models have small differences (difference < 1e-3)")
    else:
        print("\n❌ Models have significant differences!")


def main():
    # 默认路径
    tflite_path = 'aquila/param/tflite/trackVer8_policy_stabler.tflite'
    pkl_path = 'aquila/param/trackVer8_policy_stabler.pkl'
    tmplog_path = 'tmplog.txt'
    
    # 命令行参数
    if len(sys.argv) > 1:
        tflite_path = sys.argv[1]
    if len(sys.argv) > 2:
        pkl_path = sys.argv[2]
    if len(sys.argv) > 3:
        tmplog_path = sys.argv[3]
    
    # 加载固定输入
    print("=" * 60)
    print("Loading test input data")
    print("=" * 60)
    fixed_input = load_input_from_tmplog(tmplog_path)
    if fixed_input is None:
        print("⚠️  Warning: Using random inputs instead")
    print()
    
    # 测试TFLite模型
    success = test_tflite_model(tflite_path, num_tests=100, fixed_input=fixed_input)
    
    if not success:
        sys.exit(1)
    
    # 如果原始模型存在，进行比较
    if os.path.exists(pkl_path):
        compare_with_original(tflite_path, pkl_path, fixed_input=fixed_input)
    else:
        print(f"\n⚠️  Original model not found at {pkl_path}, skipping comparison")
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()

