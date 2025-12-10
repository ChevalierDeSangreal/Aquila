#!/usr/bin/env python
# coding: utf-8

"""
将训练好的JAX/Flax模型转换为TFLite格式，以便在嵌入式设备上使用C++部署
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Note: Import JAX modules only when needed for verification to avoid dependency issues


def load_flax_params(checkpoint_path):
    """加载Flax模型参数"""
    print(f"Loading Flax model from: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        params = data['params']
        env_config = data.get('env_config', {})
        final_loss = data.get('final_loss', 'Unknown')
        training_epochs = data.get('training_epochs', 'Unknown')
        action_repeat = data.get('action_repeat', 10)
        buffer_size = data.get('action_obs_buffer_size', 10)
        input_dimension = data.get('input_dimension', None)
    else:
        params = data
        env_config = {}
        final_loss = 'Unknown'
        training_epochs = 'Unknown'
        action_repeat = 10
        buffer_size = 10
        input_dimension = None
    
    print("✅ Flax parameters loaded successfully!")
    print(f"   Final loss: {final_loss}")
    print(f"   Training epochs: {training_epochs}")
    print(f"   Action repeat: {action_repeat}")
    print(f"   Buffer size: {buffer_size}")
    print(f"   Input dimension: {input_dimension}")
    
    return params, env_config, action_repeat, buffer_size, input_dimension


def extract_mlp_weights(flax_params):
    """从Flax参数中提取MLP权重和偏置"""
    weights = []
    biases = []
    
    # Flax参数格式：params['params']['Dense_0']['kernel'], params['params']['Dense_0']['bias']
    params_dict = flax_params['params']
    
    # 按层顺序提取权重
    layer_idx = 0
    while f'Dense_{layer_idx}' in params_dict:
        layer_name = f'Dense_{layer_idx}'
        kernel = np.array(params_dict[layer_name]['kernel'])
        bias = np.array(params_dict[layer_name]['bias'])
        
        weights.append(kernel)
        biases.append(bias)
        
        print(f"  Layer {layer_idx}: kernel shape = {kernel.shape}, bias shape = {bias.shape}")
        layer_idx += 1
    
    print(f"✅ Extracted {len(weights)} layers from Flax model")
    return weights, biases


def create_tensorflow_model(input_dim, hidden_dims, output_dim, weights, biases):
    """使用TensorFlow/Keras创建相同结构的模型并加载权重"""
    
    # 创建Sequential模型
    model = keras.Sequential()
    
    # 输入层
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # 隐藏层
    for i, (hidden_dim, w, b) in enumerate(zip(hidden_dims, weights[:-1], biases[:-1])):
        layer = keras.layers.Dense(
            hidden_dim,
            activation='relu',
            name=f'dense_{i}'
        )
        model.add(layer)
    
    # 输出层（不使用激活函数，因为我们会在后面手动添加tanh）
    output_layer = keras.layers.Dense(
        output_dim,
        activation=None,  # 先不加激活
        name=f'dense_output'
    )
    model.add(output_layer)
    
    # 添加tanh激活层
    model.add(keras.layers.Activation('tanh', name='tanh_activation'))
    
    # 构建模型
    model.build(input_shape=(None, input_dim))
    
    # 加载权重
    for i, (w, b) in enumerate(zip(weights, biases)):
        model.layers[i].set_weights([w, b])
    
    print(f"✅ TensorFlow model created with {len(weights)} layers")
    return model


def verify_model_equivalence(tf_model, flax_params, input_dim, hidden_dims, action_dim):
    """验证TensorFlow模型和Flax模型输出是否一致"""
    print("\n验证模型等价性...")
    
    try:
        # 延迟导入JAX和Flax模块，避免依赖问题
        import jax
        import jax.numpy as jnp
        from aquila.modules.mlp import MLP
        
        # 创建Flax模型
        flax_model = MLP([input_dim] + hidden_dims + [action_dim], initial_scale=0.2)
        
        # 生成随机测试输入
        test_input = np.random.randn(1, input_dim).astype(np.float32)
        
        # TensorFlow模型推理
        tf_output = tf_model(test_input).numpy()
        
        # Flax模型推理
        flax_input = jnp.array(test_input)
        flax_output = flax_model.apply(flax_params, flax_input)
        flax_output_np = np.array(flax_output)
        
        # 比较输出
        diff = np.abs(tf_output - flax_output_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Mean difference: {mean_diff:.8f}")
        
        if max_diff < 1e-5:
            print("✅ 模型验证通过！TensorFlow和Flax输出一致")
            return True
        else:
            print("⚠️  警告：模型输出存在差异，请检查转换过程")
            return False
    
    except ImportError as e:
        print(f"⚠️  跳过模型验证（缺少JAX/Flax依赖）: {e}")
        print("   转换仍然可以继续，但无法验证模型等价性")
        print("   建议: pip install jax flax distrax（如果需要验证）")
        return None


def convert_to_tflite(tf_model, output_path, optimize=False, target_tflite_version='2.14.0'):
    """将TensorFlow模型转换为TFLite格式
    
    注意：optimize 参数会启用量化优化，但是如果没有提供代表性数据集，
    可能会导致严重的精度损失！建议保持 optimize=False 以确保模型精度。
    """
    print("\n开始转换为TFLite格式...")
    print(f"  目标TFLite版本: {target_tflite_version}")
    
    # Keras 3.x兼容性：需要先确保模型已构建
    if not tf_model.built:
        print("  构建模型...")
        # 使用正确的输入形状构建模型
        input_shape = tf_model.input_shape
        if input_shape[0] is None:
            # 如果batch维度是None，使用一个示例输入
            sample_input = tf.random.normal((1,) + tuple(input_shape[1:]))
            tf_model(sample_input)
    
    # 使用 from_keras_model 方式转换（推荐方式，更可靠）
    print("  使用 from_keras_model 转换器...")
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    
    if optimize:
        # 优化选项 - 警告：可能导致精度损失
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  ⚠️  启用默认优化（可能导致精度损失）")
        print("  建议：提供代表性数据集或禁用优化")
    else:
        print("  ✓ 禁用优化（保持最佳精度）")
    
    # 转换模型
    print("  正在转换模型...")
    tflite_model = converter.convert()
    
    # 保存TFLite模型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"✅ TFLite模型已保存到: {output_path}")
    print(f"   文件大小: {file_size:.2f} KB")
    
    return tflite_model


def test_tflite_model(tflite_path, input_dim, num_tests=5):
    """测试TFLite模型推理"""
    print(f"\n测试TFLite模型推理 (共{num_tests}次)...")
    
    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # 获取输入输出张量信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  输入张量: shape={input_details[0]['shape']}, dtype={input_details[0]['dtype']}")
    print(f"  输出张量: shape={output_details[0]['shape']}, dtype={output_details[0]['dtype']}")
    
    # 多次推理测试
    inference_times = []
    for i in range(num_tests):
        test_input = np.random.randn(1, input_dim).astype(np.float32)
        
        # 设置输入
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # 推理
        import time
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        
        # 获取输出
        output = interpreter.get_tensor(output_details[0]['index'])
        
        inference_time = (end_time - start_time) * 1000  # ms
        inference_times.append(inference_time)
        
        if i == 0:
            print(f"\n  测试输入: {test_input[0][:5]}... (显示前5个)")
            print(f"  测试输出: {output[0]}")
    
    avg_time = np.mean(inference_times)
    print(f"\n  平均推理时间: {avg_time:.4f} ms")
    print(f"  推理时间范围: {np.min(inference_times):.4f} - {np.max(inference_times):.4f} ms")
    print("✅ TFLite模型测试完成！")


def save_model_info(output_dir, input_dim, action_dim, buffer_size, action_repeat, env_config):
    """保存模型信息到文本文件，便于C++部署时参考"""
    info_path = os.path.join(output_dir, 'model_info.txt')
    
    with open(info_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TrackVer6 TFLite Model Information\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("模型结构:\n")
        f.write(f"  输入维度: {input_dim}\n")
        f.write(f"  输出维度 (动作维度): {action_dim}\n")
        f.write(f"  隐藏层: [128, 128]\n")
        f.write(f"  激活函数: ReLU (隐藏层), Tanh (输出层)\n\n")
        
        f.write("缓冲区配置:\n")
        f.write(f"  动作-状态缓冲区大小: {buffer_size}\n")
        f.write(f"  动作重复: {action_repeat} steps\n\n")
        
        f.write("环境配置:\n")
        for key, value in env_config.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("C++部署说明:\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 输入格式:\n")
        f.write(f"   - 输入是一个长度为{input_dim}的浮点数数组\n")
        f.write(f"   - 格式: [obs_0 + action_0, obs_1 + action_1, ..., obs_{buffer_size-1} + action_{buffer_size-1}]\n")
        f.write(f"   - 每个元素包含观测和动作的拼接\n\n")
        
        f.write("2. 输出格式:\n")
        f.write(f"   - 输出是一个长度为{action_dim}的浮点数数组\n")
        f.write(f"   - 表示四旋翼的控制指令\n")
        f.write(f"   - 输出范围: [-1, 1] (经过tanh激活)\n\n")
        
        f.write("3. 推理流程:\n")
        f.write(f"   - 维护一个大小为{buffer_size}的动作-状态缓冲区\n")
        f.write(f"   - 每{action_repeat}个时间步获取一次新动作\n")
        f.write(f"   - 将缓冲区展平为{input_dim}维向量作为网络输入\n")
        f.write(f"   - 网络输出{action_dim}维动作向量\n\n")
        
        f.write("4. TFLite C++ API使用示例:\n")
        f.write("   ```cpp\n")
        f.write("   // 加载模型\n")
        f.write("   auto model = tflite::FlatBufferModel::BuildFromFile(\"trackVer6_policy.tflite\");\n")
        f.write("   tflite::ops::builtin::BuiltinOpResolver resolver;\n")
        f.write("   tflite::InterpreterBuilder builder(*model, resolver);\n")
        f.write("   std::unique_ptr<tflite::Interpreter> interpreter;\n")
        f.write("   builder(&interpreter);\n")
        f.write("   interpreter->AllocateTensors();\n\n")
        f.write("   // 准备输入\n")
        f.write(f"   float* input = interpreter->typed_input_tensor<float>(0);\n")
        f.write(f"   // 填充输入数据 (input[0] 到 input[{input_dim-1}])\n\n")
        f.write("   // 推理\n")
        f.write("   interpreter->Invoke();\n\n")
        f.write("   // 获取输出\n")
        f.write(f"   float* output = interpreter->typed_output_tensor<float>(0);\n")
        f.write(f"   // 使用输出数据 (output[0] 到 output[{action_dim-1}])\n")
        f.write("   ```\n\n")
        
    print(f"✅ 模型信息已保存到: {info_path}")


def main():
    # ==================== 配置 ====================
    # 输入文件路径
    checkpoint_path = 'aquila/param/hoverVer2_policy.pkl'
    
    # 输出文件路径
    output_dir = 'aquila/param/tflite'
    tflite_path = os.path.join(output_dir, 'hoverVer2_policy.tflite')
    
    # 是否启用优化（警告：启用优化可能导致严重的精度损失！）
    optimize = False  # 建议保持 False 以确保模型精度
    
    print("=" * 60)
    print("JAX/Flax to TFLite Converter")
    print("=" * 60)
    
    # ==================== 加载Flax模型 ====================
    flax_params, env_config, action_repeat, buffer_size, input_dimension = load_flax_params(checkpoint_path)
    
    # ==================== 获取维度信息 ====================
    print("\n获取维度信息...")
    
    # 优先从保存的配置中读取，避免创建环境
    action_dim = 4  # 四旋翼控制维度
    obs_dim = env_config.get('obs_dim', 18)  # TrackEnvVer6默认观测维度
    
    # 如果配置中没有观测维度，尝试创建环境获取
    if 'obs_dim' not in env_config:
        try:
            from aquila.envs.target_trackVer6 import TrackEnvVer6
            from aquila.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
            
            print("  从环境获取维度信息...")
            env = TrackEnvVer6(
                max_steps_in_episode=1000,
                dt=0.01,
                delay=0.03,
                omega_std=0.1,
                action_penalty_weight=0.5,
                target_height=2.0,
                target_init_distance_min=0.5,
                target_init_distance_max=1.5,
                target_speed_max=1.0,
                reset_distance=100.0,
                max_speed=20.0,
                thrust_to_weight_min=1.5,
                thrust_to_weight_max=3.0,
            )
            env = MinMaxObservationWrapper(env)
            env = NormalizeActionWrapper(env)
            
            action_dim = env.action_space.shape[0]
            obs_dim = env.observation_space.shape[0]
        except ImportError as e:
            print(f"  ⚠️  无法导入环境模块: {e}")
            print(f"  使用默认维度: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # 计算输入维度
    if input_dimension is None:
        input_dim = buffer_size * (obs_dim + action_dim)
    else:
        input_dim = input_dimension
    
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  缓冲区大小: {buffer_size}")
    print(f"  输入维度: {input_dim}")
    
    # ==================== 提取Flax权重 ====================
    print("\n提取Flax模型权重...")
    weights, biases = extract_mlp_weights(flax_params)
    
    # ==================== 创建TensorFlow模型 ====================
    print("\n创建TensorFlow模型...")
    hidden_dims = [128, 128]  # 与训练脚本中的MLP结构一致
    tf_model = create_tensorflow_model(input_dim, hidden_dims, action_dim, weights, biases)
    
    # 打印模型摘要
    print("\nTensorFlow模型结构:")
    tf_model.summary()
    
    # ==================== 验证模型等价性 ====================
    verify_model_equivalence(tf_model, flax_params, input_dim, hidden_dims, action_dim)
    
    # ==================== 转换为TFLite ====================
    tflite_model = convert_to_tflite(tf_model, tflite_path, optimize=optimize)
    
    # ==================== 测试TFLite模型 ====================
    test_tflite_model(tflite_path, input_dim, num_tests=10)
    
    # ==================== 保存模型信息 ====================
    save_model_info(output_dir, input_dim, action_dim, buffer_size, action_repeat, env_config)
    
    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"TFLite模型路径: {tflite_path}")
    print(f"模型信息文件: {os.path.join(output_dir, 'model_info.txt')}")
    print(f"输入维度: {input_dim}")
    print(f"输出维度: {action_dim}")
    print(f"优化: {'启用' if optimize else '禁用'}")
    print("\n模型已准备好用于C++部署！")
    print("=" * 60)


if __name__ == "__main__":
    main()

