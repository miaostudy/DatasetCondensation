#!/bin/bash

# 参数选择
models=("ConvNet")  # 模型列表
proxy_models=("ResNet18" "ConvNet")  # 代理模型列表
proxy_epochs=(1 5 10)  # 代理训练次数
cam_types=("GradCAM" "GradCAM++" "XGradCAM")  # CAM 类型列表

# 固定的实验命令部分
base_command="python CAMDM_proxy.py --dataset oct --ipc 10 --dsa_strategy color_crop_cutout_flip_scale_rotate --init real --lr_img 1 --num_exp 1 --num_eval 5"

# 日志文件
log_file="experiment_logDM.txt"

# GPU 分配
gpus=(0 1 2 3 4)  # 可用的 GPU 编号数组
num_gpus=${#gpus[@]}  # GPU 数量
gpu_max_jobs=1  # 每张 GPU 最大同时运行的实验数
gpu_jobs=(0 0 0 0 0)  # 初始化每个 GPU 上的实验计数数组
experiment_count=0  # 实验计数器

# 运行实验的函数
run_experiment() {
    local model=$1  # 输入的模型参数
    local proxy_model=$2  # 输入的代理模型参数
    local proxy_epoch=$3  # 输入的代理训练次数
    local cam_type=$4  # 输入的 CAM 类型
    local gpu=$5  # 所选的 GPU 编号
    local gpu_idx=$6  # GPU 编号在数组中的索引
    local cmd="CUDA_VISIBLE_DEVICES=$gpu $base_command --model $model --proxy_model $proxy_model --epoch_proxy $proxy_epoch --cam_type $cam_type"  # 构建完整的运行命令
    echo "Running experiment on GPU $gpu: $cmd"  # 显示运行命令
    echo "====================" >> $log_file  # 记录到日志文件
    echo "Running experiment on GPU $gpu: $cmd" >> $log_file  # 记录到日志文件
    start_time=$(date +%s)  # 记录实验开始时间

    # 运行实验命令并将输出重定向到日志文件
    eval $cmd >> $log_file 2>&1 &

    gpu_jobs[$gpu_idx]=$((gpu_jobs[$gpu_idx] + 1))  # 更新所选 GPU 上的实验计数
    end_time=$(date +%s)  # 记录实验结束时间
    duration=$((end_time - start_time))  # 计算实验持续时间
    echo "Experiment finished in $duration seconds"  # 显示实验持续时间
    echo "Experiment finished in $duration seconds" >> $log_file  # 记录到日志文件
    echo "" >> $log_file  # 记录空行到日志文件
}

# 检查 GPU 上运行的实验数
gpu_job_count() {
    local gpu_idx=$1  # GPU 编号在数组中的索引
    echo ${gpu_jobs[$gpu_idx]}  # 返回所选 GPU 上的实验计数
}

# 逐个运行所有实验
for model in "${models[@]}"; do  # 遍历模型列表
    for proxy_model in "${proxy_models[@]}"; do  # 遍历代理模型列表
        for proxy_epoch in "${proxy_epochs[@]}"; do  # 遍历 proxy_epoch 参数的不同值
            for cam_type in "${cam_types[@]}"; do  # 遍历 CAM 类型
                # 寻找可用的 GPU
                while true; do
                    for ((i=0; i<$num_gpus; i++)); do  # 遍历可用的 GPU
                        current_jobs=$(gpu_job_count $i)  # 获取当前 GPU 上的实验数
                        if [ $current_jobs -lt $gpu_max_jobs ]; then  # 检查是否有空闲的 GPU 资源
                            run_experiment "$model" "$proxy_model" "$proxy_epoch" "$cam_type" "${gpus[$i]}" "$i"  # 运行实验
                            experiment_count=$((experiment_count + 1))  # 更新实验计数
                            break 2  # 跳出两个循环
                        fi
                    done
                    sleep 1  # 等待 1 秒后再次检查
                done
            done
        done
    done
done

# 等待所有后台进程完成
wait
echo "Total experiments run: $experiment_count"

echo "All experiments completed."  # 显示所有实验完成的消息
