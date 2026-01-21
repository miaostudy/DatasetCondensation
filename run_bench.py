import subprocess
import re
import os
import time
import pandas as pd
from datetime import datetime
from collections import deque
import sys

# ================= 配置区域 =================
# 1. 数据集 (12个)
DATASETS = [
    'PathMNIST', 'OCTMNIST', 'ChestMNIST', 'BreastMNIST',
    'TissueMNIST', 'BloodMNIST', 'PneumoniaMNIST',
    'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST',
    'RetinaMNIST', 'DermaMNIST'
]

# 2. IPC 设置 (3个) -> 总任务数 = 12 * 3 = 36
IPCS = [1, 10, 50]

# 3. 核心修改：资源池配置

REAL_GPU_IDS = ['0', '1']
MAX_WORKERS_PER_GPU = 4  # 每张卡跑 4 个任务

# 构造虚拟 GPU 资源池: ['0', '1', '0', '1', '0', '1', '0', '1']
AVAILABLE_RESOURCES = deque(REAL_GPU_IDS * MAX_WORKERS_PER_GPU)

# 通用参数
MODEL = 'ConvNet'
PROXY_MODEL = 'ResNet18'
CAM_TYPE = 'GradCAM'
PROJECT_NAME = 'Benchmark_0905'


# ===========================================

def parse_output_from_file(log_path):
    """从日志文件中读取并解析 Accuracy 和 F1 Score"""
    try:
        if not os.path.exists(log_path):
            return "No Log", "No Log"

        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        acc_pattern = re.search(r'Final Accuracy - Mean = ([\d\.]+)%,', content)
        f1_patterns = re.findall(r'F1mean = ([\d\.]+) ', content)

        acc = acc_pattern.group(1) if acc_pattern else "N/A"
        f1 = f1_patterns[-1] if f1_patterns else "N/A"

        return acc, f1
    except Exception as e:
        return f"Err: {str(e)}", "Err"


def run_parallel_experiments():
    # 初始化任务队列
    task_queue = deque([(ds, ipc) for ipc in IPCS for ds in DATASETS])

    # 正在运行的任务: {process_object: (gpu_id, dataset, ipc, log_file)}
    running_procs = {}

    results = []

    # 准备目录
    log_dir = f"./{PROJECT_NAME}/logs"
    res_dir = f"./{PROJECT_NAME}/result"
    for d in [log_dir, res_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    total_tasks = len(task_queue)
    finished_tasks = 0

    print(f"🚀 开始并行评测 (8进程/2GPU)")
    print(f"📌 总任务数: {total_tasks}")
    print(f"💻 物理 GPU: {REAL_GPU_IDS}")
    print(f"⚡ 并发策略: 每张卡 {MAX_WORKERS_PER_GPU} 个任务 (总并发 {len(AVAILABLE_RESOURCES)})")
    print("-" * 60)

    # 主循环
    while task_queue or running_procs:

        # --- A. 检查完成的任务 ---
        for proc in list(running_procs.keys()):
            if proc.poll() is not None:
                gpu_id, dataset, ipc, log_file = running_procs[proc]

                # 1. 回收资源 (把 GPU ID 放回池子)
                AVAILABLE_RESOURCES.append(gpu_id)
                # 2. 移除记录
                del running_procs[proc]

                finished_tasks += 1

                # 3. 解析结果
                acc, f1 = parse_output_from_file(log_file)
                status = "✅" if proc.returncode == 0 else "❌"

                # 打印简洁进度
                print(
                    f"[{finished_tasks}/{total_tasks}] {status} 完成: {dataset} (IPC={ipc}) | GPU: {gpu_id} | Acc: {acc}%")

                results.append({
                    "Dataset": dataset,
                    "IPC": ipc,
                    "Accuracy": acc,
                    "F1 Score": f1
                })

        # --- B. 分发新任务 ---
        while task_queue and AVAILABLE_RESOURCES:
            # 拿到一个“虚拟工位” (比如 '0' 号卡的一个名额)
            gpu_id = AVAILABLE_RESOURCES.popleft()
            dataset, ipc = task_queue.popleft()

            log_filename = f"{dataset}_IPC{ipc}.log"
            log_path = os.path.join(log_dir, log_filename)

            cmd = [
                "python", "-u", "mWCAMDM.py",
                "--dataset", dataset,
                "--model", MODEL,
                "--ipc", str(ipc),
                "--proxy_model", PROXY_MODEL,
                "--cam_type", CAM_TYPE,
                "--eval_mode", "SS",
                "--save_path", res_dir,
                "--log_dir", log_dir
            ]

            # 关键：指定该进程只能看到分配给它的那张卡
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id

            print(f"🚀 启动: {dataset} (IPC={ipc}) -> GPU {gpu_id} (队列剩余: {len(task_queue)})")

            with open(log_path, 'w') as f:
                proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)

            running_procs[proc] = (gpu_id, dataset, ipc, log_path)

        # 稍微休息一下，避免 CPU 空转
        time.sleep(2)

    # ================= 汇总输出 =================
    print("\n" + "=" * 50)
    print("📊 最终评测结果汇总")
    print("=" * 50)

    df = pd.DataFrame(results)
    if not df.empty:
        # 按 Dataset 排序方便查看
        df = df.sort_values(by=['Dataset', 'IPC'])
        print(df.to_markdown(index=False))

        csv_path = f"./{PROJECT_NAME}_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存至: {csv_path}")


if __name__ == "__main__":
    run_parallel_experiments()