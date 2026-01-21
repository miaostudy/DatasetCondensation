import subprocess
import re
import os
import time
import pandas as pd
from datetime import datetime
from collections import deque
import sys


DATASETS = [
    'PathMNIST', 'OCTMNIST', 'ChestMNIST', "BreastMNIST", "TissueMNIST", "BloodMNIST", "PneumoniaMNIST", "OrganAMNIST", "OrganCMNIST", "OrganSMNIST"
]

IPCS = [1, 10, 50]

REAL_GPU_IDS = ['0', '1']
MAX_WORKERS_PER_GPU = 4
AVAILABLE_RESOURCES = deque(REAL_GPU_IDS * MAX_WORKERS_PER_GPU)

MODEL = 'ConvNet'
PROXY_MODEL = 'ResNet18'
CAM_TYPE = 'GradCAM'
PROJECT_NAME = 'Benchmark_0905'


def run_parallel_experiments():
    task_queue = deque([(ds, ipc) for ipc in IPCS for ds in DATASETS])
    running_procs = {}

    results = []

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
                status = "✅" if proc.returncode == 0 else "❌"

                # 打印简洁进度
                print(
                    f"[{finished_tasks}/{total_tasks}] {status} 完成: {dataset} (IPC={ipc}) | GPU: {gpu_id}")

                results.append({
                    "Dataset": dataset,
                    "IPC": ipc,
                })

        while task_queue and AVAILABLE_RESOURCES:
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
                "--log_dir", log_dir,
                "--num_eval", "5",
                "--save_model_path", res_dir
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id

            print(f"启动: {dataset} (IPC={ipc}) -> GPU {gpu_id} (队列剩余: {len(task_queue)})")

            with open(log_path, 'w') as f:
                proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)

            running_procs[proc] = (gpu_id, dataset, ipc, log_path)

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