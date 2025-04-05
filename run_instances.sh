#!/bin/bash
# 这个脚本展示如何使用--instance-ids参数来运行特定的SWE-bench实例



# 设置并行
NUM_PROCESSES=1
NUM_CANDIDATE_SOLUTIONS=1

# 设置最大运行时间（单位：秒）
# 这将限制每个agent在单个问题上的最大运行时间
MAX_RUNTIME=300  
# 指定要运行的实例IDs
# 这里可以添加多个实例ID，用空格分隔
INSTANCE_IDS=(
"scikit-learn__scikit-learn-14629"
# "scikit-learn__scikit-learn-14141"
)

# 运行脚本
python run_agent_on_swebench_problem.py \
  --instance-ids "${INSTANCE_IDS[@]}" \
  --num-processes $NUM_PROCESSES \
  --num-candidate-solutions $NUM_CANDIDATE_SOLUTIONS \

echo "完成运行指定实例！" 