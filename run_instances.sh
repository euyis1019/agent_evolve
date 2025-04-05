#!/bin/bash
# 这个脚本展示如何使用--instance-ids参数来运行特定的SWE-bench实例

# 设置并行
NUM_PROCESSES=1
NUM_CANDIDATE_SOLUTIONS=2

# 设置最大运行时间（单位：秒）
# 这将限制每个agent在单个问题上的最大运行时间
MAX_RUNTIME=300  
# 指定要运行的实例IDs
# 这里可以添加多个实例ID，用空格分隔
INSTANCE_IDS=(
# "scikit-learn__scikit-learn-14629"
# "scikit-learn__scikit-learn-14141"
sympy__sympy-14248
pylint-dev__pylint-4551
pallets__flask-5014
sphinx-doc__sphinx-7440
astropy__astropy-14096
sympy__sympy-17655
django__django-17084
django__django-16485
django__django-16901
django__django-15957
django__django-13809
django__django-14608
django__django-16263
django__django-11239
matplotlib__matplotlib-25332
astropy__astropy-8872
sympy__sympy-13551
django__django-11999
django__django-16454
django__django-14351
scikit-learn__scikit-learn-14710
django__django-16116
django__django-14404
django__django-15103
sphinx-doc__sphinx-7985
django__django-14017
django__django-14053
pylint-dev__pylint-6903
django__django-15037
django__django-14792
)

# 运行脚本
python run_agent_on_swebench_problem.py \
  --instance-ids "${INSTANCE_IDS[@]}" \
  --num-processes $NUM_PROCESSES \
  --num-candidate-solutions $NUM_CANDIDATE_SOLUTIONS \

echo "完成运行指定实例！"
