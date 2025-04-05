# Augment SWE-bench Verified Agent

## Installation
```
git clone git@github.com:euyis1019/agent_evolve.git
cd agent_evolve
conda create -n code_agent python=3.12
conda activate code_agent
pip install -r requirements.txt
pip install litellm
pip install datasets
pip install docker
```

## Usage
```
python cli.py
python run_agent_on_swebench_problem.py --num-examples 1 --num-candidate-solutions 1
./run_instances.sh
```

## Change log 4.6
- .env file for API keys
- specify insatance ids
- Support Deepseek API through the `LiteLLM` library

## Todolist
- ~~max tokens limitation~~ Seem not necessary, there is a MAX_TURNS parameters.
- In my enviornment, the system cann't process concurrent requests even I set the number `--num-processes` to 2, Have a Check?
- #Need to check Label.


