# Augment SWE-bench Verified Agent

## Installation
```
conda create -n py312 python=3.12
cd agent_evolve
conda activate py312
./setup.sh
source .venv/bin/activate
uv pip install litellm
```

## Usage
```
python cli.py
```

## Change log 4.6
- .env file for API keys
- specify insatance ids
- Support Deepseek API through the `LiteLLM` library


## Todolist
- ~~max tokens limitation~~ Seem not necessary, there is a MAX_TURNS parameters.
- In my enviornment, the system cann't process concurrent requests even I set the number `--num-processes` to 2, Have a Check?
- #Need to check Label.


