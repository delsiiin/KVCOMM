python experiments/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_default --compress-mode --compress-method "rkv" --compress-budget 1024 --compress-divide-length 128

python experiments/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_default --compress-mode --compress-method "rkv" --compress-budget 1024 --compress-divide-length 128

python experiments/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_default --compress-mode --compress-method "rkv" --compress-budget 1024 --compress-divide-length 128
