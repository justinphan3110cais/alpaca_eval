import fire
import subprocess
import os

def main(model_path, 
        api_model=False,
        output_path="tmp.json",
        tensor_parallel_size=1,
        chat_template="",
    ):
    
    print("===Generating model Answer ===")
    if api_model:
        raise NotImplementedError
    else:
        subprocess.run([
            "python", "generate_completions_alpaca_eval.py", 
            "--model_path", str(model_path),
            "--output_path", str(output_path),
            "--tensor_parallel_size", str(tensor_parallel_size),
            "--chat_template", chat_template,
        ])

    os.system(f"alpaca_eval --model_outputs {str(output_path)} --precomputed_leaderboard results/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv --name {str(output_path)} --is_cache_leaderboard {False} --is_return_instead_of_print {False}")

if __name__ == "__main__":
    fire.Fire(main)