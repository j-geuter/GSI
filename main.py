import argparse
import wandb
from models.base_models import load_big_model, load_small_model
from models.prm_model import load_prm_model
from inference.speculative import run_reward_tilted_decoding, run_reward_threshold_decoding
from dataloaders.dataset_loader import load_problem_dataset, load_olympiad_dataset, load_minerva_dataset
from grading.grader import grade_answer, extract_last_boxed
from utils.utils import group_stats
import traceback
import os

def main():
    parser = argparse.ArgumentParser(
        description="Reward Guided Speculative Decoding on Problem Datasets"
    )
    parser.add_argument(
        "--big_model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="Name or path for the big model",
    )
    parser.add_argument(
        "--small_model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Name or path for the small model",
    )
    parser.add_argument(
        "--prm_model_path",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="Path to the PRM model directory.",
    )
    parser.add_argument(
        "--beta", type=float, default=20, help="Beta scaling for reward"
    )
    parser.add_argument(
        "--small_model_device",
        type=str,
        default="cuda:0",
        help="Device for small model",
    )
    parser.add_argument(
        "--big_model_device", type=str, default="cuda:1", help="Device for big model"
    )
    parser.add_argument(
        "--prm_model_device", type=str, default="cuda:2", help="Device for PRM model"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="HuggingFaceH4/MATH-500,Hothan/OlympiadBench,minerva",
        help="Name of the datasets to use, separated by commas",
    )
    parser.add_argument(
        "--max_samples", type=int, default=100, help="Maximum number of samples to run"
    )
    parser.add_argument(
        "--n_small",
        type=int,
        default=64,
        help="Number of small model generations to sample from",
    )
    parser.add_argument(
        "--n_big",
        type=int,
        default=64,
        help="Number of big model generations to sample from",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--task_id", type=int, default=1, help="Task id passed by SLURM script"
    )
    parser.add_argument(
        "--array_id", type=str, default=1, help="Array id passed by SLURM script"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.9)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="reward_speculative_decoding",
        help="WandB project name",
    )
    parser.add_argument(
        "--decoding-mode",
        type=str,
        default="rejection-sampling",
        help="Decoding mode; `rejection-sampling` or `reward-tilted` or `reward_threshold`.",
    )
    parser.add_argument(
        "--check-and-resample-big-model",
        action="store_true",
    )
    parser.add_argument(
        "--tilt-only-by-big-model",
        action="store_true",
        help="If set, the reward is only tilted by the big model. This only applies to the `reward-tilted` decoding mode.",
    )
    parser.add_argument(
        "--resample-reward-threshold",
        type=float,
        default=10.0,
        help="Resample threshold for reward-tilted decoding mode. If the reward is below this threshold, resample.",
    )
    parser.add_argument(
        "--threshold-on-tilted",
        action="store_true",
        help="If set, the reward threshold is applied to the tilted reward (instead of just the reward). This only applies to the `reward-tilted` decoding mode.",
    )
    parser.add_argument(
        "--runs_per_dataset",
        type=int,
        default=1,
        help="Number of runs per dataset to average over",
    )
    args = parser.parse_args()

    # Initialize wandb with a run name based on SLURM array and task IDs.
    wandb.init(
        project=args.wandb_project,
        name=f"{args.array_id}_{args.task_id}",
        id=f"{args.array_id}_{args.task_id}",
        config=vars(args),
        resume='allow',
    )

    if args.decoding_mode == "reward-tilted":
        decoding_function = run_reward_tilted_decoding
    elif args.decoding_mode == "reward_threshold":
        decoding_function = run_reward_threshold_decoding

    # Load models on specified devices.
    big_model, big_tokenizer = load_big_model(
        args.big_model_name, device=args.big_model_device
    )
    small_model, small_tokenizer = load_small_model(
        args.small_model_name, device=args.small_model_device
    )
    prm_model = load_prm_model(
        args.prm_model_path, device=args.prm_model_device
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.small_model_device.split(":")[-1]
    big_vocab = big_tokenizer.get_vocab()
    big_stop_ids = [big_vocab[key] for key in big_vocab if "\n\n" in big_tokenizer.decode(big_vocab[key])]
    if isinstance(big_tokenizer.eos_token_id, (list, tuple)):
        big_stop_ids.extend(big_tokenizer.eos_token_id)
    else:
        big_stop_ids.append(big_tokenizer.eos_token_id)
    print(f"Number of stop IDs in big model: {len(big_stop_ids)}")

    small_vocab = small_tokenizer.get_vocab()
    small_stop_ids = [small_vocab[key] for key in small_vocab if "\n\n" in small_tokenizer.decode(small_vocab[key])]
    if isinstance(small_tokenizer.eos_token_id, (list, tuple)):
        small_stop_ids.extend(small_tokenizer.eos_token_id)
    else:
        small_stop_ids.append(small_tokenizer.eos_token_id)
    print(f"Number of stop IDs in small model: {len(small_stop_ids)}")

    datasets = args.datasets.split(",")
    for dataset in datasets:
            for i in range(args.runs_per_dataset):
                system_prompt = None
                # Load dataset using the modular dataset loader.
                if dataset == "HuggingFaceH4/MATH-500":
                    dataset_samples = load_problem_dataset(dataset)
                elif dataset == "Hothan/OlympiadBench":
                    dataset_samples = load_olympiad_dataset()
                elif dataset == "minerva":
                    dataset_samples = load_minerva_dataset()
                    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}. Do not include units in your final answer. For example, if the answer is `5 m/s`, write `\\boxed{5}`."
                if args.max_samples < len(dataset_samples):
                    dataset_samples = dataset_samples[:args.max_samples]

                total = 0
                total_no_errors = 0
                total_correct = 0
                sample_correct = []
                nb_errors = 0

                wandb.log({"accuracy (over samples)": 0.0})
                for sample in dataset_samples:
                    if total >= args.max_samples:
                        break
                    total += 1
                    problem = sample["problem"]
                    ground_truth = sample["answer"]
                    try:
                        # Run speculative decoding on the problem.
                        generated_text = decoding_function(
                            big_model,
                            big_stop_ids,
                            big_tokenizer,
                            small_model,
                            small_stop_ids,
                            small_tokenizer,
                            prm_model,
                            beta=args.beta,
                            prompt=problem,
                            n_small=args.n_small,
                            n_big=args.n_big,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            check_and_resample_big_model=args.check_and_resample_big_model,
                            resample_reward_threshold=args.resample_reward_threshold,
                            threshold_on_tilted=args.threshold_on_tilted,
                            system_prompt=system_prompt,
                        )
                        total_no_errors += 1

                    except Exception: # catches rare errors in generation
                        print(f"Error processing sample {total}:")
                        traceback.print_exc()
                        nb_errors += 1
                        sample_correct.append(0)
                        continue

                    correct = grade_answer(generated_text, ground_truth, extract=True)
                    total_correct += correct
                    answer = extract_last_boxed(generated_text)
                    sample_correct.append(1 if correct else 0)
                    print(f"Correct solution found: {correct}\nPredicted answer: {answer}\nGround truth: {ground_truth}")
                    wandb.log({
                        "accuracy (over samples)": total_correct / total,
                    })

                mean_accuracy, std_accuracy = group_stats(sample_correct, group_size=args.max_samples//5)
                wandb.log({"accuracy": mean_accuracy, "std accuracy": std_accuracy, "nb_errors": nb_errors})

if __name__ == "__main__":
    main()
