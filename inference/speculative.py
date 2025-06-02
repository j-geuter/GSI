import numpy as np
import torch
import wandb
from vllm import SamplingParams
import re

def extract_token_ids_between_markers(outputs):
    # Marker sequence we're looking for
    marker = [151644, 77091, 198]
    
    # Initialize list to store results
    token_ids = []
    
    # Process each output
    for output in outputs:
        # Get the prompt token IDs
        prompt_ids = output.prompt_token_ids
        
        # Find the last occurrence of the marker sequence
        start_idx = -1
        for j in range(len(prompt_ids) - len(marker), -1, -1):
            if prompt_ids[j:j+len(marker)] == marker:
                start_idx = j + len(marker)
                break
        
        # Extract tokens between marker and the last two tokens
        if start_idx != -1 and start_idx < len(prompt_ids) - 2:
            extracted = prompt_ids[start_idx:-2]
            token_ids.append(extracted)
        else:
            # If marker not found or too close to the end, append empty list
            token_ids.append([])
    
    return token_ids

def generate_candidates(
    model, 
    tokenizer, 
    messages, 
    num_candidates, 
    max_new_tokens, 
    stop_ids,
    temperature=0.9,
    top_p=0.95,
    return_probs=False,
):
    """
    Generates num_candidates candidate reasoning steps given the conversation (messages).
    """
    if return_probs:
        logprobs = 0
    else:
        logprobs = None
    continue_final = (len(messages) > 0 and messages[-1]["role"] == "assistant")
    add_gen_prompt = False if continue_final else True
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=num_candidates,
        stop_token_ids=stop_ids,
        logprobs=logprobs,
    )
    outputs = model.chat(
        messages,
        sampling_params=sampling_params,
        continue_final_message=continue_final,
        add_generation_prompt=add_gen_prompt,
        use_tqdm=False
    )
    candidates = []
    token_ids = []
    for output in outputs[0].outputs:
        candidates.append(tokenizer.decode(output.token_ids))
        token_ids.append(output.token_ids)
    if return_probs:
        logprobs = []
        for output in outputs[0].outputs:
            logprobs.append(output.cumulative_logprob)
    return candidates, logprobs, token_ids

def evaluate_candidates(prm_model, prompt, responses):
    """
    Evaluates candidate reasoning steps with the PRM.
    Here, `prompt` is a string representing the current conversation (e.g. original problem plus previous steps)
    and `responses` is a list of newly generated assistant responses.

    Returns:
        all_rewards: a list of reward scalar values (one per candidate).
    """
    messages = [
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        for response in responses
    ]
    scores = prm_model.score(messages, return_full_prm_result=False, step_sep="\n\n")
    return scores

def constant_threshold(threshold=0.8, **kwargs):
    return threshold

def constant_reward_cutoff(reward, threshold=0.8, **kwargs):
    return reward >= threshold

def run_reward_tilted_decoding(
    big_model,
    big_stop_ids,
    big_tokenizer,
    small_model,
    small_stop_ids,
    small_tokenizer,
    prm_model,
    beta,
    prompt,
    n_small=16,
    n_big=16,
    max_new_tokens=512,
    temperature=0.7,
    top_p=1.0,
    threshold_function=constant_threshold,
    check_and_resample_big_model=True,
    resample_reward_threshold=10,
    threshold_on_tilted=True,
    system_prompt=None,
    **kwargs,
):
    """
    Runs Guided Speculative Decoding (GSI).
    """
    if system_prompt is None:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    # Initialize conversation history in chat format.
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    current_response = ""
    logprob_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=1,
        n=1,
        stop_token_ids=big_stop_ids,
        logprobs=0,
        prompt_logprobs=0,
    )
    n_accepted = 0
    n_rejected = 0
    iteration = 0
    exceed_max_iter = 0

    overall_response_len = 0
    overall_small_response_len = 0
    overall_big_response_len = 0
    while True:
        iteration += 1
        # Generate candidate reasoning steps using the small model.
        candidates, logprobs, token_ids = generate_candidates(
            small_model,
            small_tokenizer,
            messages,
            n_small,
            max_new_tokens,
            stop_ids=small_stop_ids,
            temperature=temperature,
            top_p=top_p,
            return_probs=True,
        )
        responses = [current_response + candidate for candidate in candidates]
        rewards = evaluate_candidates(
            prm_model, prompt, responses
        )
        if max(rewards) < 0.1:
            print("Warning: All rewards are low. Stopping sampling.")
            break

        response_lens = [len(ids) for ids in token_ids]
        big_messages = [
                    [
                {
                    "role": "system",
                    "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for response in responses
        ]
        tokenized = big_tokenizer.apply_chat_template(big_messages, tokenize=False, add_generation_prompt=False)
        new_outputs = big_model.generate(tokenized, logprob_params)
        prompt_logprobs = [new_outputs[i].prompt_logprobs for i in range(len(new_outputs))]
        try:
            big_logprobs = [
                sum([prompt_logprobs[j][- response_lens[j] - 2 + i][token_ids[j][i]].logprob for i in range(response_lens[j])])
                for j in range(len(new_outputs))
            ]
        except Exception: # in rare cases, the tokens of small and big model do not match and we need to recompute the logprobs
            token_ids = extract_token_ids_between_markers(new_outputs)
            response_lens = [len(ids) for ids in token_ids]
            big_logprobs = [
                sum([prompt_logprobs[j][- response_lens[j] - 2 + i][token_ids[j][i]].logprob for i in range(response_lens[j])])
                for j in range(len(new_outputs))
            ]
            new_small_outputs = small_model.generate(tokenized, logprob_params)
            small_prompt_logprobs = [new_small_outputs[i].prompt_logprobs for i in range(len(new_small_outputs))]
            logprobs = [
                sum([small_prompt_logprobs[j][- response_lens[j] - 2 + i][token_ids[j][i]].logprob for i in range(response_lens[j])])
                for j in range(len(new_small_outputs))
            ]

        rewards_array = beta * np.array(rewards) + np.array(big_logprobs) - np.array(logprobs)
        rewards_only_array = beta * np.array(rewards)
        max_reward = rewards_array.max()
        exp_values = np.exp(rewards_array - max_reward)
        rewards_only_array = np.exp(rewards_only_array - rewards_only_array.max())
        probabilities = exp_values / np.max([exp_values.sum(), 1e-10])
        rewards_only_probabilities = rewards_only_array / np.max([rewards_only_array.sum(), 1e-10])
        sampled_index = np.random.choice(len(candidates), p=probabilities)
        chosen_step = candidates[sampled_index]
        chosen_logprob = logprobs[sampled_index]
        chosen_reward = rewards[sampled_index]
        chosen_response_len = len(token_ids[sampled_index])
        chosen_big_logprob = big_logprobs[sampled_index]
        chosen_tilted_reward = beta * chosen_reward + chosen_big_logprob - chosen_logprob
        chosen_probability = probabilities[sampled_index]
        chosen_rewards_only_probability = rewards_only_probabilities[sampled_index]
        probability_delta = chosen_probability - chosen_rewards_only_probability
        relative_probability_delta = (chosen_probability - chosen_rewards_only_probability) / np.max([chosen_rewards_only_probability, 1e-10])
        pi_ratio = np.exp(chosen_logprob - chosen_big_logprob)
        small_prob = torch.exp(torch.tensor(chosen_logprob)).item()
        big_prob = torch.exp(torch.tensor(chosen_big_logprob)).item()
        log_dict = {
            "speculative_chosen_reward": chosen_reward,
            "speculative_pi_S": small_prob,
            "speculative_pi_B": big_prob,
            "small_logprob": chosen_logprob,
            "big_logprob": chosen_big_logprob,
            "big-small_logprob": chosen_big_logprob - chosen_logprob,
            "chosen_tilted_reward": chosen_tilted_reward,
            "pi_ratio": pi_ratio,
            "chosen_probability": chosen_probability,
            "chosen_rewards_only_probability": chosen_rewards_only_probability,
            "probability_delta": probability_delta,
            "relative_probability_delta": relative_probability_delta,
        }

        if check_and_resample_big_model:
            threshold = threshold_function(threshold=resample_reward_threshold, **kwargs)
            print(f"chosen reward: {chosen_reward}, chosen tilted reward: {chosen_tilted_reward}, threshold: {threshold}")
            if threshold_on_tilted:
                chosen_reward = chosen_tilted_reward
            if chosen_reward > threshold: # Accepted. NOTE: here we check `>`; in spec. dec., we check `<`.
                n_accepted += 1
                log_dict["Accepted Ratio"] = n_accepted / np.max([iteration, 1e-10])
                log_dict["Accepted: Big Prob"] = big_prob
                log_dict["Accepted: Small Prob"] = small_prob
                overall_small_response_len += chosen_response_len

            else: # Rejected
                n_rejected += 1
                candidates, logprobs, token_ids = generate_candidates(
                    big_model,
                    big_tokenizer,
                    messages,
                    n_big,
                    max_new_tokens,
                    stop_ids=big_stop_ids,
                    temperature=temperature,
                    top_p=top_p,
                    return_probs=True,
                )
                responses = [current_response + candidate for candidate in candidates]

                rewards = evaluate_candidates(
                    prm_model, prompt, responses
                )
                rewards_array = np.array(rewards)
                max_reward = rewards_array.max()
                if max_reward < 0.1:
                    print("Warning: All rewards are low. Stopping sampling.")
                    break
                exp_values = np.exp(beta * (rewards_array - max_reward))
                probabilities = exp_values / np.max([exp_values.sum(), 1e-10])
                sampled_index = np.random.choice(len(candidates), p=probabilities)
                chosen_step = candidates[sampled_index]
                chosen_response_len = len(token_ids[sampled_index])
                overall_big_response_len += chosen_response_len
                big_prob_accepted = torch.exp(torch.tensor(logprobs[sampled_index])).item()
                log_dict["Rejected: Big Prob Rejected"] = big_prob
                log_dict["Rejected: Big Prob Accepted"] = big_prob_accepted
                log_dict["Rejected: Big Prob Ratio"] = big_prob_accepted / np.max([big_prob, 1e-10])
        else:
            n_accepted += 1
        overall_response_len += chosen_response_len
        current_response += chosen_step
        print(f"Current Response (iteration {iteration}): {current_response}")
        print("\n----------------------------------\n")
        if messages[-1]["role"] == "assistant":
            messages[-1]["content"] += chosen_step
        else:
            messages.append({"role": "assistant", "content": chosen_step})

        wandb.log(log_dict)
        if re.search(r'\\boxed\{.*?\}', chosen_step):
            print("Final answer found with small model in iteration {}\n".format(iteration))
            break

        if iteration >= 45:
            print("Warning: Maximum iterations (45) reached without finding a final answer.")
            exceed_max_iter += 1
            break
    overall_response_len /= iteration
    overall_small_response_len /= max(n_accepted, 1)
    overall_big_response_len /= max(n_rejected, 1)
    wandb.log(
        {
            "Exceed Max Iter (per sample)": exceed_max_iter,
            "Num Accepted (per sample)": n_accepted,
            "Num Rejected (per sample)": n_rejected,
            "Num Iterations (per sample)": iteration,
            "Accepted Ratio (per sample)": n_accepted / np.max([iteration, 1e-10]),
            "Overall Response Length (per sample)": overall_response_len,
            "Overall Small Response Length (per sample)": overall_small_response_len,
            "Overall Big Response Length (per sample)": overall_big_response_len,
        }
    )
    final_text = small_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return final_text

def run_reward_threshold_decoding(
    big_model,
    big_stop_ids,
    big_tokenizer,
    small_model,
    small_stop_ids,
    small_tokenizer,
    prm_model,
    beta,
    prompt,
    n_small=16,
    n_big=16,
    max_new_tokens=512,
    temperature=0.7,
    top_p=1.0,
    threshold_function=constant_reward_cutoff,
    resample_reward_threshold=0.7,
    system_prompt=None,
    **kwargs,
):
    """
    Reward Guided Speculative Decoding (RSD) adapted to best-of-n.
    """
    if system_prompt is None:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    # Initialize conversation history in chat format.
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    current_response = ""
    n_accepted = 0
    n_rejected = 0
    iteration = 0
    exceed_max_iter = 0

    overall_response_len = 0
    overall_small_response_len = 0
    overall_big_response_len = 0
    while True:
        iteration += 1
        # Generate candidate reasoning steps using the small model.
        candidates, _, token_ids = generate_candidates(
            small_model,
            small_tokenizer,
            messages,
            n_small,
            max_new_tokens,
            stop_ids=small_stop_ids,
            temperature=temperature,
            top_p=top_p,
            return_probs=True,
        )
        responses = [current_response + candidate for candidate in candidates]
        rewards = evaluate_candidates(
            prm_model, prompt, responses
        )
        rewards_array = np.array(rewards)
        max_reward = rewards_array.max()
        if max_reward < 0.1:
            print("Warning: All rewards are low. Stopping sampling.")
            break
        exp_values = np.exp(beta * (rewards_array - max_reward))
        probabilities = exp_values / np.max([exp_values.sum(), 1e-10])
        sampled_index = np.random.choice(len(candidates), p=probabilities)
        chosen_step = candidates[sampled_index]
        chosen_reward = rewards[sampled_index]
        chosen_response_len = len(token_ids[sampled_index])

        threshold = threshold_function(chosen_reward, threshold=resample_reward_threshold, **kwargs)
        if threshold: # Accepted
            n_accepted += 1
            overall_small_response_len += chosen_response_len

        else: # Rejected
            n_rejected += 1
            candidates, _, token_ids = generate_candidates(
                big_model,
                big_tokenizer,
                messages,
                n_big,
                max_new_tokens,
                stop_ids=big_stop_ids,
                temperature=temperature,
                top_p=top_p,
                return_probs=True,
            )
            responses = [current_response + candidate for candidate in candidates]

            rewards = evaluate_candidates(
                prm_model, prompt, responses
            )
            rewards_array = np.array(rewards)
            max_reward = rewards_array.max()
            if max_reward < 0.1:
                print("Warning: All rewards are low. Stopping sampling.")
                break
            exp_values = np.exp(beta * (rewards_array - max_reward))
            probabilities = exp_values / np.max([exp_values.sum(), 1e-10])
            sampled_index = np.random.choice(len(candidates), p=probabilities)
            chosen_step = candidates[sampled_index]
            chosen_response_len = len(token_ids[sampled_index])
            overall_big_response_len += chosen_response_len
        overall_response_len += chosen_response_len
        current_response += chosen_step
        print(f"Current Response (iteration {iteration}): {current_response}")
        print("\n----------------------------------\n")
        if messages[-1]["role"] == "assistant":
            messages[-1]["content"] += chosen_step
        else:
            messages.append({"role": "assistant", "content": chosen_step})

        if re.search(r'\\boxed\{.*?\}', chosen_step):
            print("Final answer found with small model in iteration {}\n".format(iteration))
            break

        if iteration >= 45:
            print("Warning: Maximum iterations (45) reached without finding a final answer.")
            exceed_max_iter += 1
            break
    overall_response_len /= iteration
    overall_small_response_len /= max(n_accepted, 1)
    overall_big_response_len /= max(n_rejected, 1)
    wandb.log(
        {
            "Exceed Max Iter (per sample)": exceed_max_iter,
            "Num Accepted (per sample)": n_accepted,
            "Num Rejected (per sample)": n_rejected,
            "Num Iterations (per sample)": iteration,
            "Accepted Ratio (per sample)": n_accepted / iteration,
            "Overall Response Length (per sample)": overall_response_len,
            "Overall Small Response Length (per sample)": overall_small_response_len,
            "Overall Big Response Length (per sample)": overall_big_response_len,
        }
    )
    final_text = small_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return final_text
