# Guided Speculative Inference for Efficient Test-Time Alignment of LLMs

This is the code repository implementing _Guided Speculative Inference_ (GSI) from the paper
[Guided Speculative Inference for Efficient Test-Time Alignment of LLMs](https://arxiv.org/abs/2506.04118) (J. Geuter, Y. Mroueh, D. Alvarez-Melis, 2025).
GSI is an inference-time algorithm for LLMs which combines soft best-of-n sampling with speculative sampling from a small model,
and allows for efficient test-time scaling of LLM reasoning models.

## Installation
To run the code, you need to install the `requirements.txt`.
Our models are implemented with vLLM using Red Hat AI Innovation Team's [RewardHub](https://github.com/Red-Hat-AI-Innovation-Team/reward_hub).
Since we made slight changes to their implementation, the updated RewardHub implementation is included in this repo.
Simply run

```bash
cd reward_hub
pip install -e .
cd ..
```

to install RewardHub. If the build fails, instead clone the [RewardHub](https://github.com/Red-Hat-AI-Innovation-Team/reward_hub) repository,
and replace the file `reward_hub/reward_hub/vllm/reward.py` by the one in this repo, then install RewardHub as outlined above.

## Usage

You can launch a SLURM job by running `job.slurm`. This launches `main.py` which evaluates GSI on the datasets specified in the job file.
Make sure to replace all relevant parameters in `job.slurm`.
Then, you can run the `main.py` file
by running the SLURM file `job.slurm`. This will start a job with 3 GPUs (one each for the small, large, and reward model). Change the job file accordingly with your credentials.

## Credits
We would like to thank Red Hat AI Innovation Team for their [RewardHub](https://github.com/Red-Hat-AI-Innovation-Team/reward_hub) repository, and
OpenAI for their [PRM800k](https://github.com/openai/prm800k) repository, from which we are using the grading function for grading
correctness of LLM-generated answers.

## Citation
If you find this repository helpful, please consider citing our paper.
```
@inproceedings{
  geuter2025guided,
  title={Guided Speculative Inference for Efficient Test-Time Alignment of {LLM}s},
  author={Jonathan Geuter and Youssef Mroueh and David Alvarez-Melis},
  booktitle={ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models},
  year={2025},
  url={https://openreview.net/forum?id=cRTWN5iwiy}
}
```
