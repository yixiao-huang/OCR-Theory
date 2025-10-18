# Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers

Code for [Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](https://arxiv.org/abs/2506.10887). Yixiao Huang*, Hanlin Zhu*, Tianyu Guo*, Jiantao Jiao, Somayeh Sojoudi, Michael I. Jordan, Stuart Russell, Song Mei, NeurIPS 2025.


Install dependencies
---
Install [uv](https://docs.astral.sh/uv/getting-started/installation/). 

Then, run

```bash
uv sync
```
If you want to install additional packages:
```bash
uv add [pkg_name]
```


### Set up customized uv environment
- Create a ``.envrc`` file and write ``export UV_PROJECT_ENVIRONMENT=/path/your-env``
- install direnv: https://direnv.net/docs/installation.html
- run ``direnv allow``
Then every time you open the folder, it will execute the cmds in the ``.envrc`` file. Finally run 
``uv sync``


LLM Experiments
---
We directly adopt the codes from [extractive-structures](https://github.com/jiahai-feng/extractive-structures).

Synthetic Experiments on one-layer attention model
---
```
bash scripts/train_attn1_1heads.sh
```
### Low-rank setup
To reproduce the low rank experiments in Figure 6, run 
```
bash scripts/sweep_rank_attn1_1heads.sh
```

Citation
----
If you find our work useful, please consider citing it:
```
@article{huang2025generalization,
  title={Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers},
  author={Huang, Yixiao and Zhu, Hanlin and Guo, Tianyu and Jiao, Jiantao and Sojoudi, Somayeh and Jordan, Michael I and Russell, Stuart and Mei, Song},
  journal={arXiv preprint arXiv:2506.10887},
  year={2025}
}
```