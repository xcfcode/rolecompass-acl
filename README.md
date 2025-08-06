# 🎭 RoleCompass

This is the official implementation for our ACL Findings paper: [**Reasoning Does Not Necessarily Improve Role-Playing Ability**](https://aclanthology.org/2025.findings-acl.537.pdf).

### 🛠️ Environment Setup

We use [OpenCompass](https://github.com/open-compass/opencompass) to evaluate the models. To install the required packages, run the following command under this folder:

```bash
# conda env
conda create --name rolecompass python=3.10 -y
conda activate rolecompass

# setup opencompass env
cd rolecompass
bash setup_environment.sh

bash setup_rolecompass.sh
```

### 🚀 Run Evaluation

To run an evaluation, you need to create a configuration file that specifies the models and datasets you want to use.

**1. Create a configuration file**

Create a new Python file inside the `opencompass/configs` directory, for example, `opencompass/configs/eval_my_task.py`.

**2. Configure models and datasets**

In your new configuration file, you need to define the `models` and `datasets` variables. You can find all available model and dataset definitions in `configs/models/` and `configs/datasets/` respectively.

Here is an example of an `opencompass/configs/eval_my_task.py` file that evaluates the `deepseek_v3` model on the `charactereval_accuracy_gen` dataset:

```python
# opencompass/configs/eval_my_task.py

# Import the model and dataset you want to use from their definition files
from .models.deepseek_v3 import models as deepseek_v3
from .datasets.charactereval.charactereval_accuracy_gen import charactereval_datasets as charactereval_accuracy_gen

# Assign the imported configurations to the 'models' and 'datasets' variables
models = deepseek_v3
datasets = charactereval_accuracy_gen
```

A list of all model and dataset identifiers can be found in `opencompass/configs/eval_rolecompass_.py`.

**3. Run the evaluation**

Once your configuration file is ready, navigate to the `opencompass` directory and run the evaluation using the following command:

```bash
cd opencompass
python run.py configs/eval_my_task.py --work-dir outputs/my_task --hf-num-gpus 1 --max-num-workers 8
```

- Replace `configs/eval_my_task.py` with the path to your configuration file.
- The `--work-dir` argument specifies the directory where results will be saved.
- Adjust `--hf-num-gpus` based on your hardware.

### 🙏 Citation

If you find our work helpful, please cite our paper:

```bibtex
@inproceedings{feng-etal-2025-reasoning,
    title = "Reasoning Does Not Necessarily Improve Role-Playing Ability",
    author = "Feng, Xiachong  and
      Dou, Longxu  and
      Kong, Lingpeng",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.537/",
    doi = "10.18653/v1/2025.findings-acl.537",
    pages = "10301--10314",
    ISBN = "979-8-89176-256-5",
    abstract = "The application of role-playing large language models (LLMs) is rapidly expanding in both academic and commercial domains, driving an increasing demand for high-precision role-playing models. Simultaneously, the rapid advancement of reasoning techniques has continuously pushed the performance boundaries of LLMs. This intersection of practical role-playing demands and evolving reasoning capabilities raises an important research question: Can reasoning techniques enhance the role-playing capabilities of LLMs?'' To address this, we conduct a comprehensive study using 6 role-playing benchmarks, 24 LLMs, and 3 distinct role-playing strategies, comparing the effectiveness of direct zero-shot role-playing, role-playing with Chain-of-Thought (CoT), and role-playing using reasoning-optimized LLMs. Our findings reveal that CoT may reduce role-playing performance, reasoning-optimized LLMs are unsuitable for role-playing, reasoning ability disrupts the role-playing scaling law, and large models still lack proficiency in advanced role-playing. Furthermore, based on extensive experimental results, we propose two promising future research directions: Role-aware Chain-of-Thought for improving role-playing LLMs and Reinforcement Learning for role-playing LLMs, aiming to enhance the adaptability, consistency, and effectiveness of role-playing LLMs for both research and real-world applications."
}
```
