from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import SubjectiveSummarizer

with read_base():
    # define models
    # instruct models
    from .models.qwen2_5.hf_qwen2_5_0_5b_instruct import models as qwen2_5_0_5b_instruct
    from .models.qwen2_5.hf_qwen2_5_1_5b_instruct import models as qwen2_5_1_5b_instruct
    from .models.qwen2_5.hf_qwen2_5_3b_instruct import models as qwen2_5_3b_instruct
    from .models.qwen2_5.hf_qwen2_5_7b_instruct import models as qwen2_5_7b_instruct
    from .models.qwen2_5.hf_qwen2_5_14b_instruct import models as qwen2_5_14b_instruct
    from .models.qwen2_5.hf_qwen2_5_32b_instruct import models as qwen2_5_32b_instruct
    from .models.qwen2_5.hf_qwen2_5_72b_instruct import models as qwen2_5_72b_instruct
    from .models.hf_llama.hf_llama3_1_8b_instruct import models as llama3_1_8b_instruct
    from .models.hf_llama.hf_llama3_2_1b_instruct import models as llama3_2_1b_instruct
    from .models.hf_llama.hf_llama3_2_3b_instruct import models as llama3_2_3b_instruct
    from .models.hf_llama.hf_llama3_3_70b_instruct import models as llama3_3_70b_instruct
    from .models.gemma.hf_gemma2_2b_it import models as gemma2_2b_it
    from .models.gemma.hf_gemma2_9b_it import models as gemma2_9b_it
    from .models.gemma.hf_gemma2_27b_it import models as gemma2_27b_it
    from .models.mistral.hf_ministral_8b_instruct_2410 import models as ministral_8b_instruct_2410
    from .models.mistral.hf_mistral_7b_instruct_v0_3 import models as mistral_7b_instruct_v0_3
    # GPT and Claude
    from .models.openai.gpt_4 import models as gpt4
    # since we use ai-yyds so we use openai
    from .models.openai.claude_3_5 import models as claude3_5
    # O1-style model
    from .models.deepseek_r1 import models as deepseek_r1
    from .models.deepseek_v3 import models as deepseek_v3
    from .models.hf_deepseek_r1_distill_llama3_1_8b import models as deepseek_r1_distill_llama3_1_8b
    from .models.hf_deepseek_r1_distill_llama3_3_70b_instruct import models as deepseek_r1_distill_llama3_3_70b
    from .models.hf_deepseek_r1_distill_qwen2_5_14b import models as deepseek_r1_distill_qwen2_5_14b
    from .models.hf_deepseek_r1_distill_qwen2_5_32b import models as deepseek_r1_distill_qwen2_5_32b
    from .models.qwq_32b_preview import models as qwq_32b_preview
    from .models.openai.o1_mini_2024_09_12 import models as o1_mini_2024_09_12

    # define datasets
    from .datasets.socialbench.socialbench_sap_zh_gen import socialbench_datasets as socialbench_sap_zh_gen
    from .datasets.socialbench.socialbench_sap_en_gen import socialbench_datasets as socialbench_sap_en_gen
    from .datasets.socialbench.socialbench_sa_rolestyle_zh_gen import socialbench_datasets as socialbench_sa_rolestyle_zh_gen
    from .datasets.socialbench.socialbench_sa_rolestyle_en_gen import socialbench_datasets as socialbench_sa_rolestyle_en_gen
    from .datasets.socialbench.socialbench_sa_roleknowledge_zh_gen import socialbench_datasets as socialbench_sa_roleknowledge_zh_gen
    from .datasets.socialbench.socialbench_sa_roleknowledge_en_gen import socialbench_datasets as socialbench_sa_roleknowledge_en_gen
    from .datasets.socialbench.socialbench_mem_short_zh_gen import socialbench_datasets as socialbench_mem_short_zh_gen
    from .datasets.socialbench.socialbench_mem_short_en_gen import socialbench_datasets as socialbench_mem_short_en_gen
    from .datasets.socialbench.socialbench_mem_long_zh_gen import socialbench_datasets as socialbench_mem_long_zh_gen
    from .datasets.socialbench.socialbench_mem_long_en_gen import socialbench_datasets as socialbench_mem_long_en_gen
    # cot
    from .datasets.socialbench.cot.socialbench_sap_zh_gen_cot import socialbench_datasets as socialbench_sap_zh_gen_cot
    from .datasets.socialbench.cot.socialbench_sap_en_gen_cot import socialbench_datasets as socialbench_sap_en_gen_cot
    from .datasets.socialbench.cot.socialbench_sa_rolestyle_zh_gen_cot import socialbench_datasets as socialbench_sa_rolestyle_zh_gen_cot
    from .datasets.socialbench.cot.socialbench_sa_rolestyle_en_gen_cot import socialbench_datasets as socialbench_sa_rolestyle_en_gen_cot
    from .datasets.socialbench.cot.socialbench_sa_roleknowledge_zh_gen_cot import socialbench_datasets as socialbench_sa_roleknowledge_zh_gen_cot
    from .datasets.socialbench.cot.socialbench_sa_roleknowledge_en_gen_cot import socialbench_datasets as socialbench_sa_roleknowledge_en_gen_cot
    from .datasets.socialbench.cot.socialbench_mem_short_zh_gen_cot import socialbench_datasets as socialbench_mem_short_zh_gen_cot
    from .datasets.socialbench.cot.socialbench_mem_short_en_gen_cot import socialbench_datasets as socialbench_mem_short_en_gen_cot
    from .datasets.socialbench.cot.socialbench_mem_long_zh_gen_cot import socialbench_datasets as socialbench_mem_long_zh_gen_cot
    from .datasets.socialbench.cot.socialbench_mem_long_en_gen_cot import socialbench_datasets as socialbench_mem_long_en_gen_cot

    from .datasets.cross.cross_gen import cross_datasets as cross_gen
    from .datasets.cross.cot.cross_gen_cot import cross_datasets as cross_gen_cot

    from .datasets.rolebench.instruction_generalization_eng import instruction_generalization_eng_datasets as instruction_generalization_eng_datasets
    from .datasets.rolebench.instruction_generalization_zh import instruction_generalization_zh_datasets as instruction_generalization_zh_datasets
    from .datasets.rolebench.role_generalization_eng import role_generalization_eng_datasets as role_generalization_eng_datasets
    from .datasets.rolebench.cot.instruction_generalization_eng_cot import instruction_generalization_eng_datasets as instruction_generalization_eng_datasets_cot
    from .datasets.rolebench.cot.instruction_generalization_zh_cot import instruction_generalization_zh_datasets as instruction_generalization_zh_datasets_cot
    from .datasets.rolebench.cot.role_generalization_eng_cot import role_generalization_eng_datasets as role_generalization_eng_datasets_cot

    from .datasets.hpd.hpd_en_gen import hpd_datasets as hpd_en_gen
    from .datasets.hpd.hpd_zh_gen import hpd_datasets as hpd_zh_gen
    from .datasets.hpd.cot.hpd_en_gen_cot import hpd_datasets as hpd_en_gen_cot
    from .datasets.hpd.cot.hpd_zh_gen_cot import hpd_datasets as hpd_zh_gen_cot

    from .datasets.incharacter.incharacter_16Personalities_en_gen import incharacter_datasets as incharacter_16Personalities_en_gen
    from .datasets.incharacter.incharacter_16Personalities_zh_gen import incharacter_datasets as incharacter_16Personalities_zh_gen
    from .datasets.incharacter.incharacter_BFI_en_gen import incharacter_datasets as incharacter_BFI_en_gen
    from .datasets.incharacter.incharacter_BFI_zh_gen import incharacter_datasets as incharacter_BFI_zh_gen
    from .datasets.incharacter.incharacter_BSRI_en_gen import incharacter_datasets as incharacter_BSRI_en_gen
    from .datasets.incharacter.incharacter_BSRI_zh_gen import incharacter_datasets as incharacter_BSRI_zh_gen
    from .datasets.incharacter.incharacter_CABIN_en_gen import incharacter_datasets as incharacter_CABIN_en_gen
    from .datasets.incharacter.incharacter_CABIN_zh_gen import incharacter_datasets as incharacter_CABIN_zh_gen
    from .datasets.incharacter.incharacter_DTDD_en_gen import incharacter_datasets as incharacter_DTDD_en_gen
    from .datasets.incharacter.incharacter_DTDD_zh_gen import incharacter_datasets as incharacter_DTDD_zh_gen
    from .datasets.incharacter.incharacter_ECR_R_en_gen import incharacter_datasets as incharacter_ECR_R_en_gen
    from .datasets.incharacter.incharacter_ECR_R_zh_gen import incharacter_datasets as incharacter_ECR_R_zh_gen
    from .datasets.incharacter.incharacter_EIS_en_gen import incharacter_datasets as incharacter_EIS_en_gen
    from .datasets.incharacter.incharacter_EIS_zh_gen import incharacter_datasets as incharacter_EIS_zh_gen
    from .datasets.incharacter.incharacter_Empathy_en_gen import incharacter_datasets as incharacter_Empathy_en_gen
    from .datasets.incharacter.incharacter_Empathy_zh_gen import incharacter_datasets as incharacter_Empathy_zh_gen
    from .datasets.incharacter.incharacter_EPQ_R_en_gen import incharacter_datasets as incharacter_EPQ_R_en_gen
    from .datasets.incharacter.incharacter_EPQ_R_zh_gen import incharacter_datasets as incharacter_EPQ_R_zh_gen
    from .datasets.incharacter.incharacter_GSE_en_gen import incharacter_datasets as incharacter_GSE_en_gen
    from .datasets.incharacter.incharacter_GSE_zh_gen import incharacter_datasets as incharacter_GSE_zh_gen
    from .datasets.incharacter.incharacter_ICB_en_gen import incharacter_datasets as incharacter_ICB_en_gen
    from .datasets.incharacter.incharacter_ICB_zh_gen import incharacter_datasets as incharacter_ICB_zh_gen
    from .datasets.incharacter.incharacter_LMS_en_gen import incharacter_datasets as incharacter_LMS_en_gen
    from .datasets.incharacter.incharacter_LMS_zh_gen import incharacter_datasets as incharacter_LMS_zh_gen
    from .datasets.incharacter.incharacter_LOT_R_en_gen import incharacter_datasets as incharacter_LOT_R_en_gen
    from .datasets.incharacter.incharacter_LOT_R_zh_gen import incharacter_datasets as incharacter_LOT_R_zh_gen
    from .datasets.incharacter.incharacter_WLEIS_en_gen import incharacter_datasets as incharacter_WLEIS_en_gen
    from .datasets.incharacter.incharacter_WLEIS_zh_gen import incharacter_datasets as incharacter_WLEIS_zh_gen
    from .datasets.incharacter.cot.incharacter_16Personalities_en_gen_cot import incharacter_datasets as incharacter_16Personalities_en_gen_cot
    from .datasets.incharacter.cot.incharacter_16Personalities_zh_gen_cot import incharacter_datasets as incharacter_16Personalities_zh_gen_cot
    from .datasets.incharacter.cot.incharacter_BFI_en_gen_cot import incharacter_datasets as incharacter_BFI_en_gen_cot
    from .datasets.incharacter.cot.incharacter_BFI_zh_gen_cot import incharacter_datasets as incharacter_BFI_zh_gen_cot

    from .datasets.charactereval.charactereval_accuracy_gen import charactereval_datasets as charactereval_accuracy_gen
    from .datasets.charactereval.charactereval_behavior_gen import charactereval_datasets as charactereval_behavior_gen
    from .datasets.charactereval.charactereval_coherence_gen import charactereval_datasets as charactereval_coherence_gen
    from .datasets.charactereval.charactereval_communication_skills_gen import charactereval_datasets as charactereval_communication_skills_gen
    from .datasets.charactereval.charactereval_consistency_gen import charactereval_datasets as charactereval_consistency_gen
    from .datasets.charactereval.charactereval_diversity_gen import charactereval_datasets as charactereval_diversity_gen
    from .datasets.charactereval.charactereval_empathy_gen import charactereval_datasets as charactereval_empathy_gen
    from .datasets.charactereval.charactereval_exposure_gen import charactereval_datasets as charactereval_exposure_gen
    from .datasets.charactereval.charactereval_fluency_gen import charactereval_datasets as charactereval_fluency_gen
    from .datasets.charactereval.charactereval_hallucination_gen import charactereval_datasets as charactereval_hallucination_gen
    from .datasets.charactereval.charactereval_humanlikeness_gen import charactereval_datasets as charactereval_humanlikeness_gen
    from .datasets.charactereval.charactereval_utterance_gen import charactereval_datasets as charactereval_utterance_gen
    from .datasets.charactereval.cot.charactereval_accuracy_gen_cot import charactereval_datasets as charactereval_accuracy_gen_cot
    from .datasets.charactereval.cot.charactereval_behavior_gen_cot import charactereval_datasets as charactereval_behavior_gen_cot
    from .datasets.charactereval.cot.charactereval_coherence_gen_cot import charactereval_datasets as charactereval_coherence_gen_cot
    from .datasets.charactereval.cot.charactereval_communication_skills_gen_cot import charactereval_datasets as charactereval_communication_skills_gen_cot
    from .datasets.charactereval.cot.charactereval_consistency_gen_cot import charactereval_datasets as charactereval_consistency_gen_cot
    from .datasets.charactereval.cot.charactereval_diversity_gen_cot import charactereval_datasets as charactereval_diversity_gen_cot
    from .datasets.charactereval.cot.charactereval_empathy_gen_cot import charactereval_datasets as charactereval_empathy_gen_cot
    from .datasets.charactereval.cot.charactereval_exposure_gen_cot import charactereval_datasets as charactereval_exposure_gen_cot
    from .datasets.charactereval.cot.charactereval_fluency_gen_cot import charactereval_datasets as charactereval_fluency_gen_cot
    from .datasets.charactereval.cot.charactereval_hallucination_gen_cot import charactereval_datasets as charactereval_hallucination_gen_cot
    from .datasets.charactereval.cot.charactereval_humanlikeness_gen_cot import charactereval_datasets as charactereval_humanlikeness_gen_cot
    from .datasets.charactereval.cot.charactereval_utterance_gen_cot import charactereval_datasets as charactereval_utterance_gen_cot


# infer = dict(
#     partitioner=dict(type=NaivePartitioner),
#     runner=dict(
#         type=LocalRunner,
#         task=dict(type=OpenICLInferTask),
#         max_num_workers=1,  # Maximum concurrent evaluation task count
#     ),
# )
