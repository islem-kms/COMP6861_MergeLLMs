from utils.model_utils import *
from b_01_prepare_datasets import multitask_train, multitask_val

# Gold standard — single model trained jointly on both tasks.
# Merged models should approach but likely won't fully match this.
model_multitask = finetune(
    train_dataset=multitask_train,
    eval_dataset=multitask_val,
    output_dir=ADAPTER_MULTITASK,
    num_epochs=3,
)