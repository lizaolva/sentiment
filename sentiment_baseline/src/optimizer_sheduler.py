from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from model_and_tokenizer import model
from dataset import dataset_train
import runtime_config

cfg = runtime_config.CFG
epochs = cfg['training']['epochs']
batch_size = cfg['training']['batch_size']

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()))
total_steps = len(dataset_train) * epochs * batch_size

scheduler = get_cosine_schedule_with_warmup(
  optimizer,
  num_warmup_steps=total_steps*0.05,
  num_training_steps=total_steps
)