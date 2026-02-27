from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

def optim_sheduler(model, train_loader, epochs, batch_size):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()))
    total_steps = len(train_loader.dataset) * epochs * batch_size

    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps*0.05,
    num_training_steps=total_steps
    )
    return optimizer, scheduler