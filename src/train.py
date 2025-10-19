from model&tokenizer import model, tokenizer
import training_args
from dataset import val_loader, train_loader
from metrics import compute_metrics

from torch.utils.data import Dataset, DataLoader

trainer = Trainer(
    model=model,
    #model_init = model_init,
    args=training_args,
    train_dataset=train_loader.dataset,  
    eval_dataset=val_loader.dataset,  
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    #optimizers=(optimizer, scheduler)
    )

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")