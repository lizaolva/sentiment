from transformers import TrainingArguments
from configs import batch_size, epochs

training_args = TrainingArguments(
    output_dir='./results',          
    eval_strategy="epoch",    
    learning_rate=1.8562022777790423e-05,             
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,  
    num_train_epochs=epochs,            
    weight_decay=0.001928128030703447,             
    logging_dir="./logs",         
    logging_steps=10,             
    save_strategy="epoch",          
    load_best_model_at_end=True,  
    metric_for_best_model="f1",
    gradient_checkpointing=True
)