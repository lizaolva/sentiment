from transformers import TrainingArguments

def train_args(learning_rate, weight_decay, batch_size, epochs):
    training_args = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    save_total_limit=2,
    save_only_model=True,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1"  
    )
    return training_args