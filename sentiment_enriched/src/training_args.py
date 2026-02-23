from transformers import TrainingArguments

def train_args(checkpoints_dir, learning_rate, weight_decay, batch_size, epochs):

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        eval_strategy="epoch",
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="epoch", # возможно сократить
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        #gradient_checkpointing=True
    )
    return training_args