from transformers import BertForMaskedLM, BertTokenizer,BertConfig,Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset,Dataset,concatenate_datasets
import torch
import json
import multiprocessing
from tqdm import tqdm
from datetime import datetime

# Define tokenize function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Multiprocessing handler function
def process_batch(args):
    batch, tokenizer = args
    return Dataset.from_dict(tokenize_function(batch, tokenizer))

# Main function
def main(dataset, tokenizer, batch_size=1000, num_proc=6):
    # Split the dataset into chunks
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

    # Create a process pool
    pool = multiprocessing.Pool(processes=num_proc)

    # Prepare multiprocessing arguments (pass tokenizer to each process)
    task_args = [(batch, tokenizer) for batch in batches]

    # Use tqdm to display a progress bar
    tokenized_batches = []
    with tqdm(total=len(batches), desc="Tokenizing") as pbar:
        for result in pool.imap(process_batch, task_args):
            tokenized_batches.append(result)
            pbar.update(1)

    # Close the process pool
    pool.close()
    pool.join()

    # Merge the processed results
    tokenized_datasets = concatenate_datasets(tokenized_batches)
    return tokenized_datasets

if __name__ == "__main__":
    # Paths for model, tokenizer, and mixed data
    model_path="model\\bert\\bert_base"
    tokenizer_path="vocab\\bert_tokenizer"
    dataset_path="bert_pretrain_payloads.json"

    # **1️⃣ Load BERT MLM pre-trained model**
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    vocab_size=tokenizer.vocab_size

    # Load model config and add hidden layer attribute for easier extraction and vocab size modification later.
    # This only needs to be done once for the first training.
    config = BertConfig.from_pretrained(model_path)
    config.output_hidden_states = True

     # Modify to the new vocab size
    config.vocab_size = vocab_size
    config.save_pretrained(model_path)

    # Load the empty model
    model = BertForMaskedLM.from_pretrained(model_path,config=config, ignore_mismatched_sizes=True)

    # **2️⃣ Load dataset**
    # Read data
    with open(dataset_path, "r") as file:
        payloads = json.load(file)
    data = []
    for payload in payloads:
        flow = ""
        for j in range(3):
            flow += payload
        data.append(flow)

    dataset = Dataset.from_dict({"text": data})

    # **3️⃣ Define data preprocessing**

    # Call the main function
    tokenized_datasets = main(dataset, tokenizer, batch_size=1000, num_proc=30)
    # tokenized_datasets = tokenize_function(dataset)

    # **4️⃣ Create data loader for MLM task**
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Mask 15% of tokens
    )

    # **5️⃣ Training arguments**

    training_args = TrainingArguments(
        output_dir="C:\\Users\\Administrator\\bertgnn\\model\\bert\\pretrained_tinybert",
        # evaluation_strategy="epoch",      # Evaluate once per epoch
        save_strategy="epoch",            # Save once per epoch
        logging_steps=500,
        save_total_limit=2,
        per_device_train_batch_size=64,   # 🚀 Batch size per GPU
        # per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,    # 🚀 Gradient accumulation, equivalent to batch size = 128
        num_train_epochs=10,
        learning_rate=5e-5,
        warmup_steps=1000,
        weight_decay=0.01,
        fp16=True,                        # 🚀 Enable mixed precision
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False, # 🚀 Suitable for DDP training to avoid errors
        logging_dir="C:\\Users\\Administrator\\bertgnn\\model\\bert\\pretrained_mlmbert\\logs",
        report_to="none"
    )

    # **6️⃣ Create Trainer**
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        # eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator
    )


    # Record start time
    start_time = datetime.now()

    # **7️⃣ Run training**
    trainer.train()

    # Record end time
    end_time = datetime.now()

    # Calculate execution time
    elapsed_time = end_time - start_time
    print(f"Code execution time: {elapsed_time}")
# **1️⃣ Load BERT MLM pre-trained model**
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
vocab_size=tokenizer.vocab_size

# Load model config and add hidden layer attribute for easier extraction and vocab size modification later.
# This only needs to be done once for the first training.
config = BertConfig.from_pretrained(model_path)
config.output_hidden_states = True

 # Modify to the new vocab size
config.vocab_size = vocab_size
config.save_pretrained(model_path)

# Load the empty model
model = BertForMaskedLM.from_pretrained(model_path,config=config, ignore_mismatched_sizes=True)

# **2️⃣ Load dataset**
# Read data
with open(dataset_path, "r") as file:
    payloads = json.load(file)
data = []
for payload in payloads:
    flow = ""
    for j in range(3):
        flow += payload
    data.append(flow)

dataset = Dataset.from_dict({"text": data})

# **3️⃣ Define data preprocessing**

# Call the main function
tokenized_datasets = main(dataset, tokenizer, batch_size=1000, num_proc=30)
# tokenized_datasets = tokenize_function(dataset)

# **4️⃣ Create data loader for MLM task**
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # Mask 15% of tokens
)

# **5️⃣ Training arguments**

training_args = TrainingArguments(
    output_dir="C:\\Users\\Administrator\\bertgnn\\model\\bert\\pretrained_tinybert",
    # evaluation_strategy="epoch",      # Evaluate once per epoch
    save_strategy="epoch",            # Save once per epoch
    logging_steps=500,
    save_total_limit=2,
    per_device_train_batch_size=64,   # 🚀 Batch size per GPU
    # per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,    # 🚀 Gradient accumulation, equivalent to batch size = 128
    num_train_epochs=10,
    learning_rate=5e-5,
    warmup_steps=1000,
    weight_decay=0.01,
    fp16=True,                        # 🚀 Enable mixed precision
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False, # 🚀 Suitable for DDP training to avoid errors
    logging_dir="C:\\Users\\Administrator\\bertgnn\\model\\bert\\pretrained_mlmbert\\logs",
    report_to="none"
)

# **6️⃣ Create Trainer**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    # eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator
)


# Record start time
start_time = datetime.now()

# **7️⃣ Run training**
trainer.train()

# Record end time
end_time = datetime.now()

# Calculate execution time
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time}")
