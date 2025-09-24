from transformers import BertForMaskedLM, BertTokenizer,BertConfig,Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset,Dataset,concatenate_datasets
import torch
import json
import multiprocessing
from tqdm import tqdm
from datetime import datetime

# 定义 tokenize 函数
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# 多进程处理函数
def process_batch(args):
    batch, tokenizer = args
    return Dataset.from_dict(tokenize_function(batch, tokenizer))

# 主函数
def main(dataset, tokenizer, batch_size=1000, num_proc=6):
    # 将数据集分块
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

    # 创建进程池
    pool = multiprocessing.Pool(processes=num_proc)

    # 准备多进程参数（将 tokenizer 传递给每个进程）
    task_args = [(batch, tokenizer) for batch in batches]

    # 使用 tqdm 显示进度条
    tokenized_batches = []
    with tqdm(total=len(batches), desc="Tokenizing") as pbar:
        for result in pool.imap(process_batch, task_args):
            tokenized_batches.append(result)
            pbar.update(1)

    # 关闭进程池
    pool.close()
    pool.join()

    # 合并处理后的结果
    tokenized_datasets = concatenate_datasets(tokenized_batches)
    return tokenized_datasets

if __name__ == "__main__":
    #  模型，分词器混合数据路径
    model_path="model\\bert\\bert_base"
    tokenizer_path="vocab\\bert_tokenizer"
    dataset_path="bert_pretrain_payloads.json"

    # **1️⃣ 加载 BERT MLM 预训练模型**
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    vocab_size=tokenizer.vocab_size

    #加载模型config并增加隐藏层属性，方便后续提取以及修改词库大小,第一次训练修改后就可以不添加了
    config = BertConfig.from_pretrained(model_path)
    config.output_hidden_states = True

     # 修改为新的词库大小
    config.vocab_size = vocab_size
    config.save_pretrained(model_path)

    # 加载空模型
    model = BertForMaskedLM.from_pretrained(model_path,config=config, ignore_mismatched_sizes=True)

    # **2️⃣ 加载数据集**
    #读取数据
    with open(dataset_path, "r") as file:
        payloads = json.load(file)
    data = []
    for payload in payloads:
        flow = ""
        for j in range(3):
            flow += payload
        data.append(flow)

    dataset = Dataset.from_dict({"text": data})

    # **3️⃣ 定义数据预处理**

    # 调用主函数
    tokenized_datasets = main(dataset, tokenizer, batch_size=1000, num_proc=30)
    # tokenized_datasets = tokenize_function(dataset)

    # **4️⃣ 创建 MLM 任务的数据加载器**
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # Mask 15% 的 token
    )

    # **5️⃣ 训练参数**

    training_args = TrainingArguments(
        output_dir="C:\\Users\\Administrator\\bertgnn\\model\\bert\\pretrained_tinybert",
        # evaluation_strategy="epoch",      # 每个 epoch 评估一次
        save_strategy="epoch",            # 每个 epoch 保存一次
        logging_steps=500,
        save_total_limit=2,
        per_device_train_batch_size=64,   # 🚀 每个 GPU 的 batch size
        # per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,    # 🚀 累积梯度，相当于 batch size = 128
        num_train_epochs=10,
        learning_rate=5e-5,
        warmup_steps=1000,
        weight_decay=0.01,
        fp16=True,                        # 🚀 开启混合精度
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False, # 🚀 适用于 DDP 训练，避免报错
        logging_dir="C:\\Users\\Administrator\\bertgnn\\model\\bert\\pretrained_mlmbert\\logs",
        report_to="none"
    )

    # **6️⃣ 创建 Trainer**
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        # eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator
    )


    # 记录开始时间
    start_time = datetime.now()

    # **7️⃣ 运行训练**
    trainer.train()

    # 记录结束时间
    end_time = datetime.now()

    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time}")
