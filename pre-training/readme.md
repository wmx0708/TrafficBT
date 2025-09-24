# \\vacab\\get_tokenizer.py

(1) **tokenizer_path** saves the trained tokenizer file(generate the **vocab.txt** for bert_tokenizer, no use for bert)
(2) **bert_tokenizer_path** saves the bert_tokenizer files for bert tokenization(use this tokenizer)
(3) **pretrain_data_path** saves the **bert_pretrain_payloads.json** generated in the **get_bert_pretrain_data.py** in the directory  **data_process**
(4) **corpora_path** saves the corpora file(.txt) transformed from **bert_pretrain_payloads.json** to train the tokenizer use BPE.
(5) **vocab_path** saves the vocab.txt trained from BPE.

Modify all the path to include your own directory paths.



# bert_pretrain_multigpu.py 

(1)**model_path** you can download model.pth and conifg.json from bert_base in Huggingface
(2)**tokenizer_path** the **bert_tokenizer_path** above
(3)**dataset_path** the **bert_pretrain_payloads.json** generated in the **get_bert_pretrain_data.py** in the directory  **data_process**
(4)**output_dir** the pre-trained bert save_path

Modify all the path and change the parameters in the training_args you need. 