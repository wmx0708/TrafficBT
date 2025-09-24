# main.py

(1) **tokenizer_path** the **bert_tokenizer_path** in pre-training readme. 
(2) **bert_path** the save_path of your pre-trained bert model
(3) **data_path** the path contains **datapath_list\\splitcap\\payload.jsonl** , **datapath_list\\splitcap\\all_features.jsonl** and **datapath_list\\splitcap\\all_feature.npz**,see detailed description in directory **data preprocess**.
(4) **device** and **use parallel** depends on your device configuration.
(5) **continue_train** If set to False, the model will be trained from scratch; if True, it will resume training from previously saved parameters

Modify all the path to include your own directory paths and justify the parameters you need like **text_lr**
or **trans_max_epoch**.

the args below represents the datapath, change you own args you need.
run like "python main.py --Servicevpn"