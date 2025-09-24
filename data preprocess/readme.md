# split_pcap.py 

Modify the **datapath_list** in the file to include your own directory paths. Each specified path should contain a splitcap subdirectory, where traffic flows have been separated. Inside splitcap, there are multiple subfolders, each named after a class label. Each subfolder contains multiple PCAP files representing flows of that specific class.

# feature_extract.py

Modify the **datapath_list** in the file to include your own directory paths. The payload features and statitical features will be saved in the folder **datapath_list\\splitcap\\payload.jsonl** and **datapath_list\\splitcap\\all_features.jsonl**.Then the data in **all_features.jsonl** will be standardized and be saved in **datapath_list\\splitcap\\all_feature.npz**.

# get_bert_pretrain_data.py

Modify the **datapath_list** in the file to include your own directory paths which contains **\\splitcap\\payload.jsonl**. All the payloads from all the datasets will be conbined and saved in a single file **bert_pretrain_payloads.json**.