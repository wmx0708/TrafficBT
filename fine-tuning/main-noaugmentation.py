import os
import torch
from torch.utils.data import DataLoader, random_split
from data import NetworkFlowDataset, load_flow_data,TransformerPacketDataset,get_balanced_trans_dataset,get_balanced_masked_bert_dataset
from data import balance_and_augment_data,augment_and_balance,CombinedFlowDataset
from models import TextFeatureExtractor,TransformerClassifier,FusionModel,UnifiedFlowModel
from trainer import FlowTrainer
from transformers import BertTokenizer
import numpy as np
import swanlab
from transformers import get_scheduler
import torch.nn as nn
from sklearn.model_selection import train_test_split
import argparse
import torch_optimizer as optim

# Configuration
config = {
    "bert_path": "/root/netgpt/model/pretrained_mlmbert/checkpoint-118479",
    "data_path": "/root/autodl-tmp/data/USTC-TFC2016-master/Malware/",
    "tokenizer_path":"/root/netgpt/vocab/bert_tokenizer",
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 32,
    "text_lr": 3e-5,
    "trans_lr": 8e-5,
    "fusion_lr":5e-6,
    "epochs": 30,
    "feature_num":28,
    "target_count":1000, # Set the number of samples for each class
    "trans_max_epoch":30,
    "text_max_epoch" :10,
    "fusion_max_epoch":20,
    "trans_patience":10,
    "trans_hidden_size":256,
    "fusion_hidden_size":256,
    "temperature":0.5, # A temperature coefficient τ can be introduced after Cross-Attention or feature fusion to scale the influence of BERT features
    "bert_use_multiclassifier":False,
    "use_parallel":False,
    "start_freeze_layer":8,
    "use_unfreeze":False,
    "continue_train":False
}

def save_model(model, optimizer,scheduler, epoch, model_name="model.pth",use_parallel = False):
    if use_parallel:
        """ Save the state of the model and optimizer """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.module.state_dict(),
            'scheduler_state_dict': scheduler.module.state_dict(),  # ✅ Save scheduler
        }
    else:
        """ Save the state of the model and optimizer """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # ✅ Save scheduler
        }
    torch.save(checkpoint, model_name)
    print(f"Model saved to {model_name}")


def load_model(model, optimizer,scheduler,device, model_name="model.pth",use_parallel = False):
    """ Load the state of the model and optimizer """
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
     # Move optimizer state to cuda
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    start_epoch = checkpoint['epoch']
    print(f"Model loaded from {model_name}")
    return start_epoch,model,optimizer,scheduler

def main():
    start_epoch = 0
    device = config["device"]

    # Truncate these samples
    data = np.load(config["data_path"]+"/splitcap/all_feature.npz",allow_pickle=True)
    packet_sequences = data["sequences"]
    flow_labels = data["labels"]
    flow_features = data["stat_features"]
    # # Load flow-level statistical feature data
    # flow_data = np.load(config["data_path"]+"splitcap/all_featueres.npz")
    # flow_features = flow_data["features"]
    # flow_labels = flow_data["labels"]
    
    # # Consistency check (optional)
    # print("Label distribution:", np.unique(packet_labels, return_counts=True))
    # print("Label distribution:", np.unique(flow_labels, return_counts=True))
    # assert (packet_labels == flow_labels).all(), "Labels of the two modalities are inconsistent"
    
    # Transformer dataset
    # X1_bal,  y_bal = augment_and_balance(flow_features,flow_labels, target_count=config["target_count"], noise_level=0.01)
    # X2_bal,  y_bal = balance_and_augment_data(packet_sequences, flow_labels, config["target_count"], noise_level=0.05, mask_prob=0.1, shuffle_prob=0.1)
    X1_bal, X2_bal, y_bal = flow_features,packet_sequences,flow_labels
    # Split into training and validation sets
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1_bal, X2_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=43
    )
    trans_train_set = CombinedFlowDataset(X1_train, X2_train, y_train)
    trans_val_set = CombinedFlowDataset(X1_val, X2_val, y_val)
    
    trans_train_loader = DataLoader(trans_train_set,batch_size=config["batch_size"], shuffle=False,drop_last = True)
    trans_val_loader = DataLoader(trans_val_set, batch_size=config["batch_size"],shuffle=False,drop_last = True)


    samples, labels, label_dict = load_flow_data(config["data_path"])
    bert_y = [y for x,y in samples]
    print("Label distribution:", np.unique(flow_labels, return_counts=True))
    print("Label distribution:", np.unique(bert_y, return_counts=True))
    assert (flow_labels == bert_y).all(), "Labels of the two modalities are inconsistent"

    # BERT dataset
    num_classes = len(labels)
    if num_classes > 20: # If the number of classes is greater than 20, use a custom classifier
        config["bert_use_multiclassifier"] = True
    print(f"Total num classe is {num_classes}")
    

    tokenizer = BertTokenizer.from_pretrained(config["tokenizer_path"])
    # new_samples = get_balanced_masked_bert_dataset(
    #     samples, tokenizer=tokenizer,
    #     target_count=config["target_count"], mask_prob=0.2
    # )
    new_samples = samples
    bert_train, bert_val = train_test_split(
        new_samples ,test_size=0.2, stratify=y_bal, random_state=43
    )
    train_set = NetworkFlowDataset(bert_train, tokenizer) #"input_ids""attention_mask" "label"
    val_set = NetworkFlowDataset(bert_val, tokenizer)
    
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False,drop_last = True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,drop_last = True)

    #
    # trans_train_loader = DataLoader(TransformerPacketDataset(tras_train_X, tras_train_y), batch_size=config["batch_size"], shuffle=False)
    # trans_test_loader = DataLoader(TransformerPacketDataset(trans_val_X, trans_val_y), batch_size=config["batch_size"], shuffle=False)

    # Model initialization
    # Load BERT model, Transformer model, and fusion model
    text_model  = TextFeatureExtractor(config["bert_path"], num_classes,config["bert_use_multiclassifier"],dropout_prob = 0.5,start_freeze_layer = config["start_freeze_layer"])

    # trans_model = TransformerClassifier(config["feature_num"], num_classes, num_heads=4, num_layers=2, hidden_dim=config["trans_hidden_size"], dropout=0.2)
    trans_model = UnifiedFlowModel(stat_feat_dim = 42, seq_feat_dim=28, seq_len=100, hidden_dim=config["trans_hidden_size"], num_classes=num_classes)
    
    fusion_model = FusionModel(hidden_size=config["fusion_hidden_size"], trans_hidden_size = config["trans_hidden_size"],temperature = config["temperature"],num_classes=num_classes,dropout = 0.3)

    # Load optimizer
    # Ensure the optimizer only updates trainable parameters
    optimizer_grouped_parameters = [
    {"params": text_model.bert.bert.encoder.layer[:8].parameters(), "lr": config["text_lr"]/5},
    {"params": text_model.bert.bert.encoder.layer[8:].parameters(), "lr": config["text_lr"]},
    {"params": text_model.bert.classifier.parameters(), "lr": config["text_lr"]*2},
]
    text_optim = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    trans_optim = torch.optim.AdamW(
        trans_model.parameters(),
        weight_decay=0.01,
        lr=config["trans_lr"],
    )

    # # Replace AdamW with RAdam
    # fusion_optim = optim.RAdam(
    #     fusion_model.parameters(),
    #     lr=config["fusion_lr"]
    # )
    fusion_optim = torch.optim.AdamW(
                fusion_model.parameters(),
                lr=config["fusion_lr"]
            )
    # Learning rate scheduler
    num_training_steps = config["epochs"] * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    text_scheduler = get_scheduler(
        name="cosine",
        optimizer=text_optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    trans_scheduler = get_scheduler(
        name="cosine",
        optimizer=trans_optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    fusion_scheduler = get_scheduler(
        name="cosine",
        optimizer=fusion_optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
#     fusion_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     fusion_optim, mode='min', factor=0.5, patience=2, verbose=True
# )
    # If continuing training from a previous session
    if config["continue_train"]:
        start_epoch,text_model,text_optim,text_scheduler = load_model(text_model, text_optim,text_scheduler, config["device"],model_name = config["data_path"]+"/splitcap/text_model.pth",use_parallel = config["use_parallel"])
        start_epoch,trans_model,trans_optim,trans_scheduler = load_model(trans_model, trans_optim,trans_scheduler, config["device"],model_name = config["data_path"]+"/splitcap/trans_model.pth",use_parallel = config["use_parallel"])
        start_epoch,fusion_model,fusion_optim,fusion_scheduler = load_model(fusion_model, fusion_optim, fusion_scheduler,config["device"],model_name = config["data_path"]+"/splitcap/fusion_model.pth",use_parallel = config["use_parallel"])
    # If using multiple GPUs for training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        config["use_parallel"] = True
        text_model  = nn.DataParallel(text_model)
        trans_model = nn.DataParallel(trans_model)
        fusion_model = nn.DataParallel(fusion_model)
        
    text_model = text_model.to(device)
    trans_model = trans_model.to(device)
    fusion_model = fusion_model.to(device)


    # Initialize SwanLab experiment
    swanlab.init(
        project = "bert-triformer",
        experiment_name=config["data_path"].split("/")[-3] +"-" +config["data_path"].split("/")[-2],
        config={
            "model_type": "fusion_model",
            "device": str(device)
        }
    )
    # If a previously activated BERT is available, use it directly
    if os.path.exists(config["data_path"]+"/splitcap/fintuned_bert.pth"):
        text_model.load_state_dict(torch.load(config["data_path"]+"/splitcap/fintuned_bert.pth"))
    trainer = FlowTrainer(num_classes,text_model, trans_model,fusion_model,text_optim,trans_optim,fusion_optim,text_scheduler,trans_scheduler,fusion_scheduler, device,config["use_parallel"],config["start_freeze_layer"],config["use_unfreeze"],swanlab)
    # If continuing from previous training
    if config["continue_train"]:
        text_model.enable_fintuning(11)
        trainer = FlowTrainer(num_classes,text_model, trans_model,fusion_model,text_optim,trans_optim,fusion_optim,text_scheduler,trans_scheduler,fusion_scheduler, device,config["use_parallel"],config["start_freeze_layer"],config["use_unfreeze"],swanlab)
        for epoch in range(start_epoch,start_epoch+config["epochs"]):
            train_metrics, _, _,text_model,trans_model,fusion_model = trainer.train_epoch(
                train_loader,
                trans_train_loader
            )
    
            val_metrics, preds, labels = trainer.evaluate(val_loader,trans_val_loader)
    
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            save_model(text_model, text_optim,text_scheduler, epoch, model_name=config["data_path"]+"/splitcap/text_model.pth",use_parallel = config["use_parallel"])
            if trans_model:
                save_model(trans_model, trans_optim, trans_scheduler,epoch, model_name=config["data_path"]+"/splitcap/trans_model.pth",use_parallel = config["use_parallel"])
            if fusion_model:
                save_model(fusion_model, fusion_optim,fusion_scheduler, epoch, model_name=config["data_path"]+"/splitcap/fusion_model.pth",use_parallel = config["use_parallel"])
            # if (epoch)%5==0 or epoch == config["epochs"]:
            #     save_model(text_model, text_optim,text_scheduler, epoch, model_name=config["data_path"]+"/splitcap/text_model.pth",use_parallel = config["use_parallel"])
            #     if trans_model:
            #         save_model(trans_model, trans_optim, trans_scheduler,epoch, model_name=config["data_path"]+"/splitcap/trans_model.pth",use_parallel = config["use_parallel"])
            #     if fusion_model:
            #         save_model(fusion_model, fusion_optim,fusion_scheduler, epoch, model_name=config["data_path"]+"/splitcap/fusion_model.pth",use_parallel = config["use_parallel"])
    # If training from scratch
    else:
        if os.path.exists(config["data_path"]+"/splitcap/fintuned_bert.pth"):
            text_model.load_state_dict(torch.load(config["data_path"]+"/splitcap/fintuned_bert.pth"))
        trainer = FlowTrainer(num_classes,text_model, trans_model,fusion_model,text_optim,trans_optim,fusion_optim,text_scheduler,trans_scheduler,fusion_scheduler, device,config["use_parallel"],config["start_freeze_layer"],config["use_unfreeze"],swanlab)
        trainer.warmup_trans_model(trans_train_loader,max_epochs=config["trans_max_epoch"],val_loader=trans_val_loader, patience=config["trans_patience"])
        if not os.path.exists(config["data_path"]+"/splitcap/fintuned_bert.pth"):
            trainer.warmup_text_model(train_loader, max_epochs=config["text_max_epoch"],val_loader=val_loader, patience=config["trans_patience"])
            if config["use_parallel"]:
                torch.save(text_model.module.state_dict(), config["data_path"]+"/splitcap/fintuned_bert.pth")
            else:
                torch.save(text_model.state_dict(), config["data_path"]+"/splitcap/fintuned_bert.pth")
        trainer.warmup_fusion_model(train_loader, trans_train_loader, max_epochs=config["fusion_max_epoch"])
        # text_model.enable_finetuning(11)
        # trainer = FlowTrainer(num_classes,text_model, trans_model,fusion_model,text_optim,trans_optim,fusion_optim,text_scheduler,trans_scheduler,fusion_scheduler, device,config["use_parallel"],config["start_freeze_layer"],config["use_unfreeze"],swanlab)
        for epoch in range(start_epoch,start_epoch+config["epochs"]):
            train_metrics, _, _,text_model,trans_model,fusion_model = trainer.train_epoch(
                train_loader,
                trans_train_loader
            )
    
            val_metrics, preds, labels = trainer.evaluate(val_loader,trans_val_loader)
    
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
            save_model(text_model, text_optim,text_scheduler, epoch, model_name=config["data_path"]+"/splitcap/text_model.pth",use_parallel = config["use_parallel"])
            if trans_model:
                save_model(trans_model, trans_optim, trans_scheduler,epoch, model_name=config["data_path"]+"/splitcap/trans_model.pth",use_parallel = config["use_parallel"])
            if fusion_model:
                save_model(fusion_model, fusion_optim,fusion_scheduler, epoch, model_name=config["data_path"]+"/splitcap/fusion_model.pth",use_parallel = config["use_parallel"])
            # if (epoch)%5==0 or epoch == config["epochs"]:
            #     save_model(text_model, text_optim,text_scheduler, epoch, model_name=config["data_path"]+"/splitcap/text_model.pth",use_parallel = config["use_parallel"])
            #     if trans_model:
            #         save_model(trans_model, trans_optim, trans_scheduler,epoch, model_name=config["data_path"]+"/splitcap/trans_model.pth",use_parallel = config["use_parallel"])
            #     if fusion_model:
            #         save_model(fusion_model, fusion_optim,fusion_scheduler, epoch, model_name=config["data_path"]+"/splitcap/fusion_model.pth",use_parallel = config["use_parallel"])
           
    swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ServiceVPN", action="store_true", help="Use Service-VPN Datasets")    
    parser.add_argument("--ServiceNonVPN", action="store_true", help="Use Service-NonVPN Datasets")    
    parser.add_argument("--AppVPN", action="store_true", help="Use App-VPN Datasets")
    parser.add_argument("--AppNonVPN", action="store_true", help="Use App-NonVPN Datasets")
    parser.add_argument("--Tor", action="store_true", help="Use Tor Datasets")
    parser.add_argument("--NonTor", action="store_true", help="Use NonTor Datasets")
    parser.add_argument("--Benign", action="store_true", help="Use Benign Datasets")
    parser.add_argument("--Malware", action="store_true", help="Use Malware Datasets")
    parser.add_argument("--Flood", action="store_true", help="Use Flood Datasets")
    parser.add_argument("--RTSPBruteForce", action="store_true", help="Use RTSPBruteForce Datasets")
    parser.add_argument("--datacon2020", action="store_true", help="Use datacon2020 Datasets")
    parser.add_argument("--datacon2021part1", action="store_true", help="Use datacon2021part1 Datasets")
    parser.add_argument("--datacon2021part2", action="store_true", help="Use datacon2021part2 Datasets")
    parser.add_argument("--CrossPlatformandroid", action="store_true", help="Use CrossPlatformandroid Datasets")
    parser.add_argument("--CrossPlatformios", action="store_true", help="Use CrossPlatformios Datasets")
    parser.add_argument("--NUDT", action="store_true", help="Use NUDT Datasets")
    
    args = parser.parse_args()
    if args.ServiceVPN:
        config["data_path"] = "/root/autodl-tmp/data/ISCX-VPN-Service/VPN/"
    
    if args.ServiceNonVPN:
        config["data_path"] = "/root/autodl-tmp/data/ISCX-VPN-Service/NonVPN/"
    
    if args.AppVPN:
        config["data_path"] = "/root/autodl-tmp/data/ISCX-VPN-App/VPN/"

    if args.AppNonVPN:
        config["data_path"] = "/root/autodl-tmp/data/ISCX-VPN-App/NonVPN/"
    
    if args.Tor:
        config["data_path"] = "/root/autodl-tmp/data/ISCX-Tor/Tor/"

    if args.NonTor:
        config["data_path"] = "/root/autodl-tmp/data/ISCX-Tor/NonTor/"

    if args.Benign:
        config["data_path"] = "/root/autodl-tmp/data/USTC-TFC2016-master/Benign/"

    if args.Malware:
        config["data_path"] = "/root/autodl-tmp/data/USTC-TFC2016-master/Malware/"

    if args.Flood:
        config["data_path"] = "/root/autodl-tmp/data/CIC_IOT_Dataset2022_Attacks/Flood/"

    if args.RTSPBruteForce:
        config["data_path"] = "/root/autodl-tmp/data/CIC_IOT_Dataset2022_Attacks/RTSP-Brute-Force/"

    if args.datacon2020:
        config["data_path"] = "/root/autodl-tmp/data/datacon2020/train/"

    if args.datacon2021part1:
        config["data_path"] = "/root/autodl-tmp/data/datacon2021/part1/test/"

    if args.datacon2021part2:
        config["data_path"] = "/root/autodl-tmp/data/datacon2021/part2/train/"

    if args.CrossPlatformandroid:
        config["data_path"] = "/root/autodl-tmp/data/CrossPlatform/android/"

    if args.CrossPlatformios:
        config["data_path"] = "/root/autodl-tmp/data/CrossPlatform/ios/"
    
    if args.NUDT:
        config["data_path"] = "/root/autodl-tmp/data/NUDT/merge/"
    main()
