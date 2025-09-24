import json


if __name__ == "__main__":
    datapath_list = [
        "D:\\ISCX-VPN-Service\\VPN\\",
        "D:\\ISCX-VPN-Service\\NonVPN\\",
        "D:\\ISCX-Tor\\Tor\\",
        "D:\\ISCX-Tor\\NonTor\\",
        "D:\\CIRA-CIC-DoHBrw-2020\\detection\\",
        "D:\\NUDT_MobileTraffic\\data1\\",
        "D:\\NUDT_MobileTraffic\\data2\\",
        "D:\\NUDT_MobileTraffic\\data3\\",
        "D:\\NUDT_MobileTraffic\\data4\\",
    ]

    all_payloads = []

    # 遍历每个 jsonl 文件路径
    for path in datapath_list:
        path = path +"\\splitcap\\+payload.jsonl"
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if "payloads" in data:
                    all_payloads.extend(data["payloads"])

    # 保存结果为一个 JSON 文件
    with open("bert_pretrain_payloads.json", "w", encoding="utf-8") as out_f:
        json.dump(all_payloads, out_f, indent=2)