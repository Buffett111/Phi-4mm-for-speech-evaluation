from cProfile import label
import json

def custom_round(value):
    value = float(value)
    # X.5 -> X, X.6 -> X+1
    if value - int(value) <= 0.5:
        return float(int(value) + 0)
    return float(int(value))

def load_and_process(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data['predictions_and_labels'])} samples")
    preds = [custom_round(x['prediction']) for x in data['predictions_and_labels']]
    labels = [custom_round(x['label']) for x in data['predictions_and_labels']]
    
    for item in data['predictions_and_labels']:
        print(f"Labels: {item['label']} Predictions: {item['prediction']}")
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = correct / len(labels)

    print(f"Adjusted Accuracy: {accuracy:.6f} ({correct}/{len(labels)})")

if __name__ == "__main__":
    # 把你的 JSON 檔案命名為 data.json，或改成你實際的檔案名
    print("1764")
    load_and_process("./LTTC-Intermediate/(SOTA)Phi-4-mm_QA_NoImage_0325_1964/IS-1764/Phi-4-multimodal-instruct_QA_NoImage_0325_1964_LTTC-Dev-1764-0520_train.json")
    print("1964")
    load_and_process("./LTTC-Intermediate/(SOTA)Phi-4-mm_QA_NoImage_0325_1964/IS-1964/Phi-4-multimodal-instruct_QA_NoImage_0325_1964_LTTC-Dev-1964-0520_train.json")