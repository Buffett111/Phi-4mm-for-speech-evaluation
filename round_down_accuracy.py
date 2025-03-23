import json

def custom_round(value):
    value = float(value)
    if value - int(value) < 0.5:
        return float(int(value) + 0)
    return float(int(value))

def load_and_process(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data['all_labels'])} samples")
    # preds = [custom_round(x) for x in data["all_generated_texts"]]
    # labels = [custom_round(x) for x in data["all_labels"]]
    preds = data["all_generated_texts"]
    labels = data["all_labels"]
    for x, y in zip(data["all_labels"], data["all_generated_texts"]):
        print(f"Labels: {x} Predictions: {y}")
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = correct / len(labels)

    print(f"Adjusted Accuracy: {accuracy:.6f} ({correct}/{len(labels)})")

if __name__ == "__main__":
    # 把你的 JSON 檔案命名為 data.json，或改成你實際的檔案名
    load_and_process("./LTTC-Intermediate/IS-1964/phi-4-multimodal-instruct-lttc-NoQA-NoImage_0323/eval_before.json")
