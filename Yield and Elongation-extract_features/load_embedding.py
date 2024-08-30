import pandas as pd

def embeddings_dict(content):
    CLS = content['features'][0]["layers"][0]["values"]
    return CLS

def main(_):
    lines = [] 
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = eval(line)
            lines.append(line)
        f.close()

    CLSs = []
    for i in range(len(lines)):
        content = lines[i]
        CLS = embeddings_dict(content)
        CLSs.append(CLS)

    CLSs = pd.DataFrame(CLSs)
    CLSs.to_excel('/Users/wangping/Desktop/pythonProject/extract_features/Verified by existing alloys.xlsx', index=False)


if __name__ == "__main__":
    file = "/Users/wangping/Desktop/pythonProject/extract_features/Verified by existing alloys.json"
    main(file)
