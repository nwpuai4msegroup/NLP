import pandas as pd
import json
def embeddings_dict(content):

    CLS = content['features'][0]["layers"][0]["values"]

    yield_demension = [8, 34, 174, 241, 246, 269, 376, 380, 420, 423, 563, 564, 722]
    E_demension = [116, 153, 221, 249, 334, 507, 536, 612, 678, 696, 701, 749, 766]

    embedding_yield = [CLS[i-1] for i in yield_demension]
    #embedding_tensile = [CLS[i-1] for i in tensile_demension]
    embedding_E = [CLS[i-1] for i in E_demension]
    return embedding_yield, embedding_E
    

def batch_process_and_write(batch_data):
    processed_batch = [embeddings_dict(item) for item in batch_data]
    yield_em = [item[0] for item in processed_batch]
    E_em = [item[1] for item in processed_batch]
    yield_em = pd.DataFrame(yield_em)
    E_em = pd.DataFrame(E_em)
    print(yield_em)
    print(E_em)
    with pd.ExcelWriter("/home/wangping/paleituo/yield_MLP_450.xlsx", engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer1:
        yield_em.to_excel(writer1, index=False, header=False, startrow=writer1.sheets['Sheet1'].max_row)
    with pd.ExcelWriter("/home/wangping/paleituo/E_MLP_450.xlsx", engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer2:
        E_em.to_excel(writer2, index=False, header=False, startrow=writer2.sheets['Sheet1'].max_row)


def main(_):


    with pd.ExcelWriter("/home/wangping/paleituo/yield_MLP_450.xlsx", engine='openpyxl', mode='w') as writer1:
        pd.DataFrame(columns=["X8", "X34","X174","X241","X246","X269","X376","X380","X420","X423","X563","X564","X722"]).to_excel(writer1, index=False)
    with pd.ExcelWriter("/home/wangping/paleituo/E_MLP_450.xlsx", engine='openpyxl',mode='w') as writer2:
        pd.DataFrame(columns=["X116","X153","X221","X249","X334","X507","X536","X612","X678","X696","X701","X749","X766"]).to_excel(writer2, index=False)


    batch_size = 10000
    batch_data = []

    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            batch_data.append(data)
            if len(batch_data) >= batch_size:
                batch_process_and_write(batch_data)
                batch_data = []

    if batch_data:
        batch_process_and_write(batch_data)

if __name__ == "__main__":
    file = "/home/wangping/exfeatures/candidates_all_em.json"
    main(file)






