import pandas as pd

if __name__ == '__main__':
    stock_index = pd.read_csv(r"data/custom/stock_index.csv").rename(columns={
        'Open' : 'SI_Open',
        'High' : 'SI_High',
        'Low'  : 'SI_Low',
        'Close': 'SI_Close',
    })
    adapted_6269 = pd.read_csv(r"data/custom/6269_adapt.csv").drop(columns=['Date'])
    
    combined_data = pd.concat([stock_index, adapted_6269], axis=1)
    combined_data.to_csv(r"data/custom/combined.csv", index=False)