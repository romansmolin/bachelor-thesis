import pandas as pd 
from app.utils import stemSentence

def clean_data(input_path, output_path):
    print("Script started...")

    column_names = ['cat', 'text']
    #Here we are loading raw data
    df = pd.read_csv(input_path, header=None, names=column_names, low_memory=False)
    #Applying our normalization function
    df['text'] = df['text'].apply(stemSentence)
    
    df.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    clean_data('data/ecommerceDataset.csv', 'data/normalized.csv')