"""
Preprocessing script for DrugBank DDI dataset from TDC.
"""
import pickle
from pathlib import Path
from tdc.multi_pred import DDI

def preprocess_drugbank():
    print("Loading DrugBank DDI dataset from TDC...")
    data = DDI(name='DrugBank')
    split = data.get_split()
    
    # Extract train, val, test DataFrames
    train_df = split['train']
    val_df = split['valid']
    test_df = split['test']
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Checking column names
    print(f"\nColumns: {train_df.columns.tolist()}")
    print(f"Sample row:\n{train_df.iloc[0]}")
    
    # Save as pickle files
    output_dir = Path(__file__).parent
    
    print("\nSaving preprocessed data...")
    with open(output_dir / 'train_reactions.pkl', 'wb') as f:
        pickle.dump(train_df, f)
    
    with open(output_dir / 'valid_reactions.pkl', 'wb') as f:
        pickle.dump(val_df, f)
    
    with open(output_dir / 'test_reactions.pkl', 'wb') as f:
        pickle.dump(test_df, f)
    
    print(f"✓ Saved train_reactions.pkl")
    print(f"✓ Saved valid_reactions.pkl")
    print(f"✓ Saved test_reactions.pkl")

if __name__ == '__main__':
    preprocess_drugbank()
