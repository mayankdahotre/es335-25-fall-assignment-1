import os
import numpy as np
import pandas as pd

workspace_root = os.path.dirname(os.path.dirname(__file__))
filepath = os.path.join(workspace_root, 'Datasets', 'UCI HAR Dataset')
with open(os.path.join(filepath, 'features.txt'), 'r') as f:
    lines = f.readlines()

features = []
exists = {}

for line in lines:
    new_line = line.replace('\n', '').replace('-', '_')[line.find(' ') + 1:]
    if new_line in exists:
        features.append(new_line + "_" + str(exists[new_line]))
        exists[new_line] += 1
    else:
        features.append(new_line)
        exists[new_line] = 1

def generate_dataset(folder: str = 'train') -> tuple[np.ndarray, np.ndarray]:
    dataset_path = os.path.join(filepath, folder)
    
    print(f"Loading feature data from {dataset_path}...")
    with open(os.path.join(dataset_path, f'X_{folder}.txt')) as X_file:
        lines = X_file.readlines()
    X = []
    for line in lines:
        row = np.array(line.split(), dtype=np.float64)
        X.append(row)
    X = np.array(X)

    with open(os.path.join(dataset_path, f'subject_{folder}.txt')) as subj_file:
        subject_lines = subj_file.readlines()
    subjects = np.array([int(subj.strip()) for subj in subject_lines], dtype=np.int32)

    with open(os.path.join(dataset_path, f'y_{folder}.txt')) as y_file:
        lines = y_file.readlines()
    y = np.array([int(line.strip()) for line in lines], dtype=np.int32) - 1  # Adjust for 1-based indexing

    X_df = pd.DataFrame(X, columns=features)
    X_df['Subject'] = pd.Series(subjects)
    X_df['y'] = pd.Series(y)

    subject_activity_counts = X_df.groupby(['Subject', 'y']).size().unstack(fill_value=0)
    # print(subject_activity_counts)

    min_rows_per_activity = X_df.groupby(['Subject', 'y']).size().min()

    trimmed_data = []
    y_labels = []
    subjects = X_df['Subject'].unique()
    
    for subject in subjects:
        subject_data = X_df[X_df['Subject'] == subject]
        subject_activities = []
        
        for activity in sorted(subject_data['y'].unique()):
            activity_data = subject_data[subject_data['y'] == activity].iloc[:min_rows_per_activity, :-2].values  
            subject_activities.append(activity_data)
            y_labels.append(activity)  # Append the activity label
            
        trimmed_data.append(np.stack(subject_activities))  
            
    X_trimmed, y_trimmed = np.stack(trimmed_data), np.array(y_labels)

    X_flattened = X_trimmed.reshape(-1, X_trimmed.shape[2], X_trimmed.shape[3])
    
    return X_flattened, y_trimmed

X_train, y_train = generate_dataset('train')
X_test, y_test = generate_dataset('test')
min_columns = min(X_train.shape[1], X_test.shape[1])

X_train = X_train[:, :min_columns]
X_test = X_test[:, :min_columns]

if __name__ == "__main__":
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
