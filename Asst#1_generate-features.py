import os
import pandas as pd
import tsfel
from pathlib import Path


def generate_features(input_base_dir, output_base_dir, activities):
    for activity in activities:
        activity_dir = os.path.join(input_base_dir, activity)
        output_activity_dir = os.path.join(output_base_dir, activity)
        Path(output_activity_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(activity_dir):
            print(f"Warning: {activity_dir} does not exist. Skipping.")
            continue
        subject_files = [f for f in os.listdir(activity_dir) if f.endswith('.csv')]
        for file in subject_files:
            file_path = os.path.join(activity_dir, file)
            df = pd.read_csv(file_path).iloc[100:600, :]
            cfg = tsfel.get_features_by_domain() 
            for domain in cfg:
                for feature in cfg[domain]:
                    cfg[domain][feature]['use'] = 'yes'
            features = tsfel.time_series_features_extractor(cfg, df, fs=50)
            subject_id = file.split('.')[0]
            output_file = os.path.join(output_activity_dir, f'{subject_id}.csv')
            features.to_csv(output_file, index=False)

activities = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']

# Generate features for Train set
generate_features('Datasets/Combined/Train', 'Datasets/TSFEL_3axes_allfeatures/Train', activities)
# Generate features for Test set
generate_features('Datasets/Combined/Test', 'Datasets/TSFEL_3axes_allfeatures/Test', activities)
