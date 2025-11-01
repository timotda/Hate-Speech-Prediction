import pandas as pd
from sklearn.utils import shuffle

# Define the input and output file paths
input_csv_path = '/home/federico/EPFL/DeepLearningProject/EPFL-EE559-HateSpeechDetection/data/data_olid_clean_double.csv'
output_csv_path = '/home/federico/EPFL/DeepLearningProject/EPFL-EE559-HateSpeechDetection/data/sampled_data.csv'

# Number of samples per class
n_samples_per_class = 2000

try:
    # Load the dataset
    df = pd.read_csv(input_csv_path)

    # Separate classes
    df_class_0 = df[df['label'] == 0]
    df_class_1 = df[df['label'] == 1]

    # Sample from each class
    # Use random_state for reproducibility if needed
    sampled_class_0 = df_class_0.sample(n=n_samples_per_class, random_state=42)
    sampled_class_1 = df_class_1.sample(n=n_samples_per_class, random_state=42)

    # Concatenate the samples
    sampled_df = pd.concat([sampled_class_0, sampled_class_1])

    # Shuffle the combined dataframe
    shuffled_df = shuffle(sampled_df, random_state=42)

    # Reset index
    shuffled_df = shuffled_df.reset_index(drop=True)

    # Display info about the new dataframe
    print("Shape of the new sampled and shuffled dataframe:", shuffled_df.shape)
    print("\nFirst 5 rows of the new dataframe:")
    print(shuffled_df.head())
    print("\nClass distribution in the new dataframe:")
    print(shuffled_df['label'].value_counts())

    # Save the new dataframe to a CSV file
    shuffled_df.to_csv(output_csv_path, index=False)
    print(f"\nSampled and shuffled data saved to: {output_csv_path}")

except FileNotFoundError:
    print(f"Error: The file {input_csv_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you have enough samples in each class (at least 2000).")

import pandas as pd
from sklearn.utils import shuffle

# Define the input and output file paths
input_csv_path = '/home/federico/EPFL/DeepLearningProject/EPFL-EE559-HateSpeechDetection/data/data_olid_clean.csv'
output_csv_path = '/home/federico/EPFL/DeepLearningProject/EPFL-EE559-HateSpeechDetection/data/sampled_data.csv'

# Number of samples per class
n_samples_per_class = 2000

try:
    # Load the dataset
    df = pd.read_csv(input_csv_path)

    # Separate classes
    df_class_0 = df[df['label'] == 0]
    df_class_1 = df[df['label'] == 1]

    # Sample from each class
    # Use random_state for reproducibility if needed
    sampled_class_0 = df_class_0.sample(n=n_samples_per_class, random_state=42)
    sampled_class_1 = df_class_1.sample(n=n_samples_per_class, random_state=42)

    # Concatenate the samples
    sampled_df = pd.concat([sampled_class_0, sampled_class_1])

    # Shuffle the combined dataframe
    shuffled_df = shuffle(sampled_df, random_state=42)

    # Reset index
    shuffled_df = shuffled_df.reset_index(drop=True)

    # Display info about the new dataframe
    print("Shape of the new sampled and shuffled dataframe:", shuffled_df.shape)
    print("\nFirst 5 rows of the new dataframe:")
    print(shuffled_df.head())
    print("\nClass distribution in the new dataframe:")
    print(shuffled_df['label'].value_counts())

    # Save the new dataframe to a CSV file
    shuffled_df.to_csv(output_csv_path, index=False)
    print(f"\nSampled and shuffled data saved to: {output_csv_path}")

except FileNotFoundError:
    print(f"Error: The file {input_csv_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you have enough samples in each class (at least 2000).")
