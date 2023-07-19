### 生成输出目录的模块
import os
from datetime import datetime

def generate_output(random_disturb: bool, filename):
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    # Generate the folder name with the current date
    now = datetime.now()
    if random_disturb:
        folder_name = f"{slurm_job_id}_Stoch_{now.strftime('%Y-%m-%d-%H')}"#-%H-%M
    else:
        folder_name = f"{slurm_job_id}Btwn_{now.strftime('%Y-%m-%d-%H')}"

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(output_dir, filename)

    return file_path

##################################
import csv

def save_scores_to_csv(random_disturb: bool, scores, filename):
    """
    Saves a list of list of list to a CSV file with a double space separator.

    Args:
        scores (list): The list of list of list to be saved.
        disturb (bool): A boolean indicating if disturbance is present.
        filename (str): The name of the output CSV file.
    """
    # Construct the full file path
    file_path = generate_output(random_disturb, filename + ".csv")

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        for line in scores:
            writer.writerow(line)
            file.write('\n')
        file.write('----------------------------------------\n')
        
###################################
import csv

def save_to_csv(random_disturb: bool, content, filename):
    """
    Saves a list of 4-lists to a CSV file with a double space separator.
    
    Args:
        scores (list): The list of 4-lists to be saved.
        filename (str): The name of the output CSV file.
    """
    # Construct the full file path
    file_path = generate_output(random_disturb, filename+".csv")

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        for content_list in content:
            writer.writerow(content_list)