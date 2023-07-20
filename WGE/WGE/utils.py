### 生成输出目录的模块
import os
from datetime import datetime

def generate_output(random_disturb: bool, filename):
    """
    Generate the output file path based on the provided parameters.

    Parameters:
        random_disturb (bool): A flag indicating whether we use random disturbance.
        filename (str): The name of the file.

    Returns:
        str: The full file path.
    """
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    # Generate the folder name with the current date
    now = datetime.now()
    if random_disturb:
        folder_name = f"{slurm_job_id}_Stoch_{now.strftime('%Y-%m-%d')}"#-%H-%M
        filename = "Stoch_"+filename
    else:
        folder_name = f"{slurm_job_id}_Btwn_{now.strftime('%Y-%m-%d')}"
        filename = "Btwn_"+filename

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

    Parameters:
        scores (list): The list of list of list to be saved.
        disturb (bool): A boolean indicating if we use random disturbance.
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

def save_to_csv(random_disturb: bool, content: list, filename):
    """
    Saves a list of 4-lists to a CSV file with a double space separator.

    Parameters:
        random_disturb (bool): A flag indicating whether we use random disturbance.
        content (list): The list of 4-lists to be saved.
        filename (str): The name of the output CSV file.
    """

    # Construct the full file path
    file_path = generate_output(random_disturb, filename+".csv")

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        for content_list in content:
            writer.writerow(content_list)