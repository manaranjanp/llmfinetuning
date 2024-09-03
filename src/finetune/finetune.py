# The code converts a cricket ball commentary to structured json message 
# using OpenAI and stores in jsonl file for model fintuning.
# Developer: Manaranjan Pradhan
# www.manaranjanp.com

"""
Fine-tuning Script for OpenAI Models

This script facilitates the fine-tuning of OpenAI models using a specified training file.
It handles file upload, job creation, and monitoring of the fine-tuning process.

Usage:
    python finetune.py <model_name> <new_finetuned_model_name> <training_filename>

Example:
    python finetune.py gpt-4 cric-commentary-ft commentary.jsonl
"""

import sys
import time
from datetime import datetime
from typing import Dict, Any

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def upload_file(filename: str) -> Dict[str, Any]:
    """
    Create a file for fine-tuning using the OpenAI API.

    Args:
        filename (str): Path to the file to be uploaded.

    Returns:
        Dict[str, Any]: Response from the API containing file information.
    """
    try:
        with open(filename, "rb") as file:
            return client.files.create(file=file, purpose="fine-tune")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating file: {e}")
        sys.exit(1)

def create_finetune_job(training_file: str, model: str) -> Dict[str, Any]:
    """
    Create a fine-tuning job using the OpenAI API.

    Args:
        training_file (str): ID of the training file.
        model (str): Name of the model to fine-tune.

    Returns:
        Dict[str, Any]: Response from the API containing job information.
    """
    try:
        return client.fine_tuning.jobs.create(
            training_file=training_file,
            model=model,
            hyperparameters={"n_epochs": 2, "batch_size": 8}
        )
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        sys.exit(1)

def monitor_finetune_job(job_id: str) -> None:
    """
    Monitor the progress of a fine-tuning job and print updates.

    Args:
        job_id (str): ID of the fine-tuning job to monitor.
    """
    while True:
        time.sleep(30)
        try:
            job_status = client.fine_tuning.jobs.retrieve(job_id)
            print(f"------------ Job Status: {job_status.status} --------------")

            if job_status.status in ["failed", "succeeded", "cancelled"]:
                print("Job Completed. Detailed Events list:")
                events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
                for event in events:
                    print(f'{datetime.fromtimestamp(event.created_at)} {event.message}')
                
                print("######## Fine-tuned model ###########")
                print(f"{job_status.fine_tuned_model}")
                print("#####################################")
                break
        except Exception as e:
            print(f"Error monitoring job: {e}")
            break

def main() -> None:
    """
    Main function to handle command-line arguments and execute the fine-tuning process.
    """
    if len(sys.argv) < 3:
        print("Usage: python finetune.py <model_name> <training_filename>")
        print("Example: python finetune.py gpt-4o-mini-2024-07-18 training.jsonl")
        sys.exit(1)

    model_name, training_filename = sys.argv[1:4]

    print(f"Starting fine-tuning process for {model_name} with {training_filename}")

    training_file = upload_file(training_filename)
    ft_job = create_finetune_job(training_file.id, model_name)

    print(f"Fine-tuning job created: {ft_job.id}")
    monitor_finetune_job(ft_job.id)

if __name__ == "__main__":
    main()
