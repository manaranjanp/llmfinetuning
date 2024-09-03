# The code converts a cricket ball commentary to structured json message 
# using OpenAI and stores in jsonl file for model fintuning.
# Developer: Manaranjan Pradhan
# www.manaranjanp.com

# Import necessary libraries
import sys  # For accessing command-line arguments and system-specific parameters
import json  # For JSON data manipulation
import pandas as pd  # For efficient data manipulation and analysis
from packaging import version  # For version string parsing and comparison
from pydantic import BaseModel, Field  # For data validation and settings management
from openai import OpenAI  # For interacting with OpenAI's API
from typing import Optional  # For type hinting with optional fields
import logging  # For logging messages (errors, warnings, info)
from getpass import getpass  # For securely inputting passwords (not used in this script)

# Configure logging
# This setup ensures that informational messages are displayed during script execution,
# which is crucial for monitoring the progress and debugging if necessary.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

system_message = "You are a cricket analyst who extracts information from commentary text based on the provided schema."

# Define the desired output structure using Pydantic
class CommenatryInfo(BaseModel):
    """
    Pydantic model to define the structure of the extracted cricket commentary information.
    This model serves several purposes:
    1. It ensures type checking for the extracted data.
    2. It provides a clear schema for the output, making it easier to understand the data structure.
    3. It allows for easy serialization and deserialization of the data.
    
    Each field represents a specific piece of information extracted from the cricket commentary:
    """
    bowler: str  # Name of the bowler
    batsman: str  # Name of the batsman facing the delivery
    runs_scored: int  # Number of runs scored on this delivery
    is_out: bool  # Whether the batsman was dismissed on this delivery
    wicket_type: str  # If the batsman was dismissed, the type of dismissal (e.g., "caught", "bowled")
    batsman_shot_type: str  # The type of shot played by the batsman
    bowling_type: str  # The type of delivery bowled (e.g., "fast", "spin")
    fielding_position: str  # The position of the fielder involved in the play
    fielded_by: str  # Name of the fielder who fielded the ball (if applicable)
    caught_by: str  # Name of the fielder who caught the ball (if applicable)

# Initialize the OpenAI client
# This client will be used to make API calls to OpenAI's language models
client = OpenAI()

def get_request_prompt(commentary: str) -> list:
    """
    Generate the prompt for the OpenAI API request.
    
    This function creates a structured prompt that instructs the AI model
    on how to interpret and extract information from the given cricket commentary.
    
    Args:
    commentary (str): The cricket commentary text to be analyzed.
    
    Returns:
    list: A list of message dictionaries forming the conversation prompt.
           This format is specific to OpenAI's chat-based models.

    e.g.
    {
        "messages": [
            {
            "role": "system",
            "content": "You are a cricket analyst who extracts information from commentary text"
            },
            {
            "role": "user",
            "content": "Siraj to Manoj Tiwary, FOUR, walks across towards off and effortlessly picks it away over backward square leg. Brilliantly timed yet again. Fine leg was up inside the ring and Tiwary picked his spot"
            }
        ]
    }
           
    """
    return [
        {"role": "system", "content": system_message}, 
        {"role": "user", "content": commentary}
    ]

def format_request_response(commentary: str, commentary_resp: dict) -> dict:
    """
    Format the request and response for storage or further processing.
    
    This function takes the original commentary and the extracted information,
    and formats them into a structure suitable for model fine-tuning or analysis.
    
    Args:
    commentary (str): The original cricket commentary text.
    commentary_resp (dict): The extracted information from the commentary.
    
    Returns:
    dict: A formatted dictionary containing the conversation and response.
          This format mimics a conversation, which is useful for training language models.
    e.g.

    {
        "messages": [
            {
            "role": "system",
            "content": "You are a cricket analyst who extracts information from commentary text"
            },
            {
            "role": "user",
            "content": "Aaron to Parthiv Patel, FOUR, lovely shot, Aaron - from round the wicket, slides this one onto Parthiv's hips, he gets inside the line, uses the angle and clips it behind square on the on-side for a boundary"
            },
            {
            "role": "assistant",
            "content": "{\"bowler\": \"Aaron\", \"batsman\": \"Parthiv Patel\", \"runs_scored\": 4, \"is_out\": false, \"wicket_type\": \"\", \"batsman_shot_type\": \"clip\", \"bowling_type\": \"round the wicket\", \"fielding_position\": \"behind square on the on-side\", \"fielded_by\": \"\", \"caught_by\": \"\"}"
            }
        ]
    }

    """
    # Use json.dumps with custom encoder for datetime objects
    # This ensures that any datetime objects are properly serialized
    json_str = json.dumps(commentary_resp, ensure_ascii=False, default=str)
         
    return {
        "messages": [
            {"role": "system", "content": "You are a cricket analyst who extracts information from commentary text"},
            {"role": "user", "content": commentary}, 
            {"role": "assistant", "content": json_str}
        ]
    }

def extractInformation(commentary: str) -> CommenatryInfo:
    """
    Extract structured information from the cricket commentary using OpenAI's API.
    
    This function is the core of the information extraction process. It sends the
    commentary to OpenAI's model and receives a structured response conforming to
    the CommenatryInfo schema.
    
    Args:
    commentary (str): The cricket commentary text to be analyzed.
    
    Returns:
    CommenatryInfo: A Pydantic model instance containing the extracted information.
                    This ensures that the returned data adheres to the defined schema.
    """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",  # Specify the OpenAI model to use
        messages=get_request_prompt(commentary),  # Provide the formatted prompt
        response_format=CommenatryInfo  # Specify the expected response format
    )

    cric_commentary = completion.choices[0].message.parsed
    return cric_commentary

def get_commentary_info(messages: list) -> list:
    """
    Process a list of commentary messages and extract information from each.
    
    This function applies the extractInformation function to multiple commentaries,
    allowing for batch processing of commentary data.
    
    Args:
    messages (list): A list of cricket commentary texts.
    
    Returns:
    list: A list of tuples, each containing the original commentary and its extracted information.
          This paired format allows for easy comparison between input and output.
    """
    return [(msg, extractInformation(msg)) for msg in messages] 
    
# Note: This script requires three command-line arguments:
# 1. Path to the input CSV file containing cricket commentaries
# 2. Number of commentaries to sample and process
# 3. Path for the output JSONL file
#
# Example usage:
# python syntheticdata.py input_commentaries.csv 100 output_data.jsonl
#
# This will process 100 random commentaries from input_commentaries.csv,
# save the structured data to output_data.jsonl, and create a CSV file
# named commenatry_data.csv with all processed information.

# Main execution
if __name__ == "__main__":
    # The script expects command-line arguments for input file, sample size, and output file
    """
    Main function to handle command-line arguments and execute the fine-tuning process.
    """
    if len(sys.argv) < 4:
        print("Usage: python generate.py <input csv file> <number of samples> <outout jsonl filename> <output csv filename")
        print("Example: python generate.py cricket.csv 100 commentary.jsonl commentary.csv ")
        sys.exit(1)
    
    # Read the input CSV file containing cricket commentaries
    commentary_df = pd.read_csv(sys.argv[1])
    
    # Sample a subset of the commentary data
    # This allows for processing a smaller dataset, which is useful for testing or when resources are limited
    sample_commenatry_df = commentary_df.sample(int(sys.argv[2]))
    
    # Extract information from the sampled commentaries
    # This step processes each sampled commentary through the OpenAI model
    all_commentaries = get_commentary_info(sample_commenatry_df.Commentary)

    # Write the formatted request-response pairs to a JSONL file
    # This format is suitable for fine-tuning language models or for record keeping
    with open(sys.argv[3], 'w') as outfile:
        for commentary, commentary_info in all_commentaries:
            json.dump(format_request_response(commentary, commentary_info.model_dump()), outfile)
            outfile.write('\n')  # Each JSON object on a new line (JSONL format)

    # Prepare data for CSV output
    # This step converts the extracted information into a pandas DataFrame for easy manipulation
    all_commentaries_json = [commentary_info.model_dump() for commentary, commentary_info in all_commentaries]
    df = pd.DataFrame(all_commentaries_json)

    # Add original commentary and full JSON response to the DataFrame
    # This allows for a complete dataset that includes both input and output
    df['commentary'] = list(sample_commenatry_df.Commentary)
    df['commentary_json'] = [commentary_json.model_dump() for comments, commentary_json in all_commentaries]

    # Save the complete dataset to a CSV file
    # This provides a human-readable and easily analyzable format of the processed data
    df.to_csv(sys.argv[4], index=False)
