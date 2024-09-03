# The code converts a cricket ball commentary to structured json message 
# using OpenAI and stores in jsonl file for model fintuning.
# Developer: Manaranjan Pradhan
# www.manaranjanp.com

import os 
import sys
import json
import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI
from getpass import getpass

# Initialize the OpenAI client
# This client will be used to make API calls to OpenAI's services
client = OpenAI()

# Define the data structure for cricket commentary information using Pydantic
# This ensures type checking and provides a clear structure for the data we're working with
class CommenatryInfo(BaseModel):
    bowler: str  # Name of the bowler
    batsman: str  # Name of the batsman
    runs_scored: int  # Number of runs scored on this ball
    is_out: bool  # Whether the batsman is out
    wicket_type: str  # Type of wicket (e.g., "caught", "bowled", etc.), empty if not out
    batsman_shot_type: str  # Type of shot played by the batsman
    bowling_type: str  # Type of bowling (e.g., "spin", "fast")
    fielding_position: str  # Position of the fielder involved in the play
    fielded_by: str  # Name of the fielder who fielded the ball, if applicable
    caught_by: str  # Name of the fielder who caught the ball, if applicable

def extract_content(json_string):
    """
    Extract content from a JSON string containing messages.
    
    This function parses a JSON string that contains message data and extracts
    the content for 'system', 'user', and 'assistant' roles. It handles cases
    where the content might be a string or a dictionary.
    
    :param json_string: A JSON string containing message data
    :return: A dictionary with 'system', 'user', and 'assistant' content
    """
    # Parse the JSON string into a Python dictionary
    data = json.loads(json_string)
    
    # Initialize the result dictionary with empty strings for each role
    result = {
        "system": "",
        "user": "",
        "assistant": ""
    }
    
    # Iterate through each message in the data
    for message in data['messages']:
        role = message['role']
        content = message['content']
        
        # If the role is one we're interested in (system, user, or assistant)
        if role in result:
            # If the content is a dictionary, convert it to a JSON string
            # This ensures consistency in how we handle different content types
            if isinstance(content, dict):
                result[role] = json.dumps(content)
            else:
                result[role] = content
    
    return result

def compare_messages(message1, message2):
    """
    Compare two messages and return a dictionary of boolean values
    indicating whether each field matches. String comparisons are case-insensitive.
    
    This function is crucial for evaluating the performance of the model by
    comparing its output (message1) with the baseline or expected output (message2).
    
    :param message1: First message dictionary (typically the model's output)
    :param message2: Second message dictionary (typically the baseline or expected output)
    :return: Dictionary with boolean values for each field comparison
    """
    result = {}
    # Get all unique keys from both messages
    all_keys = set(message1.keys()) | set(message2.keys())
    
    for key in all_keys:
        if key in message1 and key in message2:
            value1 = message1[key]
            value2 = message2[key]
            
            # Perform case-insensitive comparison for strings
            # This helps in handling minor discrepancies in text fields
            if isinstance(value1, str) and isinstance(value2, str):
                result[key] = value1.lower() == value2.lower()
            else:
                result[key] = value1 == value2
        else:
            # If a key is missing in either message, mark it as a mismatch
            result[key] = False
    
    return result

def aggregate_comparisons(comparison_results):
    """
    Aggregate multiple comparison results and return a DataFrame
    with the number of matches and mismatches for each field.
    
    This function is essential for summarizing the performance of the model
    across multiple test cases. It provides a clear view of which fields
    the model is handling well and which might need improvement.
    
    :param comparison_results: List of comparison result dictionaries
    :return: DataFrame with aggregated results, including accuracy for each field
    """
    aggregated = {}
    
    # Iterate through all comparison results
    for result in comparison_results:
        for key, value in result.items():
            if key not in aggregated:
                aggregated[key] = {'matches': 0, 'mismatches': 0}
            
            # Count matches and mismatches for each field
            if value:
                aggregated[key]['matches'] += 1
            else:
                aggregated[key]['mismatches'] += 1
    
    # Create a DataFrame from the aggregated results
    df = pd.DataFrame.from_dict(aggregated, orient='index')
    df.index.name = 'field'
    df.reset_index(inplace=True)
    # Calculate accuracy for each field
    df['accuracy'] = df['matches'] / (df['matches'] + df['mismatches'])
    return df

def get_request_prompt(system_msg, user_msg):
    """
    Create a prompt for the OpenAI API request.
    
    This function formats the system and user messages into the structure
    expected by the OpenAI API. The system message sets the context or role
    for the AI, while the user message contains the actual query or content
    to be processed.
    
    :param system_msg: System message defining the AI's role or context
    :param user_msg: User message containing the query or content to process
    :return: List of message dictionaries formatted for the OpenAI API
    """
    return [{"role": "system", "content": sys_msg}, 
            {"role": "user", "content": user_msg}]

def extractInformation(model, prompt) -> CommenatryInfo:
    """
    Extract cricket commentary information using the OpenAI API.
    
    This function sends a request to the OpenAI API with the given model and prompt,
    and parses the response into a CommenatryInfo object. It's the core function
    that interacts with the AI model to extract structured information from
    the cricket commentary text.
    
    :param model: OpenAI model identifier to use for the API call
    :param prompt: List of message dictionaries (system and user messages)
    :return: CommenatryInfo object with extracted information
    """
    # Make an API call to OpenAI, specifying the response format as CommenatryInfo
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=prompt,
        response_format=CommenatryInfo)

    # Extract the parsed response from the API result
    cric_commentary = completion.choices[0].message.parsed

    return cric_commentary

def read_jsonl(file_path):
    """
    Read a JSONL file and return a list of JSON objects.
    
    This function reads a file where each line is a valid JSON object (JSONL format).
    It's used to load the test cases or messages that will be processed by the model.
    
    :param file_path: Path to the JSONL file
    :return: List of JSON objects, each representing a message or test case
    """
    with open(file_path, 'r') as file:
        try:
            # Read each line of the file, stripping whitespace
            messages = [line.strip() for line in file]
        except json.JSONDecodeError as e:
            # If there's an error parsing the JSON, print the error message
            print(f"Error message: {str(e)}")

    return messages            

# System messages for the OpenAI API
# These messages set the context for the AI model, instructing it on its role
# and providing an example of the expected output format

sys_msg = """"You are a cricket analyst who extracts information from commentary text."""

# Main execution block
if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) < 2:
        print("Please provide the JSONL file path")
        print("Usage: python evaluate.py <test data in jsonl format> <results file in csv format>")
        sys.exit(1)
     
    # Read messages from the JSONL file
    messages = read_jsonl(sys.argv[1])
    
    # List to store the results of comparing model outputs with baselines
    comparison_results = []

    # Process each message in the input file
    for msg in messages:
        # Extract content from the JSON message
        json_contents = extract_content(msg)
        
        # Create a prompt for the OpenAI API using the system message and user content
        prompt = get_request_prompt(sys_msg, json_contents['user'])
        
        # Extract information using the OpenAI API
        # Note: The model identifier is hardcoded here. Consider making this configurable.
        cinfo = extractInformation("ft:gpt-4o-mini-2024-07-18:personal::A2EkrtrO", prompt)
#        cinfo = extractInformation("gpt-4o-mini-2024-07-18", prompt)
        
        # Compare the extracted information with the baseline (assistant) response
        # The baseline response is assumed to be in the 'assistant' field of json_contents
        comparison_results.append(compare_messages(json.loads(json_contents['assistant']), cinfo.model_dump()))

    # Aggregate the comparison results to get overall performance metrics
    final_results = aggregate_comparisons(comparison_results)

    # Print the final results to the console
    print(final_results)

    # Save the final results to a CSV file for further analysis
    final_results.to_csv(sys.argv[2], index=False)
