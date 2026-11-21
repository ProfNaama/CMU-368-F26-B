#python_docker_run analysis/extract_csv.py <postgres full connection string> <cmu-368s25-a>



import os
import json
import base64
# if not working, install pandas using "pip install pandas psycopg2"
import pandas as pd
import psycopg2

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

file_path = os.path.abspath(__file__)
dir_name = os.path.dirname(file_path) 

inputfilename=os.path.join(dir_name, 'select_results.json')
outputfilename=os.path.join(dir_name, 'decoded.json')
csv_result_output=os.path.join(dir_name, 'extracted_data.csv')

def read_from_posgres(connectionString, expid):
    # connectionString is of the form:
    # DATABASE_URL=postgres://<user_name>:<password>@>host?:5432/<db name>
    host = connectionString.split('@')[1].split(':')[0]
    database = connectionString.split('@')[1].split('/')[1]
    user = connectionString.split('@')[0].split('//')[1].split(':')[0]
    password = connectionString.split('@')[0].split('//')[1].split(':')[1]
    resultTableName = "tri_ca_results"
    
    # Connection parameters
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )

    # Create a cursor object
    cur = conn.cursor()

    # Execute a query
    query = f"SELECT result FROM {resultTableName} WHERE expid = '{expid}'"
    print(f"Executing query: {query}")
    cur.execute(query)

    # Fetch all results
    results = cur.fetchall()
    # Close the cursor and connection
    cur.close()
    conn.close()
    
    #return list(map(lambda r:r[1], results))
    return list(map(lambda r:r[0], results))
    #return list(results)

def write_raw_select_results(connectionString, expid, outfilename):
    raw_results = read_from_posgres(connectionString, expid)
    with open(outfilename, 'w') as file:
        json.dump(raw_results, file, indent=4)

def load_json_to_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def decode_base64_to_dict(data_encoded):
    decoded_data = []
    for data_dict in data_encoded:
        if 'data' in data_dict:
            base64_decoded = base64.b64decode(data_dict['data']).decode('utf-8')
            decoded_dict = json.loads(base64_decoded)
            decoded_dict["time"] = data_dict["time"]
            decoded_data.append(decoded_dict)
        else:
            print('error decoding element')
    return decoded_data

def write_dict_to_file(data_dict, output_file_path):
    with open(output_file_path, 'w') as file:
        json.dump(data_dict, file, indent=4)

general_att_extract = {
    "time" : lambda x: x["time"],
    "treatmentGroupId" : lambda x: x["treatmentGroupId"],
    #"PROLIFIC_PID": lambda x: x["prolificUid"]["PROLIFIC_PID"],
    #"STUDY_ID": lambda x: x["prolificUid"]["STUDY_ID"],
    #"SESSION_ID": lambda x: x["prolificUid"]["SESSION_ID"],
    "code": lambda x: x["code"],

}

def extract_user_post_count(data_dict):
    return len(list(filter(lambda c: c["role"] == "user", data_dict["conversationContext"])))

def extract_user_conversation_duration(data_dict):
    user_conversation = filter(lambda c: c["role"] == "user", data_dict["conversationContext"])
    user_conversation_times = list(map(lambda c: c["interactionTime"], user_conversation))
    return int(sum(user_conversation_times) / 1000)

def extract_user_words_count(data_dict):
    user_conversation = filter(lambda c: c["role"] == "user", data_dict["conversationContext"])
    user_conversation_contents = list(map(lambda c: c["content"], user_conversation))
    words = list(map(lambda c: len(c.split()), user_conversation_contents))
    return sum(words)

def extract_user_chars_count(data_dict):
    user_conversation = filter(lambda c: c["role"] == "user", data_dict["conversationContext"])
    user_conversation_contents = list(map(lambda c: c["content"], user_conversation))
    return sum(list(map(lambda c: len(c), user_conversation_contents)))

def extract_user_sentiment_json(data_dict, sentiment_type):
    user_conversation = filter(lambda c: c["role"] == "user", data_dict["conversationContext"])
    user_conversation_contents = list(map(lambda c: c["content"], user_conversation))
    total_len = sum(map(lambda c: len(c), user_conversation_contents))
    total_len = total_len if total_len > 0 else 1.0  # Avoid division by zero
    sentiment_dic = {sentiment_type: 0.0}
    if user_conversation_contents is not None and len(user_conversation_contents) > 0:
        for c in user_conversation_contents:
            sentiment = sia.polarity_scores(c)
            for sentiment_key in sentiment:
                sentiment_dic.setdefault(sentiment_key, 0.0)
                sentiment_dic[sentiment_key] += sentiment[sentiment_key] * (len(c) / total_len)
    return sentiment_dic[sentiment_type]


conversation_attributes_extract = {
    "user_post_count" : extract_user_post_count,
    "user_conversation_duration" : extract_user_conversation_duration,
    "user_words_count" : extract_user_words_count,
    "user_chars_count" : extract_user_chars_count,
    "user_compound_sentiment_avg" : lambda d: extract_user_sentiment_json(d, "compound"),
    "user_neutral_sentiment_avg" : lambda d: extract_user_sentiment_json(d, "neu"),
    "user_positive_sentiment_avg" : lambda d: extract_user_sentiment_json(d, "pos"),
    "user_negative_sentiment_avg" : lambda d: extract_user_sentiment_json(d, "neg"),
}

def to_csv(data, csv_output_file_path):
    general_headers = []
    answers_headers = []
    csv_data = []
    for data_dict in data:
        if isinstance(data_dict, dict):
            row = {}
            try:
                for k, extractor in general_att_extract.items():
                    row[k] = extractor(data_dict)
                    if k not in general_headers:
                        general_headers.append(k)
                for k, extractor in conversation_attributes_extract.items():
                    row[k] = extractor(data_dict)
                    if k not in general_headers:
                        general_headers.append(k)
                for questionsAnswersSection in ["preQuestionsAnswers", "postQuestionsAnswers"]:
                    if questionsAnswersSection in data_dict and data_dict[questionsAnswersSection] is not None:
                        for k, v in data_dict[questionsAnswersSection].items():
                            if isinstance(v, list):
                                v = " ".join(v)
                            row[k] = v
                            if k not in answers_headers:
                                answers_headers.append(k)
                        row.update(data_dict[questionsAnswersSection])
                csv_data.append(row)
            except Exception as e:
                print(f"An error occurred: {e}, data_dict: {data_dict}")

    headers = general_headers + answers_headers
    df = pd.DataFrame(csv_data)
    df = df[headers]
    df.to_csv(csv_output_file_path, index=False, header=True)



def write_all_emails_and_courses(data, output_file_path):
    # Simple list of all entries: one row per entry, no dedupe,
    # empty email/course where the experiment did not collect them.
    rows = []
    for entry in data:
        answers = entry.get("postQuestionsAnswers") or {}
        user_email = answers.get("q_email")
        user_course = answers.get("q_course")
        email = user_email.lower().strip() if isinstance(user_email, str) else ""
        course = user_course.strip() if isinstance(user_course, str) else (str(user_course) if user_course is not None else "")
        rows.append({"email": email, "course": course})
    with open(output_file_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["email", "course"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(connection_str, expid):
    try:
        # read raw results from postgres and write to a file
        write_raw_select_results(connection_str, expid, inputfilename)
        print(f"Read Raw Postgres data successfully and written to {inputfilename}")
        
        # Load the JSON data from the input file
        original_data = load_json_to_dict(inputfilename)
        
        # Decode the 'data' key and convert to a JSON dictionary
        decoded_data = decode_base64_to_dict(original_data)
        
        # Write the resulting dictionary to an output file
        write_dict_to_file(decoded_data, outputfilename)
        print(f"Decoded data successfully written to {outputfilename}")
        
        write_all_emails_and_courses(decoded_data, outputfilename.replace(".json", ".emails.csv"))
        
        filter_date_string = "2026-08-17T"
        print(f"Flitering only data where sessionStartTime >= {filter_date_string}")
        decoded_data = list(filter(lambda x: "sessionStartTime" in x and x["sessionStartTime"] >= filter_date_string, decoded_data))

        # No pre/post requirement: an experiment may have only a
        # pre-questionnaire, only a post-questionnaire, or neither.
        outputfilename1 = outputfilename.replace(".json", ".filtered.json")
        write_dict_to_file(decoded_data, outputfilename1)

        print(f"Flitering only those conversations with a valid hidden promp")
        decoded_data = list(filter(lambda x: "systemRoleHiddenContent" in x and len(x["systemRoleHiddenContent"]) > 0, decoded_data))
        outputfilename2 = outputfilename1.replace(".json", ".valid_hidden_prompt.json")
        write_dict_to_file(decoded_data, outputfilename2)
        
        to_csv(decoded_data, csv_result_output)
        print(f"Extracted csv successfully written to {csv_result_output}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("connection_string", type=str, help="pg full connection string")
    parser.add_argument("expid", type=str, help="expid")
    args = parser.parse_args()
    main(args.connection_string, args.expid)
    