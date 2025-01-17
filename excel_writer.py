import os
import pandas as pd
from datetime import datetime

def write_results_to_excel(results, metrics, config):    
    results_array = get_results_array(results, metrics, config)
    results_df = get_df(results, metrics, results_array)
    excel_path = os.environ["RESULTS_EXCEL_PATH"]
    save_to_excel(results_df, excel_path)

def extract_questions_and_answers(results):
    samples = results.dataset.samples
    question_list = [sample.user_input for sample in samples]
    answer_list = [sample.response for sample in samples]
    return question_list, answer_list

def get_results_array(results, metrics, config):
    question_list, answer_list = extract_questions_and_answers(results)

    version_number = datetime.now().strftime("%Y-%m-%dT%H:%M")
    new_data = {
        "Version Number": [version_number],
        "Question Source": [os.environ.get('DATASET_FILENAME')],
        "Doc Format": [os.path.splitext(os.environ.get('DATASET_FILENAME'))[1].replace(".", "")],
        "Number of Questions": [len(question_list)],
        "Embeddings Model": [config["embeddings"]["model"]],
        "Model to be Evaluated": [config["llm_to_be_evaluated"]["model"]],
        "Model used for Ragas Metrics": [config["ragas_helper_llm"]["model"]],
        "Question Number": list(range(1, len(question_list) + 1)),
        "Questions": question_list,
        "Answers": answer_list,
    }
    
    for metric in metrics:
        new_data[metric.name] = results[metric.name]
        
    normalize_data_length(new_data)
    
    return new_data

def normalize_data_length(new_data):
    max_length = max(len(lst) for lst in new_data.values())
    for key in new_data:
        current_length = len(new_data[key])
        if current_length < max_length:
            new_data[key].extend([None] * (max_length - current_length))

def get_df(results, metrics, new_data):
    df_new = pd.DataFrame(new_data)
    averages = {metric.name: df_new[metric.name].mean() for metric in metrics}
    average_row = {col: None for col in df_new.columns}
    average_row.update(averages)
    average_row["Version Number"] = "Average"
    df_average = pd.DataFrame([average_row])
    df_new = pd.concat([df_new, df_average], ignore_index=True)
    empty_row = {col: '__________' for col in df_new.columns}
    df_empty = pd.DataFrame([empty_row])
    df_new = pd.concat([df_new, df_empty], ignore_index=True)
    return df_new

def save_to_excel(df_new, excel_path):
    if not os.path.exists(excel_path):
        df_new.to_excel(excel_path, index=False)
    else:
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            startrow = writer.sheets['Sheet1'].max_row if 'Sheet1' in writer.sheets else 0
            df_new.to_excel(writer, index=False, header=writer.sheets['Sheet1'].max_row == 0, startrow=startrow)