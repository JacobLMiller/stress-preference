import pandas as pd
import re
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import ast

def process_normal(f_in, f_out, exclude_failed_training=True):

    df = pd.read_csv(f_in)

    columns_to_keep = ["__js_HISTORY", "__js_ANSWERS", "__js_CONFIDENCES", "__js_NUM_INCORRECT", "__js_TIME", "Q15", "Q20", "Q14", "Q13", "Q12", "Q8", "Finished", "Q3", "StartDate", "EndDate"]

    rename_map = {"__js_HISTORY": "drawings",
                    "__js_ANSWERS": "answers",
                    "__js_CONFIDENCES": "confidence",
                    "__js_NUM_INCORRECT": "num_incorrect",
                    "__js_TIME": "time",
                    "Q15": "strategy",
                    "Q20": "overall_confidence",
                    "Q14": "difficulty",
                    "Q13": "familiarity", 
                    "Q12": "age", 
                    "Q8": "gender",
                    "Q3": "prolific_id"
                }

    new_df = df[columns_to_keep].copy()
    new_df.rename(columns=rename_map, inplace=True)
    new_df.drop([0,1])
    new_df = new_df.iloc[2:].reset_index(drop=True)

    # Remove entries that failed training
    # print(len(new_df))
    if exclude_failed_training:
        new_df['num_incorrect'] = new_df['num_incorrect'].astype(int)
        new_df = new_df[new_df['num_incorrect'] < 5]
        new_df = new_df.reset_index(drop=True)
    # print(len(new_df))

    # Remove duplicate entries from same participant
    new_df = new_df.drop_duplicates(subset='prolific_id', keep='first')

    # Remove erroneous data if applicable
    if '5cb749fda6fe0700189768a5' in new_df['prolific_id'].values:
        new_df = new_df[new_df['prolific_id'] != '5cb749fda6fe0700189768a5']

    new_df = new_df.reset_index(drop=True)

    def safe_convert_int(lst):
        if isinstance(lst, list):
            return list(map(int, lst))
        return []

    def safe_convert_float(lst):
        if isinstance(lst, list):
            return list(map(float, lst))
        return []
    
    
    new_df['answers'] = new_df['answers'].str.rstrip(';').str.split(';').apply(safe_convert_int)
    new_df['confidence'] = new_df['confidence'].str.rstrip(';').str.split(';').apply(safe_convert_int)
    new_df['time'] = new_df['time'].str.rstrip(';').str.split(';').apply(safe_convert_float)
    # new_df['time'] = new_df['time'].str.lstrip(',').str.split(',').apply(safe_convert_float)

    def get_delta(lst):
        # Regular expression to extract numbers after 'l'
        pattern = re.compile(r'l(\d+)')
        
        # Extract numbers from both elements of the list
        match1 = pattern.search(lst[0])
        match2 = pattern.search(lst[1])
        if match1 is None or match2 is None:
            return None
        
        num1 = int(match1.group(1))
        num2 = int(match2.group(1))
        
        # Calculate and return the absolute difference
        return num1, num2, abs(num1 - num2)
    

    def process_drawings(drawings):
        if pd.isna(drawings):  # Check for NaN values
            return []
        
        pairs = drawings.rstrip(';').split(';')
        tuple_list = []
        for pair in pairs:
            pair_list = pair.split(",")
            # delta = get_delta(pair_list)[2]
            drawing_1, drawing_2, delta = get_delta(pair_list)
            pair_list.append(delta)
            tuple_list.append(pair_list)

        return tuple_list
    
    # Setup correct answers
    def process_correct_answers(drawings):
        if pd.isna(drawings):  # Check for NaN values
            return []
        
        pairs = drawings.rstrip(';').split(';')
        correct_answers = []
        for pair in pairs:
            pair_list = pair.split(",")
            drawing_1, drawing_2, delta = get_delta(pair_list)
            # print(drawing_1, drawing_2)
            if drawing_1 > drawing_2: # Left has lower stress
                correct_answers.append(1)
            elif drawing_2 > drawing_1: # Right has lower stress
                correct_answers.append(3)
            elif drawing_1 == drawing_2: # Same stress
                correct_answers.append(2)
            else:
                print("error getting correct")

        return correct_answers

    new_df['drawings_deltas'] = new_df['drawings'].apply(process_drawings)
    new_df['correct_answers'] = new_df['drawings'].apply(process_correct_answers)
    new_df['deltas'] = new_df['drawings_deltas'].apply(lambda x: [i[-1] if i else None for i in x])
    new_df['drawings'] = new_df['drawings_deltas'].apply(lambda x: [i[:-1] if i else i for i in x])

    new_df = new_df.drop(columns=['drawings_deltas'])

    participants = {}

    for index, row in new_df.iterrows():
        participants[row['prolific_id']] = []
        for answer, correct_answer in zip(row["answers"], row["correct_answers"]):
            if answer == correct_answer:
                participants[row['prolific_id']].append(1)
            else:
                participants[row['prolific_id']].append(0)


    new_df['accuracy'] = new_df['prolific_id'].map(participants)

    new_df.to_csv(f_out)
    return new_df

def form_of_the_data(df, output_file):
    # print(df)
    # for col in df.columns:
    #     print(col, df[col].iloc[0])
    def absolute_difference(pair):
        l_value_1 = int(pair[0].split('_')[1][1:])
        l_value_2 = int(pair[1].split('_')[1][1:])
        return abs(l_value_1 - l_value_2)
    

    participants = {}
    for index, row in df.iterrows():
        participants[index] = []
        for answer in row["answers"]:
            if answer == 1:
                participants[index].append("left")
            elif answer == 2:
                participants[index].append("same")
            elif answer == 3:
                participants[index].append("right")
            else:
                print("error: unkown answer")

    # df['answers_text'] = df['prolific_id'].map(participants)
    df['answers_text'] = df.index.map(participants)

    participants = {}
    for index, row in df.iterrows():
        participants[index] = []
        for answer in row["confidence"]:
            if answer == 1:
                participants[index].append("confident")
            elif answer == 2:
                participants[index].append("not confident")
            else:
                print("error: unkown answer")

    # df['confidence_text'] = df['prolific_id'].map(participants)
    df['confidence_text'] = df.index.map(participants)

    with open(output_file, "w") as f_out:
        # header_1 = "graph,g1,g1,g1,g1,g1,g1,g1,g1,g1,g2,g2,g2,g2,g2,g2,g2,g2,g2,g3,g3,g3,g3,g3,g3,g3,g3,g3,g4,g4,g4,g4,g4,g4,g4,g4,g4,g5,g5,g5,g5,g5,g5,g5,g5,g5\n"
        header_1 = ("graph, " + "g1," * 9 + "g2," * 9 + "g3," * 9 + "g4," * 9 + "g5," * 9)[0:-1] + "\n"
        # header_2 = "delta,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4\n\n"
        header_2 = ("delta," + (",".join([str(x/100) for x in range(0, 45, 5)]) + ",") * 5)[0:-1] + "\n\n"
        f_out.writelines(header_1)
        f_out.writelines(header_2)

        for i, row in df.iterrows():
            # if i > 0:
            #     continue

            indexed_list = list(enumerate(row['drawings']))

            # Sort the list based on the 'g' number and then by the absolute difference of the integers next to 'l'
            sorted_indexed_list = sorted(indexed_list, key=lambda x: (int(x[1][0].split('_')[0][1]), absolute_difference(x[1])))

            # Extract the sorted list and the original indices
            sorted_list = [pair for index, pair in sorted_indexed_list]
            original_indices = [index for index, pair in sorted_indexed_list]

            new_list = row['drawings']
            sorted_new_list = [new_list[i] for i in original_indices]
            formatted_pairs = [[f"0.{pair[0][4:6]}{pair[0][-1]}", f"0.{pair[1][4:6]}{pair[1][-1]}"] for pair in sorted_new_list]
            line = f"p{i+1} stimuli: left/right,{','.join(['/'.join(pair) for pair in formatted_pairs])}\n"
            f_out.writelines(line)

            new_list = row['answers']
            # new_list = row['answers_text']
            sorted_new_list = [str(new_list[i]) for i in original_indices]
            line = f"p{i+1} response ('less stress' chosen),{','.join(sorted_new_list)}\n"
            f_out.writelines(line)

            new_list = row['accuracy']
            sorted_new_list = [str(new_list[i]) for i in original_indices]
            line = f"p{i+1} accuracy,{','.join(sorted_new_list)}\n"
            f_out.writelines(line)

            new_list = row['time']
            sorted_new_list = [str(new_list[i]) for i in original_indices]
            line = f"p{i+1} time,{','.join(sorted_new_list)}\n"
            f_out.writelines(line)

            new_list = row['confidence']
            # new_list = row['confidence_text']
            sorted_new_list = [str(new_list[i]) for i in original_indices]
            line = f"p{i+1} confidence,{','.join(sorted_new_list)}\n\n"
            f_out.writelines(line)
            
def form_deomgraphic(df, output_file):

    # with open(output_file, "w") as f_out:
    #     header = "participant;strategy;overall_confidence;difficulty;familiarity;age;gender\n"
    #     f_out.writelines(header)
    #     for index, row in df.iterrows():
    #         line = f"p{index+1};{row['strategy']};{row['overall_confidence']};{row['difficulty']};{row['familiarity']};{row['age']};{row['gender']}\n"
    #         f_out.write(line)

    # columns_to_keep = ["participant","strategy","overall_confidence","difficulty","familiarity","age","gender"]
    # df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])
    # df['participant'] = ["p" + str(i + 1) for i in df.index]

    columns_to_keep = ["participant","name","strategy","overall_confidence","difficulty","familiarity","age","gender"]
    df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])
    df['participant'] = ["p" + str(i + 1) for i in df.index]


    df.to_csv(output_file, columns=columns_to_keep, index=False)
    
def process_expert(f_in, f_out):

    df = pd.read_csv(f_in)

    columns_to_keep = ["__js_HISTORY10", "__js_ANSWERS10", "__js_CONFIDENCES10",  "__js_TIME10",
                       "__js_HISTORY25", "__js_ANSWERS25", "__js_CONFIDENCES25",  "__js_TIME25",
                       "__js_HISTORY50", "__js_ANSWERS50", "__js_CONFIDENCES50",  "__js_TIME50",
                        "__js_NUM_INCORRECT", "Q15", "Q20", "Q14", "Q13", "Q12", "Q8", "Finished", "expert_name"]

    rename_map = {"__js_HISTORY10": "drawings10",
                    "__js_ANSWERS10": "answers10",
                    "__js_CONFIDENCES10": "confidence10",
                    "__js_TIME10": "time10",
                    "__js_HISTORY25": "drawings25",
                    "__js_ANSWERS25": "answers25",
                    "__js_CONFIDENCES25": "confidence25",
                    "__js_TIME25": "time25",
                    "__js_HISTORY50": "drawings50",
                    "__js_ANSWERS50": "answers50",
                    "__js_CONFIDENCES50": "confidence50",
                    "__js_TIME50": "time50",
                    "__js_NUM_INCORRECT": "num_incorrect",
                    "Q15": "strategy",
                    "Q20": "overall_confidence",
                    "Q14": "difficulty",
                    "Q13": "familiarity", 
                    "Q12": "age", 
                    "Q8": "gender",
                    "expert_name": "name"
                }
    

    new_df = df[columns_to_keep].copy()
    new_df.rename(columns=rename_map, inplace=True)
    new_df.drop([0,1])
    new_df = new_df.iloc[2:].reset_index(drop=True)

    print(len(new_df))
    quit()

    # print(len(new_df))
    new_df = new_df[new_df['Finished'] == 'True']
    # print(len(new_df))
    # quit()
    # Remove entries that failed training
    # print(len(new_df))
    new_df['num_incorrect'] = new_df['num_incorrect'].astype(int)

    # new_df = new_df[new_df['num_incorrect'] < 5]
    # print(len(new_df))

    # Remove duplicate entries from same participant
    # new_df = new_df.drop_duplicates(subset='prolific_id', keep='first')
    # print(len(new_df))

    temp_df = new_df.copy()

    columns_to_check = ['drawings10', 'drawings25', 'drawings50', 'answers10', 'answers25', 'answers50',
                    'confidence10', 'confidence25', 'confidence50', 'time10', 'time25', 'time50']
    
    def convert_to_list(s):
        if isinstance(s, str):
            return s.split(';')[:-1]
        return []
    # Apply the conversion to each column
    for col in columns_to_check:
        temp_df[col] = temp_df[col].apply(convert_to_list)

    indices_to_drop = []

    for index, row in temp_df.iterrows():
        for col in columns_to_check:
            # print(len(row[col]))
            if len(row[col]) != 45:
                indices_to_drop.append(index)
    
    
    new_df = new_df.drop(indices_to_drop)

    # Reset index if needed
    new_df.reset_index(drop=True, inplace=True)

    # print(len(new_df))
    # print(len(temp_df))


    def safe_convert_int(lst):
        if isinstance(lst, list):
            return list(map(int, lst))
        return []

    def safe_convert_float(lst):
        if isinstance(lst, list):
            return list(map(float, lst))
        return []
    
    
    new_df['answers10'] = new_df['answers10'].str.rstrip(';').str.split(';').apply(safe_convert_int)
    new_df['confidence10'] = new_df['confidence10'].str.rstrip(';').str.split(';').apply(safe_convert_int)
    new_df['time10'] = new_df['time10'].str.rstrip(';').str.split(';').apply(safe_convert_float)

    new_df['answers25'] = new_df['answers25'].str.rstrip(';').str.split(';').apply(safe_convert_int)
    new_df['confidence25'] = new_df['confidence25'].str.rstrip(';').str.split(';').apply(safe_convert_int)
    new_df['time25'] = new_df['time25'].str.rstrip(';').str.split(';').apply(safe_convert_float)

    new_df['answers50'] = new_df['answers50'].str.rstrip(';').str.split(';').apply(safe_convert_int)
    new_df['confidence50'] = new_df['confidence50'].str.rstrip(';').str.split(';').apply(safe_convert_int)
    new_df['time50'] = new_df['time50'].str.rstrip(';').str.split(';').apply(safe_convert_float)


    




    def get_delta(lst):
        # Regular expression to extract numbers after 'l'
        pattern = re.compile(r'l(\d+)')
        
        # Extract numbers from both elements of the list
        match1 = pattern.search(lst[0])
        match2 = pattern.search(lst[1])
        if match1 is None or match2 is None:
            return None
        
        num1 = int(match1.group(1))
        num2 = int(match2.group(1))
        
        # Calculate and return the absolute difference
        return num1, num2, abs(num1 - num2)
    

    def process_drawings(drawings):
        if pd.isna(drawings):  # Check for NaN values
            return []
        
        pairs = drawings.rstrip(';').split(';')
        tuple_list = []
        for pair in pairs:
            pair_list = pair.split(",")
            delta = get_delta(pair_list)
            pair_list.append(delta)
            tuple_list.append(pair_list)

        return tuple_list
    
    def process_correct_answers(drawings):
        if pd.isna(drawings):  # Check for NaN values
            return []
        
        pairs = drawings.rstrip(';').split(';')
        correct_answers = []
        for pair in pairs:
            pair_list = pair.split(",")
            drawing_1, drawing_2, delta = get_delta(pair_list)
            # print(drawing_1, drawing_2)
            if drawing_1 > drawing_2: # Left has lower stress
                correct_answers.append(1)
            elif drawing_2 > drawing_1: # Right has lower stress
                correct_answers.append(3)
            elif drawing_1 == drawing_2: # Same stress
                correct_answers.append(2)
            else:
                print("error getting correct")

        return correct_answers

    new_df['drawings_deltas10'] = new_df['drawings10'].apply(process_drawings)
    new_df['correct_answers10'] = new_df['drawings10'].apply(process_correct_answers)
    new_df['deltas10'] = new_df['drawings_deltas10'].apply(lambda x: [i[-1] if i else None for i in x])
    new_df['drawings10'] = new_df['drawings_deltas10'].apply(lambda x: [i[:-1] if i else i for i in x])

    new_df['drawings_deltas25'] = new_df['drawings25'].apply(process_drawings)
    new_df['correct_answers25'] = new_df['drawings25'].apply(process_correct_answers)
    new_df['deltas25'] = new_df['drawings_deltas25'].apply(lambda x: [i[-1] if i else None for i in x])
    new_df['drawings25'] = new_df['drawings_deltas25'].apply(lambda x: [i[:-1] if i else i for i in x])
    
    new_df['drawings_deltas50'] = new_df['drawings50'].apply(process_drawings)
    new_df['correct_answers50'] = new_df['drawings50'].apply(process_correct_answers)
    new_df['deltas50'] = new_df['drawings_deltas50'].apply(lambda x: [i[-1] if i else None for i in x])
    new_df['drawings50'] = new_df['drawings_deltas50'].apply(lambda x: [i[:-1] if i else i for i in x])

    # comment this if we want to create new cols for each pair of drawings
    # however this will create 45 cols for the delta, answer, and time, so 135 total extra cols
    new_df = new_df.drop(columns=['drawings_deltas10'])
    new_df = new_df.drop(columns=['drawings_deltas25'])
    new_df = new_df.drop(columns=['drawings_deltas50'])

    


    participants = {}
    for index, row in new_df.iterrows():
        participants[index] = []
        for answer, correct_answer in zip(row["answers10"], row["correct_answers10"]):
            if answer == correct_answer:
                participants[index].append(1)
            else:
                participants[index].append(0)

    new_df['accuracy10'] = new_df.index.map(participants)

    participants = {}
    for index, row in new_df.iterrows():
        participants[index] = []
        for answer, correct_answer in zip(row["answers25"], row["correct_answers25"]):
            if answer == correct_answer:
                participants[index].append(1)
            else:
                participants[index].append(0)

    new_df['accuracy25'] = new_df.index.map(participants)

    participants = {}
    for index, row in new_df.iterrows():
        participants[index] = []
        for answer, correct_answer in zip(row["answers50"], row["correct_answers50"]):
            if answer == correct_answer:
                participants[index].append(1)
            else:
                participants[index].append(0)

    new_df['accuracy50'] = new_df.index.map(participants)

    

    for n in [10, 25, 50]:
        columns_to_keep = [f"drawings{n}", f"answers{n}", f"confidence{n}",  f"time{n}", f"accuracy{n}",
                            "num_incorrect", "strategy", "overall_confidence", "difficulty", "familiarity", "age", "gender", "name"]

        rename_map = {f"drawings{n}":"drawings", f"answers{n}":"answers", f"confidence{n}":"confidence",  f"time{n}":"time", f"accuracy{n}":"accuracy"}

        sub_df = new_df[columns_to_keep].copy()
        sub_df.rename(columns=rename_map, inplace=True)

        form_of_the_data(sub_df, f"processed_data\\expert{n}_formatted.csv")
    form_deomgraphic(sub_df, f"processed_data\\expert_demographics.csv")
    # df10.to_csv("tentest.csv")

    new_df.to_csv(f_out)
    return new_df


def count_correct_by_delta(df):

    participants = {}

    for index, row in df.iterrows():
        participants[row['prolific_id']] = []
        for answer, correct_answer in zip(row["answers"], row["correct_answers"]):
            if answer == correct_answer:
                participants[row['prolific_id']].append(1)
            else:
                participants[row['prolific_id']].append(0)


    df['accuracy'] = df['prolific_id'].map(participants)
    for participant, accuracy in participants.items():
        print(f"{sum(accuracy)}/45")

    # for col in df.columns:
    #     print(col, df[col].iloc[1])
    # print(len(df))
    deltas = list(range(0, 45, 5))
    delta_correct = {}
    for delta in deltas:
        delta_correct[delta] = 0

    for index, row in df.iterrows():
        for answer, delta in zip(row["accuracy"], row["deltas"]):
            delta_correct[delta] += answer

    for delta in deltas:
        delta_correct[delta] = delta_correct[delta] / len(df)
    # print(delta_correct)
    # Extract keys and values
    x = list(delta_correct.keys())
    y = list(delta_correct.values())

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='blue')

    # Add labels and title
    plt.ylabel('Number of Correct Responses')
    plt.xlabel('Delta')
    plt.title('Bar Chart of accuracy by delta')

    # Show the plot
    plt.show()


def count_confidence_by_delta(df):

    participants = {}

    for index, row in df.iterrows():
        participants[row['prolific_id']] = []
        for answer, correct_answer in zip(row["answers"], row["correct_answers"]):
            if answer == correct_answer:
                participants[row['prolific_id']].append(1)
            else:
                participants[row['prolific_id']].append(0)


    df['accuracy'] = df['prolific_id'].map(participants)

    deltas = list(range(0, 45, 5))
    delta_confident = {}
    delta_not_confident = {}
    num_for_each_delta = {}
    for delta in deltas:
        delta_confident[delta] = 0
        delta_not_confident[delta] = 0
        num_for_each_delta[delta] = 0



    total = 0
    for index, row in df.iterrows():
        for answer, delta in zip(row["confidence"], row["deltas"]):
            num_for_each_delta[delta] += 1
            if answer == 1:
                total += 1
                delta_confident[delta] += 1
            else:
                total += 1
                delta_not_confident[delta] += 1

    for delta in deltas:
        delta_confident[delta] = delta_confident[delta] / len(df)
        delta_not_confident[delta] = delta_not_confident[delta] / len(df)

    print(len(df))
    print(total)
    print(num_for_each_delta)
    print(delta_confident)
    print(delta_not_confident)
    # labels = deltas
    # confident_values = [delta_confident[delta] for delta in deltas]
    # not_confident_values = [delta_not_confident[delta] for delta in deltas]

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width/2, confident_values, width, label='Confident')
    # rects2 = ax.bar(x + width/2, not_confident_values, width, label='Not Confident')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xlabel('Deltas')
    # ax.set_ylabel('Values')
    # ax.set_title('Values by delta and confidence')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()

    # fig.tight_layout()

    # plt.show()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=deltas,
        y=[delta_confident[delta] for delta in deltas],
        name='Confident',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=deltas,
        y=[delta_not_confident[delta] for delta in deltas],
        name='Not Confident',
        marker_color='red'
    ))

    # Customize layout
    fig.update_layout(
        title='Values by Delta and Confidence',
        xaxis=dict(
            title='Deltas'
        ),
        yaxis=dict(
            title='Values'
        ),
        barmode='group'
    )

    fig.show()

def count_total_correct_with_confidence(df):
    participants_confident = {}
    participants_not_confident = {}

    for index, row in df.iterrows():
        participants_confident[row['prolific_id']] = []
        participants_not_confident[row['prolific_id']] = []
        for answer, correct_answer, confidence in zip(row["answers"], row["correct_answers"], row["confidence"]):
            if confidence == 1:
                if answer == correct_answer:
                    participants_confident[row['prolific_id']].append(1)
                else:
                    participants_confident[row['prolific_id']].append(0)
            else:
                if answer == correct_answer:
                    participants_not_confident[row['prolific_id']].append(1)
                else:
                    participants_not_confident[row['prolific_id']].append(0)


    df['accuracy_confident'] = df['prolific_id'].map(participants_confident)
    df['total_correct_confident'] = df['accuracy_confident'].apply(lambda x: sum(x))

    df['accuracy_not_confident'] = df['prolific_id'].map(participants_not_confident)
    df['total_correct_not_confident'] = df['accuracy_not_confident'].apply(lambda x: sum(x))

    # for participant, accuracy in participants.items():
    #     print(f"{sum(accuracy)}/45")
    print(df['total_correct_confident'].mean())
    print(df['total_correct_not_confident'].mean())

def count_total_correct(df):
    participants = {}

    for index, row in df.iterrows():
        participants[row['prolific_id']] = []
        for answer, correct_answer in zip(row["answers"], row["correct_answers"]):
            if answer == correct_answer:
                participants[row['prolific_id']].append(1)
            else:
                participants[row['prolific_id']].append(0)


    df['accuracy'] = df['prolific_id'].map(participants)
    df['total_correct'] = df['accuracy'].apply(lambda x: sum(x))

    # for participant, accuracy in participants.items():
    #     print(f"{sum(accuracy)}/45")
    # print(df['total_correct'].mean())
    return df['total_correct'].mean()

def get_total_time_trials(df):
    participants = {}
    for index, row in df.iterrows():
        participants[row['prolific_id']] = 0
        for time in row["time"]:
            participants[row['prolific_id']] += time

    for p, t in participants.items():
        participants[p] = t / 60

    print(participants)

def get_total_time_survey(df):
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])

    # Calculate the difference
    df['Duration'] = df['EndDate'] - df['StartDate']

    # Convert the duration to minutes and seconds
    df['Duration_minutes'] = df['Duration'].dt.total_seconds() // 60
    df['Duration_seconds'] = df['Duration'].dt.total_seconds() % 60

    print(df[['prolific_id', 'StartDate', 'EndDate', 'Duration_minutes', 'Duration_seconds']])

def check_complete(df):
    for index, row in df.iterrows():
        for num_errors in row["num_incorrect"]:
            if int(row["num_incorrect"]) >= 5:
                print(row['prolific_id'])
            # print(row['prolific_id'], row['num_incorrect'])


def fix_nt50():

    # df = pd.read_csv("n50-nt-final.csv")

    # selected_columns = df.filter(regex=r'^\d+_Main loop$')
    # selected_columns = df[['Q3']].join(selected_columns)

    # # Create a new DataFrame with these selected columns
    # new_df = selected_columns.copy()

    # df = new_df.iloc[[0,10]]



    # df.to_csv("50nt_test.csv")

    df = pd.read_csv("50nt_test1.csv")

    actual_responses = []
    for index, row in df.iterrows():
        if row['5cb749fda6fe0700189768a5'] == 'The drawing on the left has lower stress':
            actual_responses.append(1)
        elif row['5cb749fda6fe0700189768a5'] == 'The drawings have the same stress':
            actual_responses.append(2)
        elif row['5cb749fda6fe0700189768a5'] == 'The drawing on the right has lower stress':
            actual_responses.append(3)
    
    # print(len(actual_responses))
    close = [3,3,2,3,3,1,1,3,3,2,1,1,3,1,1,3,1,1,1,1,3,2,1,1,2,3,1,2,1,1,1,1,1,3,3,3,2,3,3,3,1,2,1,1]
    print(len(actual_responses), actual_responses)
    print(len(close), close)

    return actual_responses


def count_failed(df):
    df['num_incorrect'] = df['num_incorrect'].astype(int)
    return df[df['num_incorrect'] >= 5].shape[0]


def count_correct(df):
    print(sum(df.iloc[27]['accuracy']))
    

def main():
    f_in = "raw_Data/stress10.csv"

    f_out = f_in[0:-4] + "-cleaned.csv"

    df = process_normal(f_in, f_out, False)




if __name__ == "__main__":
    main()