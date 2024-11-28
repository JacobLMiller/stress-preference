import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def count_correct(df):

    # print(len(df['answers'].iloc[0]))
    # print(len(df['confidence'].iloc[0]))
    # print(len(df['time'].iloc[0]))
    # print(len(df['drawings'].iloc[0]))
    # print(len(df['deltas'].iloc[0]))
    # print(len(df['correct_answers'].iloc[0]))
    # for col in df.columns:
    #     print(col, df[col].iloc[0])
    
    participants = {}

    for index, row in df.iterrows():
        participants[row['prolific_id']] = []
        num_answers = 0
        for answer, correct_answer in zip(row["answers"], row["correct_answers"]):
            print(num_answers, answer)
            num_answers += 1
            if answer == correct_answer:
                participants[row['prolific_id']].append(1)
            else:
                participants[row['prolific_id']].append(0)
        print(num_answers)


    quit()

    for participant, accuracy in participants.items():
        print(len(accuracy))
        print(f"{sum(accuracy)}/45")






def main():
    f_in = "pilot25-cleaned.csv"

    df = pd.read_csv(f_in)
    count_correct(df)

    # print(df)


if __name__ == "__main__":
    main()