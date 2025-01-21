import glob
from datasets import Dataset, DatasetDict,Audio
import pandas as pd

def make_dataset(train_path, valid_path, test_path):
    path_wav_train = "D:/in_car_command/in_car_command/train/*.wav"
    path_wav_train_list = glob.glob(path_wav_train)
    path_wav_test = "D:/in_car_command/in_car_command/test/*.wav"
    path_wav_test_list = glob.glob(path_wav_test)
    path_wav_valid = "D:/in_car_command/in_car_command/valid/*.wav"
    path_wav_valid_list = glob.glob(path_wav_valid)

    path_label_train = "D:/in_car_command/in_car_command/train/extracted_transcriptions_train.txt"
    path_label_test = "D:/in_car_command/in_car_command/test/extracted_transcriptions_test.txt"
    path_label_valid = "D:/in_car_command/in_car_command/valid/extracted_transcriptions_valid.txt"

    transcript_list_train = []
    transcript_list_test = []
    transcript_list_valid = []

    for path_label, transcript_list in [(path_label_train, transcript_list_train), (path_label_test, transcript_list_test), (path_label_valid, transcript_list_valid)]:
        with open(path_label, 'r', encoding='utf-8') as f:
                
            for line in f:
                transcription = line.split('\t')[-1].strip()
                transcript_list.append(transcription)


    audio_lst = path_wav_train_list[:32000] + path_wav_test_list[:8000]
    transcript_lst = transcript_list_train[:32000] + transcript_list_test [:8000]


    df = pd.DataFrame(data=transcript_lst, columns = ["transcript"])
    df["raw_data"] = audio_lst

    ds = Dataset.from_dict({"audio": [path for path in df["raw_data"]],
                        "transcripts": [transcript for transcript in df["transcript"]]}).cast_column("audio", Audio(sampling_rate=16000))

    train_testvalid = ds.train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    datasets = DatasetDict({
        "train": train_testvalid["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"]})

    #datasets.push_to_hub("Angeriod/in_car_commands_27")

    return datasets