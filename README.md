# Whisper Fine-Tuning for In-Car Conversations and Commands

## Overview
This project fine-tunes the Whisper model on a dataset of in-car conversations and commands in Korean. The goal is to improve automatic speech recognition (ASR) performance for automotive use cases. The model is fine-tuned using the `eval_cer` (Character Error Rate) metric to assess accuracy.

## Setup and Hardware
- **GPU**: RTX 3060 ti 8GB (approximately 4GB used for training)
- **Model**: Whisper-base (due to VRAM limitations)
- **Dataset**: A pre-processed dataset of in-car conversations and commands (total: 32,000 training samples, 4,000 validation, 4,000 test)
- **Training Time**: 11 hours (10 epochs)
- **Metrics**: Character Error Rate (CER)
- **Training Process**: 
  - Training is done using the pre-cleaned dataset, with evaluation of `eval_cer` after each epoch.
  - The process includes both training and evaluation, with model improvements monitored through the CER metric.

## Training Details
| Epoch | Train Loss | Eval Loss | Eval CER (%) |
|-------|------------|-----------|--------------|
| 1     | 0.1208     | 0.1080    | 11.52        |
| 2     | 0.0519     | 0.0671    | 9.67         |
| 3     | 0.0313     | 0.0533    | 7.89         |
| 4     | 0.0185     | 0.0455    | 8.07         |
| 5     | 0.0103     | 0.0438    | 7.03         |
| 6     | 0.0039     | 0.0438    | 6.60         |
| 7     | 0.0014     | 0.0427    | 6.18         |
| 8     | 0.0004     | 0.0419    | 5.68         |
| 9     | 0.0002     | 0.0419    | 6.47         |
| 10    | 0.0001     | 0.0418    | 6.39         |

## Results
- **Start Test CER**: 30.6%
- **After Fine-Tuning Test CER**: 5.46%

### Example of Correct Prediction:
- **Prediction**: "눈 와서 즐거웠는데 내일도 눈 예보 있어?"
- **Label**: "눈 와서 즐거웠는데 내일도 눈 예보 있어?"

### Example of Incorrect Prediction:
- **Prediction**: "전체 조명 두 시에 켜�� 돼."
- **Label**: "전체 조명 두 시에 켜면 돼."

## Conclusion
This fine-tuning process significantly reduced the Character Error Rate (CER), demonstrating the effectiveness of Whisper for speech recognition in the context of in-car conversations. The results highlight good progress, although there are some occasional errors, such as misinterpreted words or symbols.
