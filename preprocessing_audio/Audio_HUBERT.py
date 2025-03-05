from transformers import  Wav2Vec2FeatureExtractor, AutoConfig
from models.Hubert.Model import HubertForSpeechClassification # Carrega manualmente a classe do modelo
import torch
import torch.nn.functional as functional
import numpy as np
from pydub import AudioSegment


def predict_emotion_hubert(audio_file):
    """Modelo original https://github.com/m3hrdadfi/soxan """

    model = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion") # Downloading: 362M
    print('Carregando modelo')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    sampling_rate=16000 # defined by the model; must convert mp3 to this rate.
    config = AutoConfig.from_pretrained("Rajaram1996/Hubert_emotion")

    def speech_file_to_array(path, sampling_rate):
        # using torchaudio...
        # speech_array, _sampling_rate = torchaudio.load(path)
        # resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
        # speech = resampler(speech_array).squeeze().numpy()
        sound = AudioSegment.from_file(path)
        sound = sound.set_frame_rate(sampling_rate)
        sound_array = np.array(sound.get_array_of_samples())
        return sound_array

    sound_array = speech_file_to_array(audio_file, sampling_rate)
    inputs = feature_extractor(sound_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to("cpu").float() for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = functional.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{
        "emo": config.id2label[i],
        "score": round(score * 100, 1)}
        for i, score in enumerate(scores)
    ]
    return [row for row in sorted(outputs, key=lambda x:x["score"], reverse=True) if row['score'] != '0.0%'][:2]


# Teste com a base AVEC_2016

audio_path = '../../Datasets/AVEC_2016/recordings_audio/dev_1.wav'
result = predict_emotion_hubert(audio_path)
print(result)