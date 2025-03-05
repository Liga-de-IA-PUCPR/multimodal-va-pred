from transformers import Wav2Vec2FeatureExtractor, AutoConfig
from models.Hubert.Model import HubertForSpeechClassification  # Carrega manualmente a classe do modelo
import torch
import torch.nn.functional as functional
import numpy as np
from pydub import AudioSegment


def predict_emotion_hubert(audio_file):
    """
    For each 4-second segment in the audio_file, predicts the emotion using
    the Hubert model and returns a list of dictionaries with a timestamp and predictions.
    """
    # Carrega o modelo e extrator de características
    print('Carregando modelo')
    model = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion")  # Downloading: 362M
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    sampling_rate = 16000  # definido pelo modelo; convertemos o áudio para essa taxa
    config = AutoConfig.from_pretrained("Rajaram1996/Hubert_emotion")

    def speech_file_to_array(path, sampling_rate):
        # Converte o arquivo de áudio para um array NumPy com a taxa de amostragem desejada.
        sound = AudioSegment.from_file(path)
        sound = sound.set_frame_rate(sampling_rate)
        sound_array = np.array(sound.get_array_of_samples())
        return sound_array

    # Converte o áudio para array
    sound_array = speech_file_to_array(audio_file, sampling_rate)

    # Define tamanho do segmento: 4 segundos de áudio corresponde a sampling_rate * 4 samples.
    chunk_size = sampling_rate * 4

    predictions_list = []
    # Processa o áudio em blocos de 4 segundos.
    for start in range(0, len(sound_array), chunk_size):
        chunk = sound_array[start:start + chunk_size]
        # Calcula o timestamp para o início do segmento, em segundos.
        timestamp = start / sampling_rate

        # Se o bloco estiver vazio, pula
        if len(chunk) == 0:
            continue

        # Extrai características para o segmento
        inputs = feature_extractor(chunk, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to("cpu").float() for key in inputs}

        # Gera as predições sem atualizar os gradientes
        with torch.no_grad():
            logits = model(**inputs).logits

        # Calcula softmax e converte para numpy
        scores = functional.softmax(logits, dim=1).detach().cpu().numpy()[0]

        # Formata as saídas para cada emoção e seleciona as duas maiores pontuações
        outputs = [{
            "emo": config.id2label[i],
            "score": round(score * 100, 1)
        } for i, score in enumerate(scores)]
        outputs = sorted(outputs, key=lambda x: x["score"], reverse=True)[:2] # Pega os resultados mais altos

        predictions_list.append({
            "timestamp": timestamp,
            "predictions": outputs
        })

    return predictions_list


if __name__ == "__main__":
    # Exemplo de uso com o arquivo de áudio.
    audio_path = '../../Datasets/AVEC_2016/recordings_audio/dev_1.wav'
    results = predict_emotion_hubert(audio_path)
    for segment in results:
        print(f"Timestamp (sec): {segment['timestamp']}")
        for emo in segment['predictions']:
            print(f"  Emotion: {emo['emo']}, Score: {emo['score']}%")