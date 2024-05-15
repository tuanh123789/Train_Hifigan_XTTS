import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from models.hifigan_decoder import HifiganGenerator
from configs.gpt_hifigan_config import GPTHifiganConfig

class Inferer:
    def __init__(self):
        # Add here the xtts_config path
        self.xtts_config_path = "XTTS-v2/config.json"
        # Add here the vocab file that you have used to train the model
        self.tokenizer_path = "XTTS-v2/vocab.json"
        # Add here the checkpoint that you want to do inference with
        self.xtts_checkpoint = "XTTS-v2/model.pth"
        # Add here the speaker reference
        self.speaker_reference = ["LJSpeech-1.1/wavs/LJ001-0001.wav"]
        self.hifigan_checkpoint_path = "outputs/run-May-15-2024_07+18AM-e6ee4c5/best_model.pth"
        self.hifigan_config = GPTHifiganConfig()

        self.hifigan_generator = self.load_hifigan_generator()
        self.model = self.load_xtts_checkpoint()


    def load_hifigan_generator(self):
        print("Loading model...")
        hifigan_generator = HifiganGenerator(in_channels=self.hifigan_config.gpt_latent_dim, out_channels=1, **self.hifigan_config.generator_model_params)
        hifigan_state_dict = torch.load(self.hifigan_checkpoint_path)["model"]
        hifigan_state_dict = {k.replace("model_g.", ""): v for k, v in hifigan_state_dict.items() if "model_g" in k}
        hifigan_generator.load_state_dict(hifigan_state_dict, strict=True)
        hifigan_generator.eval()
        hifigan_generator.remove_weight_norm()
        
        return hifigan_generator

    def load_xtts_checkpoint(self):
        config = XttsConfig()
        config.load_json(self.xtts_config_path)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_path=self.xtts_checkpoint, vocab_path=self.tokenizer_path, use_deepspeed=False)
        model.hifigan_decoder.waveform_decoder = self.hifigan_generator

        return model

    def infer(self, output_path):
        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=self.speaker_reference)

        print("Inference...")
        out = self.model.inference(
            "Hello my name is john",
            "en",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7, # Add custom parameters here
        )

        torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

if __name__ == "__main__":
    inferer = Inferer()
    inferer.infer("xtts_finetune_hifigan.wav")