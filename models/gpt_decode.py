from typing import Dict, List, Tuple, Union

from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer

class GPTDecode(GPTTrainer):
    def generate(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, num_samples, 80,t_m)
        cond_idxs: cond start and end indexs, (b, 2)
        cond_lens: long tensor, (b,)
        """
        latents = self.xtts.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
            return_latent=True
        )
        return latents

    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return GPTDecode(config)
    