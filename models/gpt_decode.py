from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F

from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer

class GPTDecode(GPTTrainer):
    def gpt(
        self,
        text_inputs,
        text_lengths,
        audio_codes,
        wav_lengths,
        cond_mels=None,
        cond_idxs=None,
        cond_lens=None,
        cond_latents=None,
        return_attentions=False,
        return_latent=False,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, 1, 80, s)
        cond_idxs: cond start and end indexs, (b, 2)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """
        # ❗ FIXIT
        if self.xtts.gpt.max_conditioning_inputs == 0:
            assert cond_mels is None, " ❗ cond_mels is not None, but max_conditioning_inputs == 0"

        max_text_len = text_lengths.max()
        code_lengths = torch.ceil(wav_lengths / self.xtts.gpt.code_stride_len).long() + 3

        if cond_lens is not None:
            if self.xtts.gpt.use_perceiver_resampler:
                cond_lens = cond_lens // self.xtts.gpt.perceiver_cond_length_compression
            else:
                cond_lens = cond_lens // self.xtts.gpt.code_stride_len

        if cond_idxs is not None:
            # recompute cond idxs for mel lengths
            for idx in range(cond_idxs.size(0)):
                if self.xtts.gpt.use_perceiver_resampler:
                    cond_idxs[idx] = cond_idxs[idx] // self.xtts.gpt.perceiver_cond_length_compression
                else:
                    cond_idxs[idx] = cond_idxs[idx] // self.xtts.gpt.code_stride_len

        # ensure that the cond_mel does not have padding
        # if cond_lens is not None and cond_idxs is None:
        #     min_cond_len = torch.min(cond_lens)
        #     cond_mels = cond_mels[:, :, :, :min_cond_len]

        # If len(codes) + 3 is larger than maxiumum allowed length, we truncate the codes.
        max_mel_len = code_lengths.max()

        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))

        # 💖 Lovely assertions
        assert (
            max_mel_len <= audio_codes.shape[-1]
        ), f" ❗ max_mel_len ({max_mel_len}) > audio_codes.shape[-1] ({audio_codes.shape[-1]})"
        assert (
            max_text_len <= text_inputs.shape[-1]
        ), f" ❗ max_text_len ({max_text_len}) > text_inputs.shape[-1] ({text_inputs.shape[-1]})"

        # Append stop token to text inputs
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.xtts.gpt.stop_text_token)

        # Append silence token to mel codes
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=self.xtts.gpt.stop_audio_token)

        # Pad mel codes with stop_audio_token
        audio_codes = self.xtts.gpt.set_mel_padding(
            audio_codes, code_lengths - 3
        )  # -3 to get the real code lengths without consider start and stop tokens that was not added yet

        # Build input and target tensors
        # Prepend start token to inputs and append stop token to targets
        text_inputs, text_targets = self.xtts.gpt.set_inputs_and_targets(
            text_inputs, self.xtts.gpt.start_text_token, self.xtts.gpt.stop_text_token
        )
        audio_codes, mel_targets = self.xtts.gpt.set_inputs_and_targets(
            audio_codes, self.xtts.gpt.start_audio_token, self.xtts.gpt.stop_audio_token
        )

        # Set attn_mask
        attn_mask_cond = None
        attn_mask_text = None
        attn_mask_mel = None

        if not return_latent:
            attn_mask_cond = torch.ones(
                cond_mels.shape[0],
                cond_mels.shape[-1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_text = torch.ones(
                text_inputs.shape[0],
                text_inputs.shape[1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_mel = torch.ones(
                audio_codes.shape[0],
                audio_codes.shape[1],
                dtype=torch.bool,
                device=audio_codes.device,
            )

            if cond_idxs is not None:
                # use masking approach
                for idx, r in enumerate(cond_idxs):
                    l = r[1] - r[0]
                    attn_mask_cond[idx, l:] = 0.0
            elif cond_lens is not None:
                for idx, l in enumerate(cond_lens):
                    attn_mask_cond[idx, l:] = 0.0

            for idx, l in enumerate(text_lengths):
                attn_mask_text[idx, l + 1 :] = 0.0

            for idx, l in enumerate(code_lengths):
                attn_mask_mel[idx, l + 1 :] = 0.0

        # Compute text embeddings + positional embeddings
        text_emb = self.xtts.gpt.text_embedding(text_inputs) + self.xtts.gpt.text_pos_embedding(text_inputs)

        # Compute mel embeddings + positional embeddings
        mel_emb = self.xtts.gpt.mel_embedding(audio_codes) + self.xtts.gpt.mel_pos_embedding(audio_codes)

        # Compute speech conditioning input
        if cond_latents is None:
            cond_latents = self.xtts.gpt.get_style_emb(cond_mels).transpose(1, 2)

        # Get logits
        sub = -5  # don't ask me why 😄
        if self.xtts.gpt.training:
            sub = -1
        text_logits, mel_logits = self.xtts.gpt.get_logits(
            text_emb,
            self.xtts.gpt.text_head,
            mel_emb,
            self.xtts.gpt.mel_head,
            prompt=cond_latents,
            get_attns=return_attentions,
            return_latent=return_latent,
            attn_mask_cond=attn_mask_cond,
            attn_mask_text=attn_mask_text,
            attn_mask_mel=attn_mask_mel,
        )

        return mel_logits[:, :sub], code_lengths  # sub to prevent bla.

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
        latents, code_lengths = self.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
            return_latent=True
        )
        return latents, code_lengths

    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return GPTDecode(config)
    