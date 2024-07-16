import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Phi3ForCausalLM

class LanguageDecoder(nn.Module):
    def __init__(self,
                 pretrained_llm,
                 hidden_size,
                 gradient_checkpointing_enable=False,
                 freeze_decoder=False,
                 precision='float32',
                 attn_implementation="flash_attention_2",
                 **kwargs):
        super(LanguageDecoder, self).__init__()
        assert pretrained_llm is not None, 'Pretrained LLM must be provided'
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_llm, add_eos_token=True, padding_side='left', trust_remote_code=True, use_fast=True)
        if precision == "bf16-mixed":
            torch_dtype = torch.bfloat16
        elif precision == "16-mixed":
            torch_dtype = torch.float16
        elif precision == "32-true":
            torch_dtype = torch.float32
            attn_implementation = None
        else:
            raise ValueError(f"Invalid precision: {precision}")
        
        self.decoder = AutoModelForCausalLM.from_pretrained(pretrained_llm, 
                                                            trust_remote_code=True,
                                                            torch_dtype=torch_dtype,
                                                            attn_implementation=attn_implementation)
        if gradient_checkpointing_enable:
            self.decoder.gradient_checkpointing_enable()
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def _process(self, x, video_masks, subtitles, questions=None, previous_contexts=None, device='cpu', ignore_idx=-100):
        if previous_contexts is not None:
            questions = [q + ' The previous context is the following: ' + c + ' And the given word list is as follows: ' if c != '' \
                         else q + ' And the given word list is as follows: '  for q, c in zip(questions, previous_contexts)]
            
        max_len = 0
        labels = []
        inputs_embeds = []
        for i in range(x.shape[0]):
            cur_v_embed = x[i, :int(video_masks[i].sum())].to(device) # [num_v_tokens, C]
            cur_q = self.tokenizer(questions[i], return_tensors='pt')['input_ids'][0][:-1].to(device) # [num_q_tokens]
            cur_s = self.tokenizer(subtitles[i], return_tensors='pt')['input_ids'][0][1:].to(device) # [num_s_tokens]

            cur_q_embed = self.decoder.model.embed_tokens(cur_q)
            cur_s_embed = self.decoder.model.embed_tokens(cur_s)

            cur_embed = torch.cat([cur_q_embed, cur_v_embed, cur_s_embed], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens, C]
            cur_label = torch.cat([torch.LongTensor([ignore_idx]*(len(cur_q)+len(cur_v_embed))).to(device), cur_s], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens]

            inputs_embeds.append(cur_embed)
            labels.append(cur_label)

            if len(cur_label) > max_len:
                max_len = len(cur_label)

        pad_embed = self.decoder.model.embed_tokens(torch.LongTensor([self.tokenizer.pad_token_id]).to(device))
        attn_masks = []
        for i in range(x.shape[0]):
            padded_len = max_len - len(labels[i])
            # pad cur_embed and cur_label
            attn_mask = torch.cat([torch.zeros(1, padded_len), torch.ones(1, len(labels[i]))], dim=1).to(device)
            inputs_embeds[i] = torch.cat([pad_embed.repeat(padded_len, 1), inputs_embeds[i]], dim=0)
            labels[i] = torch.cat([torch.LongTensor([ignore_idx]*padded_len).to(device), labels[i]], dim=0)
            attn_masks.append(attn_mask)
            assert len(inputs_embeds[i]) == len(labels[i])
        inputs_embeds = torch.stack(inputs_embeds, dim=0)
        attn_masks = torch.cat(attn_masks, dim=0)
        labels = torch.stack(labels, dim=0)

        return inputs_embeds[:, :-1], attn_masks[:, :-1], labels[:, 1:]

    def forward(self, x, video_masks, subtitles, questions=None, previous_contexts=None):
        inputs_embeds, attn_masks, labels = self._process(x, video_masks, subtitles, questions, previous_contexts, device=x.device)
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_masks, return_dict=True)
        return outputs, labels
