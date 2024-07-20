import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import ipdb

class LanguageDecoder(nn.Module):
    def __init__(self,
                 pretrained_llm,
                 tokenizer_config,
                 decoder_config,
                 gradient_checkpointing_enable=False,
                 freeze_decoder=False,
                 precision='float32',
                 **kwargs):
        super(LanguageDecoder, self).__init__()
        if precision == "bf16-mixed":
            torch_dtype = torch.bfloat16
        elif precision == "16-mixed":
            torch_dtype = torch.float16
        elif precision == "32-true":
            torch_dtype = torch.float32
            decoder_config.attn_implementation = None
        else:
            raise ValueError(f"Invalid precision: {precision}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_llm, **tokenizer_config)
        self.add_eos_token = True if os.path.basename(pretrained_llm) in ['Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct'] else False

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.decoder = AutoModelForCausalLM.from_pretrained(pretrained_llm, 
                                                            torch_dtype=torch_dtype,
                                                            **decoder_config)
        if gradient_checkpointing_enable:
            self.decoder.gradient_checkpointing_enable()
        if freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def _tokenize(self, text, device='cpu'):
        input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'][0].to(device)
        if self.add_eos_token:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.tokenizer.eos_token_id]).to(device)], dim=0)
        return input_ids
    
    def _process(self, x, video_masks, subtitles, questions=None, previous_contexts=None, device='cpu', ignore_idx=-100):
        if previous_contexts is not None:
            questions = [q + ' The previous context is the following: ' + c + ' And the given word list is as follows: ' if c != '' \
                         else q + ' And the given word list is as follows: '  for q, c in zip(questions, previous_contexts)]
            
        max_len = 0
        labels = []
        inputs_embeds = []
        for i in range(x.shape[0]):
            cur_v_embed = x[i, :int(video_masks[i].sum())].to(device) # [num_v_tokens, C]

            
            cur_s = self._tokenize(subtitles[i], device=device)[1:] # remove bos token
            
            if questions is not None:
                cur_q = self._tokenize(questions[i], device=device)[:-1] # remove eos token
                cur_q_embed = self.decoder.model.embed_tokens(cur_q)
                
            cur_s_embed = self.decoder.model.embed_tokens(cur_s)

            if questions is not None:
                cur_embed = torch.cat([cur_q_embed, cur_v_embed, cur_s_embed], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens, C]
                cur_label = torch.cat([torch.LongTensor([ignore_idx]*(len(cur_q)+len(cur_v_embed))).to(device), cur_s], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens]
            
            else:
                # add bos token embedding to the seqymces and the bos token to the labels
                cur_v_embed = torch.cat([self.decoder.model.embed_tokens(torch.LongTensor([self.tokenizer.bos_token_id]).to(device)), cur_v_embed], dim=0)
                cur_embed = torch.cat([cur_v_embed, cur_s_embed], dim=0)
                cur_label = torch.cat([torch.LongTensor([ignore_idx]*len(cur_v_embed)).to(device), cur_s], dim=0)

            inputs_embeds.append(cur_embed)
            labels.append(cur_label)

            if len(cur_label) > max_len:
                max_len = len(cur_label)

        pad_embed = self.decoder.model.embed_tokens(torch.LongTensor([self.tokenizer.pad_token_id]).to(device))
        attn_masks = []
        for i in range(x.shape[0]):
            padded_len = max_len - len(labels[i])
            if self.tokenizer.padding_side == "left":
                attn_mask = torch.cat([torch.zeros(1, padded_len), torch.ones(1, len(labels[i]))], dim=1).to(device)
                inputs_embeds[i] = torch.cat([pad_embed.repeat(padded_len, 1), inputs_embeds[i]], dim=0)
                labels[i] = torch.cat([torch.LongTensor([ignore_idx]*padded_len).to(device), labels[i]], dim=0)
            elif self.tokenizer.padding_side == "right":
                attn_mask = torch.cat([torch.ones(1, len(labels[i])), torch.zeros(1, padded_len)], dim=1).to(device)
                inputs_embeds[i] = torch.cat([inputs_embeds[i], pad_embed.repeat(padded_len, 1)], dim=0)
                labels[i] = torch.cat([labels[i], torch.LongTensor([ignore_idx]*padded_len).to(device)], dim=0)
            attn_masks.append(attn_mask)
            assert len(inputs_embeds[i]) == len(labels[i])
        inputs_embeds = torch.stack(inputs_embeds, dim=0)
        attn_masks = torch.cat(attn_masks, dim=0)
        labels = torch.stack(labels, dim=0)

        return inputs_embeds[:, :-1], attn_masks[:, :-1], labels[:, 1:]
    
    def _process_predict(self, x, video_masks, subtitles, questions=None, previous_contexts=None, device='cpu', ignore_idx=-100):
        if previous_contexts is not None:
            questions = [q + ' The previous context is the following: ' + c + ' And the given word list is as follows: ' if c != '' \
                         else q + ' And the given word list is as follows: '  for q, c in zip(questions, previous_contexts)]
            
        max_len = 0
        inputs_embeds = []
        for i in range(x.shape[0]):
            cur_v_embed = x[i, :int(video_masks[i].sum())].to(device) # [num_v_tokens, C]
            
            if questions is not None:
                cur_q = self._tokenize(questions[i], device=device)[:-1] # remove eos token
                cur_q_embed = self.decoder.model.embed_tokens(cur_q)

            if questions is not None:
                cur_embed = torch.cat([cur_q_embed, cur_v_embed], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens, C]
            
            else:
                cur_v_embed = torch.cat([self.decoder.model.embed_tokens(torch.LongTensor([self.tokenizer.bos_token_id]).to(device)), cur_v_embed], dim=0)
                cur_embed = cur_v_embed

            inputs_embeds.append(cur_embed)

            if len(cur_embed) > max_len:
                max_len = len(cur_embed)
        
     

        pad_embed = self.decoder.model.embed_tokens(torch.LongTensor([self.tokenizer.pad_token_id]).to(device))
        attn_masks = []
        for i in range(x.shape[0]):
            padded_len = max_len - len(inputs_embeds[i])
            if self.tokenizer.padding_side == "left":
                attn_mask = torch.cat([torch.zeros(1, padded_len), torch.ones(1, len(inputs_embeds[i]))], dim=1).to(device)
                inputs_embeds[i] = torch.cat([pad_embed.repeat(padded_len, 1), inputs_embeds[i]], dim=0)
            elif self.tokenizer.padding_side == "right":
                attn_mask = torch.cat([torch.ones(1, len(inputs_embeds[i])), torch.zeros(1, padded_len)], dim=1).to(device)
                inputs_embeds[i] = torch.cat([inputs_embeds[i], pad_embed.repeat(padded_len, 1)], dim=0)
            attn_masks.append(attn_mask)
        

        inputs_embeds = torch.stack(inputs_embeds, dim=0).to(device, dtype=next(self.decoder.parameters()).dtype)
        attn_masks = torch.cat(attn_masks, dim=0).to(device, dtype=next(self.decoder.parameters()).dtype)

        return inputs_embeds, attn_masks

    def forward(self, x, video_masks, subtitles, questions=None, previous_contexts=None):
        inputs_embeds, attn_masks, labels = self._process(x, video_masks, subtitles, questions, previous_contexts, device=x.device)
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_masks, return_dict=True)
        
        return outputs, labels
    
    def predict(self, x, video_masks, subtitles, questions=None, previous_contexts=None):
        inputs_embeds, attn_masks = self._process_predict(x, video_masks, subtitles, questions, previous_contexts, device=x.device)
        outputs = self.decoder.generate(inputs_embeds=inputs_embeds, attention_mask=attn_masks, max_new_tokens=50)
        
        return outputs
