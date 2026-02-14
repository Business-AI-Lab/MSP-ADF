import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel, Blip2ForConditionalGeneration, Blip2QFormerConfig, Blip2QFormerModel
import transformers
from peft import LoraConfig, TaskType, get_peft_model, IA3Config
import logging
from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model

import pytorch_lightning as pl
from torchmetrics import AUROC

from transformers import BitsAndBytesConfig

import math

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers.logging.set_verbosity_error()

token = "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" #"redacted"

OPERA_CT_TARGET_MODULES = ["qkv", "proj"]
OPERA_CE_TARGET_MODULES = ['conv', 'fc', 'linear']
target_module_dict = {"operaCT": OPERA_CT_TARGET_MODULES, "operaCE": OPERA_CE_TARGET_MODULES}
LLM_TARGET_MODULES = ["q_proj", "v_proj"]
LLM_TARGET_MODULES_ALLPROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]

class FlattenHead(nn.Module):
    def __init__(self, nf, out_dim, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, out_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x, no_fc=False):
        x = self.flatten(x)
        if no_fc:
            return x
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    """
    实现 DiffSoftmax，用于在训练中使用软标签或硬标签。
    - tau: 温度参数，控制 softmax 输出的平滑度
    - hard: 是否使用硬标签
    """
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class DynamicFusionLayer(nn.Module): #dynamic fusion
    def __init__(self, hidden_dim, tweak, tau=1.0, hard_gate=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.hard_gate = hard_gate

        # Gate network cho token-wise weighting
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # LayerNorm và learnable scale cho fused token
        self.fused_norm = nn.LayerNorm(hidden_dim)
        self.fuse_gate = nn.Parameter(torch.tensor(tweak))

    def forward(self, x):
        """
        x: (B, 5, H)
        return: (B, 6, H)
        """
        B, N, H = x.shape

        # compute per-token scores
        scores = self.gate_network(x).squeeze(-1)          # (B, N)
        weights = DiffSoftmax(scores, tau=self.tau, hard=self.hard_gate, dim=-1)  # (B, N)

        # weighted sum → fused token
        weighted = weights.unsqueeze(-1) * x               # (B, N, H)
        fused = weighted.sum(dim=1, keepdim=True)         # (B, 1, H)
        fused = self.fused_norm(fused)
        fused = self.fuse_gate * fused

        # concat original tokens + fused
        out = torch.cat([x, fused], dim=1)                # (B, N+1=6, H)
        return out

class MultiViewAudioEncoder(nn.Module): #multi-scale spectrogram prompting
    def __init__(self):
        super().__init__()

    def forward(self, spec):
        # spec: (B, F, T)
        B, F_dim, T_dim = spec.shape
        spec_unsq = spec.unsqueeze(1)  # (B,1,F,T)

        # --- View A: full res ---
        A = spec  # giữ nguyên

        # --- View B: downsample time, sau đó upsample lại T_dim ---
        Bv = F.avg_pool2d(spec_unsq, kernel_size=(1, 4))  # (B,1,F,T//4)
        Bv = F.interpolate(Bv, size=(F_dim, T_dim), mode='bilinear', align_corners=False).squeeze(1)

        # --- View C: downsample freq, sau đó upsample lại F_dim ---
        Cv = F.avg_pool2d(spec_unsq, kernel_size=(4, 1))  # (B,1,F//4,T)
        Cv = F.interpolate(Cv, size=(F_dim, T_dim), mode='bilinear', align_corners=False).squeeze(1)

        Dv = F.avg_pool2d(spec_unsq, kernel_size=(1, 2))  # (B,1,F,T//2)
        Dv = F.interpolate(Dv, size=(F_dim, T_dim), mode='bilinear', align_corners=False).squeeze(1)
        
        Ev = F.avg_pool2d(spec_unsq, kernel_size=(2, 1))  # (B,1,F//2,T)
        Ev = F.interpolate(Ev, size=(F_dim, T_dim), mode='bilinear', align_corners=False).squeeze(1)
        
        Fv = F.avg_pool2d(spec_unsq, kernel_size=(1, 6))  # (B,1,F,T//6)
        Fv = F.interpolate(Fv, size=(F_dim, T_dim), mode='bilinear', align_corners=False).squeeze(1)
        
        Gv = F.avg_pool2d(spec_unsq, kernel_size=(6, 1))  # (B,1,F//6,T)
        Gv = F.interpolate(Gv, size=(F_dim, T_dim), mode='bilinear', align_corners=False).squeeze(1)

        return A, Dv, Bv, Fv, Ev, Cv, Gv  # tất cả (B, F, T)

class MSP_ADF(nn.Module):

    def __init__(self, configs):
        super(MSP_ADF, self).__init__()

        self.loss = nn.CrossEntropyLoss()
        self.n_cls = configs.n_cls
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.audio_peft = configs.audio_peft
        self.d_audio = configs.enc_dim
        self.patch_nums = configs.patch_nums
        self.head_nf = self.d_ff * self.patch_nums

        self.llm_peft = configs.llm_peft
        self.llm_lora_rank = configs.llm_lora_rank
        self.llm_lora_alpha = configs.llm_lora_alpha
        self.llm_lora_dropout = configs.llm_lora_dropout
        
        self.attention = configs.attention # can be "bahdanau"
        self.num_heads = 1
        
        self.spread_audio_embedding = configs.spread_audio_embedding  # can be "multi_view"
        
        self.tweak = configs.tweak #5.4 

        self.use_audio = configs.use_audio
        self.use_context = configs.use_context

        self.pretrain_proj = nn.Linear(self.d_ff, 256 )# project LLM hidden dim -> audio embed dim
        self.audio_to_llm_proj = nn.Linear(self.d_audio, self.d_llm  )  # enc_dim=256, llm_dim=2304
        # self.audio_to_llm_proj = nn.Linear(768, self.d_llm  )
        self.lm_head = nn.Linear(self.d_llm, self.d_llm)


        # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        
        self.classifier_head = configs.classifier_head  # can be "flatten_head" or "one_peace_head"


        if configs.llm_model == 'llama':
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b') # 13.5G
            
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'llama2':
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b') # 13.5G
            model_id = "meta-llama/Llama-2-7b-chat-hf"
            # model_id = 'meta-llama/Llama-2-7b'
            self.llama_config = LlamaConfig.from_pretrained(model_id, token=token)
            self.tokenizer = LlamaTokenizer.from_pretrained(model_id, token=token)
            self.llm_model = LlamaModel.from_pretrained(model_id, token=token, config=self.llama_config)
        elif configs.llm_model == 'medalpaca':
            self.llama_config = LlamaConfig.from_pretrained("medalpaca/medalpaca-7b") # 13.5G
            
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == "OpenBioLLM":
            model_id = "aaditya/OpenBioLLM-Llama3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModel.from_pretrained(model_id)  
            # self.llama_config = LlamaConfig.from_pretrained(model_id) # 13.5G
            
            # try:
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=True,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # except EnvironmentError:  # downloads model from HF is not already done
            #     print("Local model files not found. Attempting to download...")
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # try:
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=True
            #     )
            # except EnvironmentError:  # downloads the tokenizer from HF if not already done
            #     print("Local tokenizer files not found. Atempting to download them..")
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False
            #     )
        elif configs.llm_model == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id)
            # self.llama_config = LlamaConfig.from_pretrained(model_id) # 13.5G
            
            # try:
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # except EnvironmentError:  # downloads model from HF is not already done
            #     print("Local model files not found. Attempting to download...")
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # try:
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False
            #     )
            # except EnvironmentError:  # downloads the tokenizer from HF if not already done
            #     print("Local tokenizer files not found. Atempting to download them..")
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False
            #     )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            # self.gpt2_config.num_hidden_layers = configs.llm_layers
            # self.gpt2_config.output_attentions = True
            # self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            # self.bert_config.num_hidden_layers = configs.llm_layers
            # self.bert_config.output_attentions = True
            # self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'mistral':
            model_id = "mistralai/Mistral-7B-v0.1"#"mistralai/Mistral-7B-v0.1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModel.from_pretrained(model_id, token = token)
        elif configs.llm_model == 'phi':
            model_id = "microsoft/Phi-3.5-mini-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, token = token, trust_remote_code=True)
        elif configs.llm_model == "gemma2B":
            # model_id = "google/gemma-2-2b-it"
            model_id = "google/gemma-2-2b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, token = token)
        elif configs.llm_model == "gemma9B":
            # model_id = "google/gemma-2-9b-it"
            model_id = "google/gemma-2-9b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, token = token)
            
        elif configs.llm_model == "olmo":
            model_id = "allenai/Olmo-3-7B-Think"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
            )
                        
        else:
            raise NotImplementedError('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        
        if self.llm_peft == "lora":
            self.peft_config = LoraConfig(
                r=self.llm_lora_rank, 
                lora_alpha=self.llm_lora_alpha, 
                lora_dropout=self.llm_lora_dropout,
                # target_modules=LLM_TARGET_MODULES_ALLPROJ,
            )
            if configs.llm_lora_allproj:
                self.peft_config = LoraConfig(
                    r=self.llm_lora_rank, 
                    lora_alpha=self.llm_lora_alpha, 
                    lora_dropout=self.llm_lora_dropout,
                    target_modules=LLM_TARGET_MODULES_ALLPROJ,
                )
            try:
                self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            except ValueError:
                print(self.llm_model)
                if configs.llm_model == "phi":
                    self.peft_config = LoraConfig(
                        r=self.llm_lora_rank, 
                        lora_alpha=self.llm_lora_alpha, 
                        lora_dropout=self.llm_lora_dropout,
                        target_modules=["qkv_proj"]
                    )
                else:
                    self.peft_config = LoraConfig(
                        r=self.llm_lora_rank, 
                        lora_alpha=self.llm_lora_alpha, 
                        lora_dropout=self.llm_lora_dropout,
                        target_modules=LLM_TARGET_MODULES
                    )
                self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            self.llm_model.print_trainable_parameters()
            print('LoRA Training LLM')
        elif self.llm_peft == "frozen":
            for param in self.llm_model.parameters():
                param.requires_grad = False
            print("freeze LLM")
        else:
            return NotImplementedError("LLM fine-tuning mode undefined")
        
        if configs.audio_encoder == "operaCT":
            self.audio_encoder = initialize_pretrained_model(configs.audio_encoder).encoder

        if self.audio_peft == "frozen":
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print("freeze audio encoder")
        elif self.audio_peft == "full":
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = True
            self.audio_encoder.train()
            print("full model fine-tune audio encoder")
        else:
            # peft
            if self.audio_peft == "lora":
                peft_config = LoraConfig(
                    # task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=configs.audio_lora_rank, lora_alpha=32, lora_dropout=0.1,
                    target_modules=target_module_dict[configs.audio_encoder]
                )
            elif self.audio_peft == "IA3":
                peft_config = IA3Config(
                    target_modules=target_module_dict[configs.audio_encoder],
                    feedforward_modules=['proj']
                )
            else:
                return NotImplementedError("audio fine-tuning mode undefined")
            self.audio_encoder = get_peft_model(self.audio_encoder, peft_config)
            self.audio_encoder.print_trainable_parameters()
            
        
        if self.spread_audio_embedding == "multi_view":
            self.multi_view = MultiViewAudioEncoder()
            # self.d_audio = self.d_audio  # assuming the encoder outputs the same dim for each view
            # self.patch_nums = self.patch_nums * 4  # 4 views
            # self.head_nf = self.d_ff * self.patch_nums
            print("Using multi-view audio encoder with spread audio embedding")
        else:
            print("Using single-view audio encoder without spread audio embedding")
        

        if configs.aligner == "projection":
            self.aligner = nn.Linear(self.d_audio, self.d_llm)
            print("Using linear projection aligner (default)")
        else:
            return NotImplementedError("aligner module undefined")
            
        if configs.attention == "dynamic":
            print(f"Tweak setting for Dynamic Fusion Layer: {self.tweak}")
            
            self.dynamic_fusion_layer = DynamicFusionLayer(self.d_llm, self.tweak)
            print("Using Dynamic Fusion attention")
            
        elif configs.attention == "none":
            print("No attention mechanism")
        
        self.head_dropout = configs.head_dropout
        
        if self.classifier_head == "flatten_head":
            
            self.output_projection = FlattenHead(self.head_nf, self.n_cls, head_dropout=self.head_dropout)
            print("Using FlattenHead classifier head (default)")
        
            
        self.embedding_projection = nn.Linear(self.head_nf, self.d_audio)

        self.print_trainable()

    def reinitialize_clf(self, n_cls):
        # self.output_projection = FlattenHead(self.head_nf, n_cls, head_dropout=self.head_dropout)
        if self.classifier_head == "flatten_head":
            
            self.output_projection = FlattenHead(self.head_nf, self.n_cls, head_dropout=self.head_dropout)
            print(f"Reinitialized classifier head with {n_cls} classes.")
            
    def print_trainable(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("total trainable parameters:", trainable_params)

    def reset_trainable(self):
        if self.llm_peft == "lora":
            for name, param in self.audio_encoder.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.llm_peft == "frozen":
            for param in self.llm_model.parameters():
                param.requires_grad = False
        
        for param in self.aligner.parameters():
            param.requires_grad = True
            
        if self.attention != "None":    
            if self.attention == "dynamic":     
                for param in self.dynamic_fusion_layer.parameters():
                    param.requires_grad = True
        else:
            for param in self.dynamic_fusion_layer.parameters():
                param.requires_grad = False

                
        for param in self.spread_audio_embedding.parameters():
            param.requires_grad = False
        
        if self.audio_peft == "frozen":
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        elif self.audio_peft == "full":
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
        
        for param in self.output_projection.parameters():
            param.requires_grad = True
        self.print_trainable()

    def forward(self, x_spectrogram, x_prompt, x_context, x_masked=None, no_fc=False, pretrain=False):
        if self.patch_nums == 1:
            if self.spread_audio_embedding == "multi_view":
                x_spectrogram_1, x_spectrogram_2, x_spectrogram_3, _, _, x_spectrogram_6, x_spectrogram_7 = self.multi_view(x_spectrogram)
                
                # print("x_spectrogram shape:", x_spectrogram.shape)
                x_enc = self.audio_encoder(x_spectrogram_1)
                x_enc_2 = self.audio_encoder(x_spectrogram_2)
                x_enc_3 = self.audio_encoder(x_spectrogram_3)
                # x_enc_4 = self.audio_encoder(x_spectrogram_4)
                # x_enc_5 = self.audio_encoder(x_spectrogram_5)
                x_enc_6 = self.audio_encoder(x_spectrogram_6)
                x_enc_7 = self.audio_encoder(x_spectrogram_7)
                
                # x_enc = self.audio_encoder(x_spectrogram)
                # print("x_enc shape:", x_enc.shape)
                enc_out = self.aligner(x_enc).unsqueeze(dim=1)
                enc_out_2 = self.aligner(x_enc_2).unsqueeze(dim=1)
                enc_out_3 = self.aligner(x_enc_3).unsqueeze(dim=1)
                # enc_out_4 = self.aligner(x_enc_4).unsqueeze(dim=1)
                # enc_out_5 = self.aligner(x_enc_5).unsqueeze(dim=1)
                enc_out_6 = self.aligner(x_enc_6).unsqueeze(dim=1)
                enc_out_7 = self.aligner(x_enc_7).unsqueeze(dim=1)
                
                # enc_out = torch.cat([enc_out, enc_out_2, enc_out_3], dim=1)
                enc_out = torch.cat([enc_out, enc_out_2, enc_out_3, enc_out_6, enc_out_7], dim=1)#, enc_out_4, enc_out_5], dim=1)
                                    #  , enc_out_6, enc_out_7], dim=1)
                #, enc_out_4], dim=1)  # (B, 4, d_llm)
                
                # return enc_out
                # print(f"aligner output shape: {enc_out.shape}")
                # end
                # enc_out = enc_out.unsqueeze(dim=1)
            else:
                x_enc = self.audio_encoder(x_spectrogram)
                # print("x_enc shape:", x_enc.shape)
                enc_out = self.aligner(x_enc)
                # print(f"aligner output shape: {enc_out.shape}")
                enc_out = enc_out.unsqueeze(dim=1)
        elif self.patch_nums == 64:
            x_enc = self.audio_encoder.forward_window(x_spectrogram)
            # print("x_enc shape:", x_enc.shape)
            
            # reshape to (B, N, 768)
            B, C, F, T = x_enc.shape
            x_enc = x_enc.reshape(B, C, F * T).permute(0, 2, 1)  # (B, 64, 768)
            # print("x_enc reshaped:", x_enc.shape)
            
            enc_out = self.aligner(x_enc)
            print("enc_out shape:", enc_out.shape)
        else:
            raise NotImplementedError
        
        prompt = self.tokenizer(x_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        context = self.tokenizer(x_context, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        context_embeddings = self.llm_model.get_input_embeddings()(context.to(x_enc.device))  # (batch, prompt_token, dim)

        if self.attention == "dynamic":
            # context_embs = torch.cat([prompt_embeddings, context_embeddings], dim=1)
            enc_out = self.dynamic_fusion_layer(enc_out)
            
            # print("Dynamic fusion output shape:", enc_out.shape)
            
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings, enc_out], dim=1)
        
        else:
            if self.use_audio and self.use_context:
                # print("Using audio embeddings!")
                llama_enc_out = torch.cat([prompt_embeddings, context_embeddings, enc_out], dim=1)
            elif self.use_audio and not self.use_context:
                # print("Warning: not using context embeddings!")
                llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
            else:
                # print("Warning: not using audio embeddings!")
                llama_enc_out = torch.cat([prompt_embeddings, context_embeddings], dim=1)

        dec_out = self.llm_model(inputs_embeds=llama_enc_out, output_hidden_states=True).hidden_states[-1]
        # print(dec_out)
        # print(dec_out.shape)
        
        dec_out = dec_out[:, :, :self.d_ff]
        # print(dec_out.shape)
        
        # text_output = hidden_state_to_text(dec_out, self.llm_model, self.tokenizer)
        # print(text_output)
        
        if self.classifier_head == "flatten_head":
            dec_out = dec_out.permute(0, 2, 1).contiguous()
            # print(dec_out.shape)
            
            # print("FlattenHead input:", dec_out[:, :, -self.patch_nums:])
            # print("FlattenHead input shape:", dec_out[:, :, -self.patch_nums:].shape)

            dec_out = self.output_projection(dec_out[:, :, -self.patch_nums:], no_fc=no_fc)
            
        return dec_out