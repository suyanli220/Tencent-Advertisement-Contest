import os
import math
import random
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

def set_env_runtime(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True

class RuntimeParams:
    IO_SRC = 'data/sample_data.parquet'
    CHECKPOINT_DIR = './model_weights'
    
    HIDDEN_DIM = 128
    HEAD_COUNT = 8
    BLOCK_DEPTH = 4
    FF_EXPANSION = 512
    
    LIMIT_LEN = 256
    BATCH_CAPACITY = 32
    TRAINING_ITER = 50
    LEARNING_RATE = 2e-4
    DECAY = 1e-5
    
    TEMP = 0.1
    SESSION_GAP = 1800
    TIME_SLOTS = 256
    
    ACCEL = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIXED_PRECISION = True

params = RuntimeParams()
os.makedirs(params.CHECKPOINT_DIR, exist_ok=True)
set_env_runtime()

class MatrixUnitaryOptimizer(torch.optim.Optimizer):
    """基于正交化更新的 Muon 优化器"""
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.01):
        spec = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, spec)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                
                grad = p.grad
                state = self.state[p]
                
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                
                m = state['exp_avg']
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                m.mul_(group['momentum']).add_(grad)
                

                if p.dim() >= 2:
                    flat_m = m.view(m.size(0), -1)
                    u, _, v = torch.svd_lowrank(flat_m, q=min(flat_m.shape))
                    delta = (u @ v.t()).view_as(p)
                else:
                    delta = m
                
                p.add_(delta, alpha=-group['lr'])

def generate_feature_map(dataset):
    """扫描数据建立 ID 索引与连续特征维度信息"""
    categorical_registry = defaultdict(set)
    numerical_dims = {}
    peak_item_idx = 0

    for _, row in dataset.iterrows():
        peak_item_idx = max(peak_item_idx, row['item_id'])
        
        for section in ['user_feature', 'item_feature']:
            for entry in row[section]:
                fid, ftype = entry['feature_id'], entry['feature_value_type']
                if 'int' in ftype:
                    val = entry.get('int_value') or entry.get('int_array')
                    if isinstance(val, list): [categorical_registry[fid].add(v) for v in val]
                    else: categorical_registry[fid].add(val)
                elif 'float_array' in ftype:
                    numerical_dims[fid] = len(entry['float_array'])

        seqs = row['seq_feature']
        for s_type in ['item_seq', 'action_seq']:
            if s_type in seqs:
                for entry in seqs[s_type]:
                    if 'int_array' in entry:
                        [categorical_registry[entry['feature_id']].add(v) for v in entry['int_array']]

    encoded_map = {fid: {v: i+1 for i, v in enumerate(sorted(vals))} 
                   for fid, vals in categorical_registry.items()}
    return encoded_map, numerical_dims, peak_item_idx

class RotaryPositionalBias(nn.Module):
    def __init__(self, dim, max_seq=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq = max_seq

    def forward(self, t):
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class SequentialTemporalBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.heads = heads
        self.scale = (d_model // heads) ** -0.5

    def forward(self, x, mask=None):
        src = self.ln1(x)
        b, n, c = src.shape
        
        qkv = self.qkv(src).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        dots = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            dots = dots.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = F.softmax(dots, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        
        x = x + self.drop(self.out_proj(out))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class DeepContextNet(nn.Module):
    """主模型架构：深度上下文推荐网络"""
    def __init__(self, cat_map, num_dims, peak_id):
        super().__init__()
        self.cat_map = cat_map
        
        self.item_bank = nn.Embedding(peak_id + 5, params.HIDDEN_DIM, padding_idx=0)
        self.act_bank = nn.Embedding(5, params.HIDDEN_DIM)
        self.time_bank = nn.Embedding(params.TIME_SLOTS, params.HIDDEN_DIM)
        
        self.cat_embs = nn.ModuleDict({
            str(fid): nn.Embedding(len(m)+1, params.HIDDEN_DIM, padding_idx=0)
            for fid, m in cat_map.items()
        })
        self.num_projs = nn.ModuleDict({
            str(fid): nn.Linear(d, params.HIDDEN_DIM, bias=False)
            for fid, d in num_dims.items()
        })

        self.global_token = nn.Parameter(torch.randn(1, 1, params.HIDDEN_DIM))
        self.transformers = nn.ModuleList([
            SequentialTemporalBlock(params.HIDDEN_DIM, params.HEAD_COUNT, params.FF_EXPANSION)
            for _ in range(params.BLOCK_DEPTH)
        ])
        
        self.bottleneck = nn.Linear(params.HIDDEN_DIM, params.HIDDEN_DIM)
        self.classifier = nn.Linear(params.HIDDEN_DIM, 1)

    def _encode_fields(self, field_list, device):
        if not field_list: return torch.zeros(1, params.HIDDEN_DIM, device=device)
        
        nodes = []
        for feat in field_list:
            fid, vtype = str(feat['feature_id']), feat['feature_value_type']
            vec = torch.zeros(params.HIDDEN_DIM, device=device)
            
            if 'int' in vtype:
                lookup = self.cat_map.get(int(fid), {})
                raw_val = feat.get('int_value') or feat.get('int_array')
                if isinstance(raw_val, list):
                    idxs = torch.tensor([lookup.get(v, 0) for v in raw_val], device=device)
                    vec = self.cat_embs[fid](idxs).mean(0)
                else:
                    vec = self.cat_embs[fid](torch.tensor(lookup.get(raw_val, 0), device=device))
            elif 'float_array' in vtype and fid in self.num_projs:
                raw_arr = torch.tensor(feat['float_array'], dtype=torch.float, device=device)
                vec = self.num_projs[fid](raw_arr)
            nodes.append(vec)
            
        return torch.stack(nodes).mean(0, keepdim=True)

    def forward(self, data_batch):
        dev = self.item_bank.weight.device
        b_size = data_batch['item_seq'].size(0)
        
        seq_feat = (self.item_bank(data_batch['item_seq'].to(dev)) + 
                    self.act_bank(data_batch['action_seq'].to(dev)) + 
                    self.time_bank(data_batch['time_diff'].to(dev)))
        
        u_nodes = torch.cat([self._encode_fields(f, dev) for f in data_batch['u_raw']], dim=0).unsqueeze(1)
        i_nodes = torch.cat([self._encode_fields(f, dev) for f in data_batch['i_raw']], dim=0).unsqueeze(1)
        
        cls_tok = self.global_token.expand(b_size, -1, -1)
        combined = torch.cat([cls_tok, u_nodes, i_nodes, seq_feat], dim=1)

        full_mask = torch.cat([
            torch.ones(b_size, 3, dtype=torch.bool, device=dev),
            data_batch['mask'].to(dev)
        ], dim=1)
        
        for block in self.transformers:
            combined = block(combined, full_mask)
            
        latent = combined[:, 0, :]
        return {
            'logits': self.classifier(F.relu(self.bottleneck(latent))).squeeze(-1),
            'embed': F.normalize(latent, dim=-1)
        }

class SequenceDataset(Dataset):
    def __init__(self, source_df, cat_map):
        self.data = source_df.to_dict('records')
        self.cat_map = cat_map

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        seqs = item['seq_feature']
        i_seq = seqs.get('item_seq', [{}])[0].get('int_array', [])
        a_seq = seqs.get('action_seq', [{}])[0].get('int_array', [])
        t_seq = seqs.get('action_seq', [{}])[1].get('int_array', [])

        n = min(len(i_seq), len(a_seq), len(t_seq), params.LIMIT_LEN)
        i_seq, a_seq, t_seq = i_seq[:n], a_seq[:n], t_seq[:n]

        t_diff = [0] + [min(int(math.log2(max(1, t_seq[i]-t_seq[i-1]))), params.TIME_SLOTS-1) for i in range(1, n)]
        
        return {
            'i_seq': i_seq, 'a_seq': a_seq, 't_diff': t_diff,
            'u_raw': item['user_feature'], 'i_raw': item['item_feature'],
            'label': 1.0 if item['label'][0]['action_type'] == 2 else 0.0
        }

def fast_collate(batch):
    max_n = max(len(x['i_seq']) for x in batch)
    
    res = defaultdict(list)
    for x in batch:
        pad = max_n - len(x['i_seq'])
        res['item_seq'].append(x['i_seq'] + [0]*pad)
        res['action_seq'].append(x['a_seq'] + [0]*pad)
        res['time_diff'].append(x['t_diff'] + [0]*pad)
        res['mask'].append([True]*len(x['i_seq']) + [False]*pad)
        res['u_raw'].append(x['u_raw'])
        res['i_raw'].append(x['i_raw'])
        res['label'].append(x['label'])
        
    return {
        'item_seq': torch.tensor(res['item_seq'], dtype=torch.long),
        'action_seq': torch.tensor(res['action_seq'], dtype=torch.long),
        'time_diff': torch.tensor(res['time_diff'], dtype=torch.long),
        'mask': torch.tensor(res['mask'], dtype=torch.bool),
        'u_raw': res['u_raw'], 'i_raw': res['i_raw'],
        'label': torch.tensor(res['label'], dtype=torch.float)
    }

class Engine:
    def __init__(self, model):
        self.model = model.to(params.ACCEL)
        self.opt = MatrixUnitaryOptimizer(model.parameters(), lr=params.LEARNING_RATE, weight_decay=params.DECAY)
        self.scaler = GradScaler(enabled=params.MIXED_PRECISION)
        self.criterion = nn.BCEWithLogitsLoss()

    def run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        summary_loss = []
        all_y, all_p = [], []
        
        for batch in loader:
            with autocast(enabled=params.MIXED_PRECISION):
                out = self.model(batch)
                loss = self.criterion(out['logits'], batch['label'].to(params.ACCEL))

            if is_train:
                self.opt.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            
            summary_loss.append(loss.item())
            all_y.extend(batch['label'].numpy())
            all_p.extend(torch.sigmoid(out['logits']).detach().cpu().numpy())
            
        return np.mean(summary_loss), roc_auc_score(all_y, all_p)

def main():
    print(" DeepContextNet 初始化中...")
    raw_df = pd.read_parquet(params.IO_SRC)
    c_map, n_dims, p_id = generate_feature_map(raw_df)
    
    train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(SequenceDataset(train_df, c_map), batch_size=params.BATCH_CAPACITY, 
                              shuffle=True, collate_fn=fast_collate)
    val_loader = DataLoader(SequenceDataset(val_df, c_map), batch_size=params.BATCH_CAPACITY, 
                            collate_fn=fast_collate)
    
    model = DeepContextNet(c_map, n_dims, p_id)
    trainer = Engine(model)
    
    best_auc = 0
    for epoch in range(params.TRAINING_ITER):
        t_loss, t_auc = trainer.run_epoch(train_loader)
        v_loss, v_auc = trainer.run_epoch(val_loader, is_train=False)
        
        status = "⭐" if v_auc > best_auc else "  "
        if v_auc > best_auc: best_auc = v_auc
            
        print(f"Epoch {epoch+1:02d} | Loss: {t_loss:.4f} | Val AUC: {v_auc:.4f} {status}")

if __name__ == '__main__':
    main()
