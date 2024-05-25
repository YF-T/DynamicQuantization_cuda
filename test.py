from k_cache_file import K_Cache_Class
from v_cache_file import V_Cache_Class
import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)
import math
import os
import time

base_path = "/u/user75/llama2/llama"
q_path = os.path.join(base_path, "q_caches.pt")
score_path = os.path.join(base_path, "score_caches.pt")
kv_path = os.path.join(base_path, "kv_caches.pt")

token_len = 256
q_caches = torch.load(q_path, map_location=torch.device('cpu')) # (layer, args.max_batch_size,args.max_seq_len,self.n_local_kv_heads,self.head_dim,)
score_caches = torch.load(score_path, map_location=torch.device('cpu')) # (layer, args.max_batch_size,self.n_local_kv_heads,args.max_seq_len,args.max_seq_len,)
kv_caches = torch.load(kv_path, map_location=torch.device('cpu')) # (2, layer, args.max_batch_size,args.max_seq_len,self.n_local_kv_heads,self.head_dim,)
q_caches = q_caches[:, :, 0:token_len, :4, :64]
score_caches = score_caches[:, :, :4, 0:token_len, 0:token_len]
kv_caches = kv_caches[:, :, :, 0:token_len, :4, :64]
# q_caches = torch.zeros((1, 1, token_len, 4, 64), dtype=torch.int16)
# kv_caches = torch.zeros((2, 1, 1, token_len, 4, 64), dtype=torch.int16)
# score_cache = torch.zeros((1, 1, 4, token_len, token_len), dtype=torch.int16)
# q_caches = torch.zeros_like(q_caches, dtype=torch.int16)
# kv_caches = torch.zeros_like(kv_caches, dtype=torch.int16)

# q_caches = 0x3c00 + q_caches
# kv_caches = 0x3c18 + kv_caches
#
# q_caches = q_caches.view(torch.half)
# kv_caches = kv_caches.view(torch.half)

# q_caches = torch.rand_like(q_caches, dtype=torch.half)
# kv_caches = torch.rand_like(kv_caches, dtype=torch.half)
# score_caches = torch.rand_like(score_caches, dtype=torch.half)
#
# q_caches = (torch.zeros_like(q_caches, dtype=torch.int16) + torch.tensor([0x3a00, 0x3b00, 0x3c00, 0x3d00] * 16,dtype=torch.int16)).view(torch.half)
# kv_caches = (torch.zeros_like(kv_caches, dtype=torch.int16) + torch.tensor([0x3a00, 0x3b00, 0x3c00, 0x3d00] * 16,dtype=torch.int16)).view(torch.half)
# score_caches = (torch.zeros_like(score_caches, dtype=torch.int16) + torch.tensor([0x3c00],dtype=torch.int16)).view(torch.half)

total_seqlen = 128

#　input 64 output 128
bsz = 1
n_local_kv_heads = 4
head_dim = 64

gpu_device = torch.device('cuda:1')
k_cache_model = K_Cache_Class(bsz, total_seqlen, n_local_kv_heads, head_dim, device=gpu_device)
v_cache_model = V_Cache_Class(bsz, total_seqlen, n_local_kv_heads, head_dim, device=gpu_device)

def prefill_k(seqlen: int):
    q = q_caches[0, :, 0:seqlen, :, :].to(gpu_device)
    k = kv_caches[0, 0, :, 0:seqlen, :, :].to(gpu_device)
    q = torch.repeat_interleave(q, bsz // q.shape[0], dim=0)
    k = torch.repeat_interleave(k, bsz // k.shape[0], dim=0)
    # my result
    k_cache_model.save(k, 0, seqlen)
    
def decoding_compute_k(start_pos: int, seqlen: int):
    q = q_caches[0, :, start_pos:start_pos + seqlen, :, :].to(gpu_device)
    q = torch.repeat_interleave(q, bsz // q.shape[0], dim=0)
    k_new = kv_caches[0, 0, :, start_pos:start_pos + seqlen, :, :].to(gpu_device)
    k_new = torch.repeat_interleave(k_new, bsz // k_new.shape[0], dim=0)
    # my result
    k_cache_model.save(k_new, start_pos, seqlen)
    result = k_cache_model.decoding_compute(q, start_pos, seqlen)[:, :, :, :start_pos + seqlen]
    result2 = k_cache_model.decoding_compute(q, start_pos, seqlen, reference=True)[:, :, :, :start_pos + seqlen]
    # reference result
    k_total = kv_caches[0, 0, :, 0:start_pos + seqlen, :, :]
    k_total = k_total.to(gpu_device)
    k_total = torch.repeat_interleave(k_total, bsz // k_total.shape[0], dim=0)
    k_total = k_total.transpose(1, 2) # (bsz, n_head, seqlen, head_dim)
    q = q.to(gpu_device).transpose(1, 2) # (bsz, n_head, 1, head_dim)
    scores = torch.matmul(q, k_total.transpose(2, 3)) / math.sqrt(q.shape[-1]) # (bsz, n_head, 1, seqlen)
    # compare
    # print("q: ", q)
    # print("k_total: ", k_total)
    # print("k_total sum: ", torch.sum(k_total, dim=-1))
    print("result shape: ", result.shape)
    print(result)
    print("result2 shape: ", result2.shape)
    print(result2)
    print("scores shape: ", scores.shape)
    print(scores)
    print("stack result, result2, scores: ", torch.stack([result, result2, scores], dim=-1))

def prefill_v(seqlen: int):
    s = score_caches[0, :, :, 0:seqlen, 0:seqlen].to(gpu_device)
    v = kv_caches[1, 0, :, 0:seqlen, :, :].to(gpu_device)
    s = torch.repeat_interleave(s, bsz // s.shape[0], dim=0)
    v = torch.repeat_interleave(v, bsz // v.shape[0], dim=0)
    # my result
    # print("v: ", v)
    v_cache_model.save(v, 0, seqlen)
    # print("first 8: ", v_cache_model.v_cache_first_8)

def decoding_compute_v(start_pos: int, seqlen: int):
    s = score_caches[0, :, :, start_pos:start_pos + seqlen, 0:start_pos + seqlen].to(gpu_device)
    s = torch.repeat_interleave(s, bsz // s.shape[0], dim=0)
    v_new = kv_caches[1, 0, :, start_pos:start_pos + seqlen, :, :].to(gpu_device)
    v_new = torch.repeat_interleave(v_new, bsz // v_new.shape[0], dim=0)
    # my result
    # print("v_new: ", v_new)
    v_cache_model.save(v_new, start_pos, seqlen)
    # print("first 8: ", v_cache_model.v_cache_first_8)
    result = v_cache_model.decoding_compute(s, start_pos, seqlen, reference=False)
    result2 = v_cache_model.decoding_compute(s, start_pos, seqlen, reference=True)
    # reference result
    v_total = kv_caches[1, 0, :, 0:start_pos + seqlen, :, :]
    v_total = v_total.to(gpu_device)
    v_total = torch.repeat_interleave(v_total, bsz // v_total.shape[0], dim=0)
    v_total = v_total.transpose(1, 2) # (bsz, n_head, seqlen, head_dim)
    s = s.to(gpu_device) # (bsz, n_head, 1, seqlen)
    o = torch.matmul(s, v_total)
    sum = torch.zeros_like(o)
    for i in range(start_pos + seqlen):
        sum += v_total[:, :, i:i+1, :] * s[:, :, :, i:i+1]
    # compare
    # print("result shape: ", result.shape)
    # print(result)
    # print("result2 shape: ", result2.shape)
    # print(result2)
    # print("o shape: ", o.shape)
    # print(o)
    print("stack result, result2, o: ", torch.stack([result, result2, o, sum], dim=-1))

n = 50
prefill_v(n)
decoding_compute_v(n, 1)
decoding_compute_v(n+1, 1)