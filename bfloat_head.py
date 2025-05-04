
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def bfloat16_to_components(x: torch.Tensor):
    bits16 = x.view(torch.int16)
    ubits  = bits16.to(torch.int32) & 0xFFFF
    sign = (ubits >> 15) & 0x1
    exp  = (ubits >>  7) & 0xFF
    man  =  ubits        & 0x7F
    return sign, exp, man

def components_to_bfloat16(sign: torch.Tensor,
                           exp:  torch.Tensor,
                           man:  torch.Tensor) -> torch.Tensor:
    bits16 = ((sign.to(torch.int32) << 15) |
              (exp .to(torch.int32) <<  7) |
              (man .to(torch.int32)      ))
    bits32 = bits16 << 16
    floats32 = bits32.view(torch.float32)
    return floats32.to(torch.bfloat16)


class BfloatSequencePredictor(nn.Module):
    def __init__(self,
                 model_dim:   int = 768,
                 embed_dim:   int = 768,
                 num_floats:  int = 1,
                 exp_classes: int = 256,
                 man_classes: int = 128):
        super().__init__()
        self.model_dim   = model_dim
        self.embed_dim   = embed_dim
        self.num_floats  = num_floats
        self.exp_classes = exp_classes
        self.man_classes = man_classes

        self.sin_embed = nn.Linear(embed_dim, embed_dim)

        # heads + FiLM gates
        self.sign_heads = nn.ModuleList()
        self.exp_projs  = nn.ModuleList()
        self.man_projs  = nn.ModuleList()
        self.e_gates    = nn.ModuleList()   # just gamma, no beta
        self.m_gates    = nn.ModuleList()

        for i in range(num_floats):
            prior_dim = embed_dim * 3 * i

            # sign
            self.sign_heads.append(
                nn.Linear(model_dim + prior_dim, 2)
            )
            # project into embedding space
            self.exp_projs .append(
                nn.Linear(model_dim + prior_dim + embed_dim, embed_dim)
            )
            self.man_projs .append(
                nn.Linear(model_dim + prior_dim + embed_dim*2, embed_dim)
            )
            self.e_gates   .append(
                nn.Linear(model_dim + prior_dim + embed_dim, embed_dim)
            )
            self.m_gates   .append(
                nn.Linear(model_dim + prior_dim + embed_dim*2, embed_dim)
            )

        # buffers of IDs
        self.register_buffer('exp_ids', torch.arange(exp_classes))
        self.register_buffer('man_ids', torch.arange(man_classes))

    def sinusoidal_embedding(self, ids: torch.Tensor) -> torch.Tensor:
        # standard sin/cos to linear
        x = ids.unsqueeze(-1).float()  # [...,1]
        half = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=x.device) * -(math.log(10000.0)/(half-1))
        )           # [half]
        ang  = x * freqs  # [..., half]
        emb  = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if self.embed_dim % 2:
            emb = F.pad(emb, (0,1))
        return self.sin_embed(emb)  # [..., E]

    def forward(self, x, s_t, e_t, m_t):
        # flatten batch/time
        if x.ndim==3:
            B,S,D = x.shape
            x_flat = x.view(B*S, D)
            s_flat = s_t.view(B*S, -1)
            e_flat = e_t.view(B*S, -1)
            m_flat = m_t.view(B*S, -1)
        elif x.ndim==2:
            x_flat, s_flat, e_flat, m_flat = x, s_t, e_t, m_t
            B = x_flat.size(0)
        else:
            raise ValueError

        # precompute static tables once
        e_base = self.sinusoidal_embedding(self.exp_ids)  # [C_e, E]
        m_base = self.sinusoidal_embedding(self.man_ids)  # [C_m, E]

        total_loss = 0.0
        prev_embs  = []

        for i in range(self.num_floats):
            # - sign -
            h_s      = torch.cat([x_flat, *prev_embs], dim=-1) if prev_embs else x_flat
            logits_s = self.sign_heads[i](h_s)
            total_loss += F.cross_entropy(logits_s, s_flat[:,i])
            s_emb    = self.sinusoidal_embedding(s_flat[:,i])

            h_e     = torch.cat([x_flat, *prev_embs, s_emb], dim=-1)
            phi_e   = self.exp_projs[i](h_e)      # [B,E]
            gamma_e = torch.sigmoid(self.e_gates[i](h_e))  # [B,E]
            phi_e   = phi_e * gamma_e             # [B,E]
            logits_e= phi_e @ e_base.T            # [B, C_e]
            total_loss += F.cross_entropy(logits_e, e_flat[:,i])
            e_emb   = self.sinusoidal_embedding(e_flat[:,i])

            # - mantissa with FiLM scale -
            h_m     = torch.cat([x_flat, *prev_embs, s_emb, e_emb], dim=-1)
            phi_m   = self.man_projs[i](h_m)      # [B,E]
            gamma_m = torch.sigmoid(self.m_gates[i](h_m))  # [B,E]
            phi_m   = phi_m * gamma_m             # [B,E]
            logits_m= phi_m @ m_base.T            # [B, C_m]
            total_loss += F.cross_entropy(logits_m, m_flat[:,i])
            m_emb   = self.sinusoidal_embedding(m_flat[:,i])

            prev_embs.extend([s_emb, e_emb, m_emb])

        return total_loss

    @torch.no_grad()
    def predict(self, x, top_p=1.0):
        # same flatten logic...
        if x.ndim==3:
            B,S,D = x.shape
            x_flat = x.view(B*S, D)
        elif x.ndim==2:
            x_flat = x
        else:
            raise ValueError

        # precompute
        e_base = self.sinusoidal_embedding(self.exp_ids)  # [C_e, E]
        m_base = self.sinusoidal_embedding(self.man_ids)  # [C_m, E]

        prev_embs = []
        s_list,e_list,m_list = [],[],[]

        for i in range(self.num_floats):
            # sign
            h_s      = torch.cat([x_flat,*prev_embs], dim=-1) if prev_embs else x_flat
            s_i      = self._top_p_sample(self.sign_heads[i](h_s), top_p)
            s_emb    = self.sinusoidal_embedding(s_i)
            s_list.append(s_i)

            # exponent
            h_e      = torch.cat([x_flat,*prev_embs,s_emb], dim=-1)
            phi_e    = self.exp_projs[i](h_e)
            gamma_e  = torch.sigmoid(self.e_gates[i](h_e))
            phi_e_mod= phi_e * gamma_e
            e_i      = self._top_p_sample(phi_e_mod @ e_base.T, top_p)
            e_emb    = self.sinusoidal_embedding(e_i)
            e_list.append(e_i)

            # mantissa
            h_m      = torch.cat([x_flat,*prev_embs,s_emb,e_emb], dim=-1)
            phi_m    = self.man_projs[i](h_m)
            gamma_m  = torch.sigmoid(self.m_gates[i](h_m))
            phi_m_mod= phi_m * gamma_m
            m_i      = self._top_p_sample(phi_m_mod @ m_base.T, top_p)
            m_emb    = self.sinusoidal_embedding(m_i)
            m_list.append(m_i)

            prev_embs.extend([s_emb,e_emb,m_emb])

        # stack & reshape back if needed
        s_flat = torch.stack(s_list, dim=1)
        e_flat = torch.stack(e_list, dim=1)
        m_flat = torch.stack(m_list, dim=1)
        if x.ndim==3:
            return (s_flat.view(B,S,-1),
                    e_flat.view(B,S,-1),
                    m_flat.view(B,S,-1))
        return s_flat, e_flat, m_flat

    def _top_p_sample(self, logits, top_p):
        if top_p>=1.0:
            return logits.argmax(-1)
        probs = F.softmax(logits, -1)
        picks = []
        for p in probs:
            vals, idxs = p.sort(descending=True)
            cum = vals.cumsum(0)
            cut = max(1, torch.searchsorted(cum, top_p).item())
            top = idxs[:cut]
            w   = p[top]
            w  /= w.sum()
            picks.append(top[torch.multinomial(w,1).item()])
        return torch.tensor(picks, device=logits.device)
