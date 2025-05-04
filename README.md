# BfloatSequencePredictor

bfloat16 ↔ (sign, exp, mantessa) → sinusoidal embed for ordinal → FiLM style scale head for modulation → predict 


```python
from module import bfloat16_to_components, components_to_bfloat16, BfloatSequencePredictor
import torch

# convert
s,e,m = bfloat16_to_components(torch.randn(10, dtype=torch.bfloat16))
x_rec = components_to_bfloat16(s,e,m)

# model
model = BfloatSequencePredictor(model_dim=512, embed_dim=256, num_floats=2)
x   = torch.randn(8,16,512)
s_t = torch.randint(0,2,   (8,16,2))
e_t = torch.randint(0,256, (8,16,2))
m_t = torch.randint(0,128, (8,16,2))

loss = model(x, s_t, e_t, m_t)
s,e,m = model.predict(x[:,0,:], top_p=0.9)
```
