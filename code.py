import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

#model configs
batch_size = 64
block_size = 256 #max number of tokens processed in one go
max_iters = 10000
eval_interval = 1000 #steps at which it displays current mean loss, can be used to save model with best loss insteaf of at the last training step
learning_rate = 3e-4
device = 'cuda' #change to cpu is gpu not present, will take much longer to train, scale down the model
eval_iters = 200
n_embd = 32 #token embedding size
n_embd = 384
n_head = 6 #number of attentionheads
n_layer = 6 #number of transformer blocks
dropout = 0.2
torch.manual_seed(1234)


with open('train.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

#creating a dictionary of your vocab to be used for encoding and decoding
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#creating training and validation dataset
data = torch.tensor(encode(text),dtype = torch.long) #converting encoded training data into tensors
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


#getting random batches of data, use the given seed to get same batches
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix =  torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])    
    x,y = x.to(device), y.to(device)
    return x,y


#tells pytorch no backprop on this function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss =  model(X,Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()
    return out


#train a batch of data for the given number of max_iters
def train():
    for iter in range(max_iters):
        if iter%eval_interval==0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb,yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

#Single head of self attention
class Head(nn.Module):
    def __init__(self, head_seize):
        super().__init__()
        self.key = nn.Linear(n_embd, head_seize, bias=False)
        self.query = nn.Linear(n_embd, head_seize, bias=False)
        self.value = nn.Linear(n_embd, head_seize, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,P,E = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * E**-0.5
        wei =  wei.masked_fill(self.tril[:P, :P] == 0, float('-inf')) #filling a lower triangular matrix so that a token can only communicate with past tokens
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out =  wei @ v
        return out

#concatinating multiple attention heads
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

#a simple MLP on top of the attention block for additional computation
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd,4*n_embd),nn.ReLU(),nn.Linear(4*n_embd,n_embd), nn.Dropout(dropout)) #MLP at the end of attention head
    
    def forward(self, x):
        return self.net(x)

#single transformer block
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) #normalization to avoid overfitting
        self.ln2 = nn.LayerNorm(n_embd) #normalization to avoid overfitting
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #adding residual connections
        x = x + self.ffwd(self.ln2(x)) #adding residual connections
        return x

# a simple bigram language model but with extended context size as the default only looks back one step
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #creating token embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #creating position embeddingtable
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) #create as many transformer blocks as set in n_layer
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,P = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(P, device=device))
        x = token_emb+pos_emb #combing token embeddings with position embeddings
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets==None: #for generate mode
            loss = None
        else:
            B, P, E = logits.shape
            logits = logits.view(B*P, E)
            targets = targets.view(B*P)    
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1) #converting logits to probabilities, summing up to 1
            idx_next = torch.multinomial(probs, num_samples=1) #selects one token based on the multinomial distribution
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == 'train': #training mode
        model = BigramLanguageModel()
        m = model.to(device)
        optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
        train()
        torch.save(m,'./saved_gpt.pt') #save model for inference
    elif mode == 'generate': #generate mode
        model = BigramLanguageModel()
        model = torch.load('./saved_gpt.pt') #load saved model for generation
        m = model.to(device)
        input_text = sys.argv[2]
        input_tokens = encode(input_text)
        context = (torch.tensor(input_tokens, dtype=torch.long, device=device)[None, ...]) #convert input tokens into tensors
        print(decode(m.generate(context, max_new_tokens=50)[0].tolist()))
