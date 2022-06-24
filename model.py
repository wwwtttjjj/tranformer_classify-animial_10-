import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

'''pair function'''
def pair(t):
    return t if isinstance(t, tuple) else (t,t)

'''preNorm'''
'''layernorm层'''
class PreNrom(torch.nn.Module):
    def __init__(self, dim, fn):
        super(PreNrom, self).__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

'''FeedforWard'''
class FeedForWard(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForWard, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

'''Attention'''
class Attention(torch.nn.Module):
    def __init__(self, dim, heads = 8, dim_head=64,dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not(heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = torch.nn.Softmax(dim=-1)
        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout),
        ) if project_out else torch.nn.Identity()
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        #softmax(qxk^t / (根号d))xV


        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn = self.attend(dots)
        out = torch.einsum('b h i j , b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

'''Transform'''
class Transform(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transform, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                PreNrom(dim, Attention(dim, heads=heads, dim_head=dim_head,dropout=dropout)),
                PreNrom(dim, FeedForWard(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

'''vit'''
class Vit(torch.nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels = 3, dim_head=64,dropout=0., emb_dropout = 0.):
        super(Vit,self).__init__()
        #return height and width, in general, input is one number
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in ('cls', 'mean')

        #分为N, divideNum, embedding
        self.to_patch_embedding = torch.nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w)(p1 p2 c)', p1 = patch_height, p2 = patch_width),
            torch.nn.Linear(patch_dim, dim))

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches + 1, dim))
        #like nlp start flag
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = torch.nn.Dropout(emb_dropout)
        self.transformer = Transform(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = torch.nn.Identity()
        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, num_classes))
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_token = repeat(self.cls_token, '() n d->b n d', b = b)
        x =torch.cat((cls_token, x), dim=1)
        
        x += self.pos_embedding[:,:(n + 1)]


        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == "mean" else x[:, 0]

        x =self.to_latent(x)
        return self.mlp_head(x)

if __name__ == '__main__':
    model = Vit(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    model = model.to(torch.device('cuda'))
    x = torch.randn(16, 3, 256, 256).to(torch.device('cuda'))
    preds = model(x)
    print(preds.shape)


