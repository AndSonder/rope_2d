import paddle
from paddle.incubate.nn.functional import fused_rotary_position_embedding

batch_size = 2
seq_len = 16
num_heads = 64
head_dim = 64
rope_theta = 100.0

# q, k, v: [batch_size, seq_len, num_heads, head_dim]
q = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype='float16')
k = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype='float16')
v = paddle.randn([batch_size, seq_len, num_heads, head_dim], dtype='float16')


def init_t_xy(end_x: int, end_y: int):
    t = paddle.arange(end_x * end_y, dtype=paddle.float32)
    t_x = t % end_x
    t_y = paddle.divide(t, paddle.to_tensor(end_x, dtype=paddle.float32))
    return t_x, t_y

# ----- calculate the sin and cos values for 2D relative positional encoding -----
# sin, cos: [1, seq_len, 1, head_dim]
seq_len_x = seq_len_y = seq_len**0.5
t_x, t_y = init_t_xy(seq_len_x, seq_len_y)
thetas = 1.0 / (
    rope_theta ** (paddle.arange(0, head_dim, 4)[: (head_dim // 4)] / head_dim)
)  # [head_dim//4]

# thetas: [head_dim//2]
thetas = paddle.to_tensor([item for item in thetas for _ in range(2)])

# cos_x, cos_y, sin_x, sin_y: [seq_len, head_dim//2]
cos_x = paddle.cos(paddle.outer(t_x, thetas))
cos_y = paddle.cos(paddle.outer(t_y, thetas))
sin_x = paddle.sin(paddle.outer(t_x, thetas))
sin_y = paddle.sin(paddle.outer(t_y, thetas))

# cos, sin: [seq_len, head_dim]
cos = paddle.concat([cos_x, cos_y], axis=1)
sin = paddle.concat([sin_x, sin_y], axis=1)
# cos, sin: [1, seq_len, 1, head_dim]
cos = cos.unsqueeze(0).unsqueeze(2)
sin = sin.unsqueeze(0).unsqueeze(2)
print(cos.shape, sin.shape)
# [1, 16, 1, 64] [1, 16, 1, 64]
# --------------------------------------------------------------------------------

out_q, out_k, out_v = fused_rotary_position_embedding(q, k, v, sin=sin, cos=cos)
