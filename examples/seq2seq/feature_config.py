SAMPLE_ID = "user_id"
SEQ_LEN = 1024
FEATURE_CONFIG = [
    {
        "name": "item_id",
        "emb_dim": 128,
        "hash_bucket_size": 2048000,
        "shard_name": "item_id",
    },
    {
        "name": "cate_id",
        "emb_dim": 128,
        "hash_bucket_size": 2048,
        "shard_name": "cate_id",
    },
    {
        "name": "behavior_id",
        "emb_dim": 128,
        "hash_type": "murmur",
        "hash_bucket_size": 0,
        "shard_name": "behavior_id",
    },
    {
        "name": "timestamp",
        "emb_dim": 128,
        "hash_type": "murmur",
        "hash_bucket_size": 2048000,
        "shard_name": "timestamp",
    },
]
