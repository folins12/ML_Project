USE_SLIPPERY = False  

MAP_NAME = '8x8'

params_slippery = {
    "learning_rate": 0.0003,
    "discount_factor": 0.99,
    "network_sync_rate": 1000,
    "replay_memory_size": 50000,
    "mini_batch_size": 256,
    "epsilon_start": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 0.99995,
    "episodes": 20000,
    "hidden_nodes": 128,
    "test_ep_slippery": 2000
}

params_nonslippery = {
    "learning_rate": 0.002,
    "discount_factor": 0.85,
    "network_sync_rate": 300,
    "replay_memory_size": 8000,
    "mini_batch_size": 64,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00066,
    "episodes": 2000,
    "hidden_nodes": 128,
    "test_ep_nonslippery": 200
}





def get_config():
    params = params_slippery if USE_SLIPPERY else params_nonslippery
    return {
        "slippery": USE_SLIPPERY,
        "map_name": MAP_NAME,
        **params
    }
