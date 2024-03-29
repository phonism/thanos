from gym.envs.registration import register

register(
    id='Gomoku19x19-v0',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'beginner', # beginner opponent policy has defend and strike rules
        'board_size': 19,
    },
    nondeterministic=True,
)

register(
    id='Gomoku15x15-v0',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'beginner', # beginner opponent policy has defend and strike rules
        'board_size': 15,
    },
    nondeterministic=True,
)

register(
    id='Gomoku9x9-v0',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'beginner', # random policy is the simplest
        'board_size': 9,
    },
    nondeterministic=True,
)

register(
    id='Gomoku6x6-v0',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random', # random policy is the simplest
        'board_size': 6,
    },
    nondeterministic=True,
)

