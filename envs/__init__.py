from gym.envs.registration import register

register(
    id='FourRooms-v1',
    entry_point='fourrooms.fourrooms:FourRooms',
)
