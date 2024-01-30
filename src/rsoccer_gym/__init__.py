from gymnasium.envs.registration import register

register(id="VSS-v0", entry_point="rsoccer_gym.environments.vss.vss_gym:VSSEnv", max_episode_steps=1200)

register(
    id="SSLStaticDefenders-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.static_defenders:SSLHWStaticDefendersEnv",
    kwargs={"field_type": 2},
    max_episode_steps=1000,
)

register(
    id="SSLDribbling-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.dribbling:SSLHWDribblingEnv",
    max_episode_steps=4800,
)

register(
    id="SSLContestedPossession-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.contested_possession:SSLContestedPossessionEnv",
    max_episode_steps=1200,
)

register(
    id="SSLPassEndurance-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.pass_endurance:SSLPassEnduranceEnv",
    max_episode_steps=1200,
)

register(
    id="SSLGoToBall-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.go_to_ball:SSLGoToBallEnv",
    max_episode_steps=1200,
)
