"""
StintEngine — F1 Pit Stop Strategy Gymnasium Environment (v2 — Weather)
10D observation space with fuel, safety car, rain, and wet compound logic.
"""

import gymnasium
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, List

from src.config import (
    TOTAL_LAPS, NUM_DRIVERS, STARTING_COMPOUND, STARTING_POSITION,
    PIT_STOP_TIME_LOSS, MAX_TYRE_AGE, TYRE_CLIFF_AGE,
    COMPOUND_TO_IDX, IDX_TO_COMPOUND, COMPOUNDS,
    REWARD_POSITION_GAIN, REWARD_PIT_COST,
    REWARD_TYRE_CLIFF_PENALTY, REWARD_FINISH_BONUS_SCALE,
    REWARD_SC_PIT_BONUS, REWARD_WRONG_TYRE_PENALTY,
    FUEL_LOAD_KG, FUEL_BURN_PER_LAP, FUEL_TIME_PENALTY,
    COLD_TYRE_PENALTY, COLD_TYRE_WARMUP_LAPS,
    SAFETY_CAR_PROBABILITY, SAFETY_CAR_MIN_LAPS, SAFETY_CAR_MAX_LAPS,
    SAFETY_CAR_LAPTIME, GAP_NORMALIZER,
    RAIN_PROBABILITY, RAIN_MIN_LAPS, RAIN_MAX_LAPS,
    DRY_ON_WET_PENALTY, WET_ON_DRY_PENALTY,
    INTER_RAIN_BONUS, WET_HEAVY_RAIN_BONUS,
    DRY_COMPOUNDS, WET_COMPOUNDS,
)
from src.tyre_model import predict_laptime, get_base_laptime


class F1StrategyEnv(gymnasium.Env):
    """
    Advanced F1 Pit Stop Strategy Environment v2 — with Weather.

    Observation (10D, normalized [0,1]):
        [0] lap_progress      — current_lap / total_laps
        [1] tyre_age           — normalised tyre age
        [2] compound           — SOFT=0, MED=0.25, HARD=0.5, INTER=0.75, WET=1.0
        [3] lap_time_delta     — delta from base lap time
        [4] position           — position / 20
        [5] fuel_load          — remaining fuel fraction
        [6] gap_to_leader      — normalised gap in seconds
        [7] safety_car         — 1.0 if SC active, else 0.0
        [8] rain_intensity     — 0.0=dry, 0.5=light, 1.0=heavy
        [9] is_raining         — 1.0 if rain active, else 0.0

    Actions (Discrete(6)):
        0 = Stay Out
        1 = Pit for Soft
        2 = Pit for Medium
        3 = Pit for Hard
        4 = Pit for Intermediates
        5 = Pit for Full Wets
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tyre_coefficients: Optional[Dict] = None,
        total_laps: int = TOTAL_LAPS,
        render_mode: Optional[str] = None,
        weather_mode: str = "Random",
        sc_probability: float = SAFETY_CAR_PROBABILITY,
        starting_position: int = STARTING_POSITION,
    ):
        super().__init__()

        self.total_laps = total_laps
        self.render_mode = render_mode
        self.weather_mode = weather_mode
        self.sc_probability = sc_probability
        self.starting_pos_override = starting_position

        # Tyre degradation model
        if tyre_coefficients is None:
            self.tyre_coefficients = {
                "SOFT":   (91.0, 0.08, 0.003),
                "MEDIUM": (91.5, 0.05, 0.002),
                "HARD":   (92.0, 0.03, 0.001),
                "INTER":  (93.0, 0.04, 0.001),  # slower on dry but durable
                "WET":    (95.0, 0.03, 0.0008), # slowest on dry, very durable
            }
        else:
            # Ensure wet compounds have defaults if not in fitted data
            if "INTER" not in tyre_coefficients:
                tyre_coefficients["INTER"] = (93.0, 0.04, 0.001)
            if "WET" not in tyre_coefficients:
                tyre_coefficients["WET"] = (95.0, 0.03, 0.0008)
            self.tyre_coefficients = tyre_coefficients

        self.base_laptime = get_base_laptime(self.tyre_coefficients)

        # Spaces — 10D observation, 6 actions
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)

        # State (initialized in reset)
        self._init_state()

    def _init_state(self):
        """Initialize all state variables."""
        self.current_lap = 1
        self.tyre_age = 1
        self.compound = STARTING_COMPOUND
        self.position = self.starting_pos_override
        self.last_laptime = self.base_laptime
        self.pit_stops = 0
        self.pit_laps: List[int] = []

        # Advanced state
        self.fuel_kg = FUEL_LOAD_KG
        self.cumulative_time = 0.0
        self.leader_time = 0.0
        self.gap_to_leader = 0.0
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0
        self.cold_tyre_laps = 0

        # Weather state
        self.is_raining = (self.weather_mode == "Force Rain")
        self.rain_intensity = 2 if self.is_raining else 0
        self.rain_laps_remaining = 999 if self.is_raining else 0

        # Race log for UI replay
        self.race_log: List[dict] = []

    def _get_obs(self) -> np.ndarray:
        """Build normalised 10D observation vector."""
        compound_norm = COMPOUND_TO_IDX[self.compound] / (len(COMPOUNDS) - 1)
        lap_delta = np.clip((self.last_laptime - self.base_laptime) / 10.0, 0.0, 1.0)
        fuel_frac = max(self.fuel_kg / FUEL_LOAD_KG, 0.0)
        gap_norm = np.clip(self.gap_to_leader / GAP_NORMALIZER, 0.0, 1.0)
        rain_norm = self.rain_intensity / 2.0  # 0, 0.5, 1.0

        return np.array([
            self.current_lap / self.total_laps,       # [0] lap progress
            min(self.tyre_age / MAX_TYRE_AGE, 1.0),   # [1] tyre age
            compound_norm,                              # [2] compound type
            lap_delta,                                  # [3] lap time delta
            self.position / NUM_DRIVERS,               # [4] position
            fuel_frac,                                  # [5] fuel load
            gap_norm,                                   # [6] gap to leader
            1.0 if self.safety_car_active else 0.0,    # [7] safety car
            rain_norm,                                  # [8] rain intensity
            1.0 if self.is_raining else 0.0,           # [9] is raining
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "lap": self.current_lap,
            "tyre_age": self.tyre_age,
            "compound": self.compound,
            "position": self.position,
            "pit_stops": self.pit_stops,
            "pit_laps": list(self.pit_laps),
            "laptime": round(self.last_laptime, 3),
            "fuel_kg": round(self.fuel_kg, 2),
            "gap_to_leader": round(self.gap_to_leader, 2),
            "safety_car": self.safety_car_active,
            "cumulative_time": round(self.cumulative_time, 3),
            "is_raining": self.is_raining,
            "rain_intensity": self.rain_intensity,
        }

    def _log_lap(self, action: int, reward: float):
        """Store lap data for UI replay."""
        self.race_log.append({
            **self._get_info(),
            "action": action,
            "reward": round(reward, 3),
        })

    def _weather_penalty(self) -> float:
        """Calculate penalty/bonus for tyre-weather mismatch."""
        if self.is_raining:
            # Rain is active
            if self.compound in DRY_COMPOUNDS:
                # Dry tyres in rain = dangerous + slow
                return DRY_ON_WET_PENALTY * (1 + self.rain_intensity * 0.5)
            elif self.compound == "INTER":
                if self.rain_intensity == 1:
                    return INTER_RAIN_BONUS  # optimal
                elif self.rain_intensity == 2:
                    return 1.5  # inters struggle in heavy rain
                return 0.0
            elif self.compound == "WET":
                if self.rain_intensity == 2:
                    return WET_HEAVY_RAIN_BONUS  # optimal
                elif self.rain_intensity == 1:
                    return 0.5  # full wets overkill in light rain
                return 0.0
        else:
            # Dry conditions
            if self.compound in WET_COMPOUNDS:
                return WET_ON_DRY_PENALTY  # wet tyres on dry = very slow
        return 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._init_state()
        self.last_laptime = predict_laptime(self.compound, self.tyre_age, self.tyre_coefficients)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        pitted_this_lap = False

        # ══════════════════════════════════════════════════════════════════
        # WEATHER SYSTEM
        # ══════════════════════════════════════════════════════════════════
        if self.weather_mode == "Force Rain":
            self.is_raining = True
            self.rain_intensity = 2
            self.rain_laps_remaining = 999
        elif self.weather_mode == "Force Dry":
            self.is_raining = False
            self.rain_intensity = 0
            self.rain_laps_remaining = 0
        else:
            if self.is_raining:
                self.rain_laps_remaining -= 1
                if self.rain_laps_remaining <= 0:
                    self.is_raining = False
                    self.rain_intensity = 0
                elif self.np_random.random() < 0.15:
                    self.rain_intensity = self.np_random.integers(1, 3)
            else:
                if self.np_random.random() < RAIN_PROBABILITY:
                    self.is_raining = True
                    self.rain_intensity = self.np_random.integers(1, 3)
                    self.rain_laps_remaining = self.np_random.integers(
                        RAIN_MIN_LAPS, RAIN_MAX_LAPS + 1
                    )

        # ══════════════════════════════════════════════════════════════════
        # SAFETY CAR LOGIC
        # ══════════════════════════════════════════════════════════════════
        if self.safety_car_active:
            self.safety_car_laps_remaining -= 1
            if self.safety_car_laps_remaining <= 0:
                self.safety_car_active = False
        else:
            sc_prob = self.sc_probability
            if self.is_raining:
                sc_prob *= 2.5  # rain increases SC chance
            if self.np_random.random() < sc_prob:
                self.safety_car_active = True
                self.safety_car_laps_remaining = self.np_random.integers(
                    SAFETY_CAR_MIN_LAPS, SAFETY_CAR_MAX_LAPS + 1
                )

        # ══════════════════════════════════════════════════════════════════
        # PIT STOP
        # ══════════════════════════════════════════════════════════════════
        if action > 0:
            new_compound = IDX_TO_COMPOUND[action - 1]

            if new_compound != self.compound:
                self.compound = new_compound
                self.tyre_age = 1
                self.pit_stops += 1
                self.pit_laps.append(self.current_lap)
                pitted_this_lap = True
                self.cold_tyre_laps = COLD_TYRE_WARMUP_LAPS

                if self.safety_car_active:
                    positions_lost = self.np_random.integers(0, 2)
                    reward += REWARD_SC_PIT_BONUS
                else:
                    positions_lost = int(PIT_STOP_TIME_LOSS / 1.5)
                    positions_lost = max(1, positions_lost + self.np_random.integers(-2, 3))

                self.position = min(NUM_DRIVERS, self.position + positions_lost)
                self.cumulative_time += PIT_STOP_TIME_LOSS
                reward += REWARD_PIT_COST

        # ══════════════════════════════════════════════════════════════════
        # SIMULATE LAP
        # ══════════════════════════════════════════════════════════════════
        if self.safety_car_active:
            self.last_laptime = SAFETY_CAR_LAPTIME
        else:
            # Base tyre degradation
            self.last_laptime = predict_laptime(
                self.compound, self.tyre_age, self.tyre_coefficients
            )

            # Fuel effect: lighter car = faster
            fuel_penalty = self.fuel_kg * FUEL_TIME_PENALTY
            self.last_laptime += fuel_penalty

            # Cold tyre penalty (out-lap after pit)
            if self.cold_tyre_laps > 0:
                warmup_factor = self.cold_tyre_laps / COLD_TYRE_WARMUP_LAPS
                self.last_laptime += COLD_TYRE_PENALTY * warmup_factor
                self.cold_tyre_laps -= 1

            # Weather tyre mismatch penalty/bonus
            weather_delta = self._weather_penalty()
            self.last_laptime += weather_delta
            if abs(weather_delta) > 3.0:
                reward += REWARD_WRONG_TYRE_PENALTY

        # Burn fuel
        self.fuel_kg = max(0.0, self.fuel_kg - FUEL_BURN_PER_LAP)

        # Update cumulative time
        self.cumulative_time += self.last_laptime

        # Leader time (simplified)
        leader_lap = self.base_laptime + max(0, self.fuel_kg * FUEL_TIME_PENALTY * 0.8)
        if self.safety_car_active:
            leader_lap = SAFETY_CAR_LAPTIME
        if self.is_raining:
            leader_lap += 3.0 * self.rain_intensity  # leader also slowed by rain
        self.leader_time += leader_lap
        self.gap_to_leader = max(0.0, self.cumulative_time - self.leader_time)

        # ══════════════════════════════════════════════════════════════════
        # POSITION CHANGES (skip during SC — positions frozen)
        # ══════════════════════════════════════════════════════════════════
        if not self.safety_car_active and not pitted_this_lap:
            tyre_perf = self.last_laptime - self.base_laptime
            if tyre_perf < 1.5:
                if self.np_random.random() < 0.55:
                    old_pos = self.position
                    self.position = max(1, self.position - 1)
                    reward += REWARD_POSITION_GAIN * (old_pos - self.position)
            elif tyre_perf > 3.5:
                if self.np_random.random() < 0.35:
                    old_pos = self.position
                    self.position = min(NUM_DRIVERS, self.position + 1)
                    reward += REWARD_POSITION_GAIN * (old_pos - self.position)

        # ══════════════════════════════════════════════════════════════════
        # TYRE CLIFF PENALTY
        # ══════════════════════════════════════════════════════════════════
        if self.tyre_age > TYRE_CLIFF_AGE:
            reward += REWARD_TYRE_CLIFF_PENALTY

        # ══════════════════════════════════════════════════════════════════
        # ADVANCE STATE
        # ══════════════════════════════════════════════════════════════════
        self.tyre_age += 1
        self.current_lap += 1

        terminated = self.current_lap > self.total_laps
        truncated = False

        if terminated:
            reward += REWARD_FINISH_BONUS_SCALE * (NUM_DRIVERS - self.position) / NUM_DRIVERS

        # Log for replay
        self._log_lap(action, reward)

        return self._get_obs(), reward, terminated, truncated, self._get_info()


# ─── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running F1StrategyEnv v2 (Weather) smoke test...\n")

    env = F1StrategyEnv()

    for ep in range(3):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        sc_laps = 0
        rain_laps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if info.get("safety_car"):
                sc_laps += 1
            if info.get("is_raining"):
                rain_laps += 1

        print(f"Ep {ep+1}: Pos=P{info['position']}, Pits={info['pit_stops']}, "
              f"SC={sc_laps}L, Rain={rain_laps}L, "
              f"Fuel={info['fuel_kg']:.1f}kg, Reward={total_reward:.2f}")

    print("\n✓ Weather smoke test passed")

    print("\nRunning check_env()...")
    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)
    print("✓ check_env passed (10D observation space)")
