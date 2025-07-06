import gymnasium as gym
import numpy as np
from gymnasium import spaces
import requests
import json
from typing import Dict, Any, Tuple, Optional
import time


class MiningEnvironment(gym.Env):
    """
    OpenAI Gym environment for OSRS Mining skill training using PPO.

    Based on the MiningEnv contract and inspired by Naton1's OSRS PvP RL approach.
    Communicates with Elvarg RSPS server via HTTP API.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, server_host: str = "localhost", server_port: int = 8080,
                 max_steps: int = 10000, max_duration_minutes: int = 60):
        super().__init__()

        # Server connection
        self.server_host = server_host
        self.server_port = server_port
        self.base_url = f"http://{server_host}:{server_port}/api/mining"

        # Episode configuration
        self.max_steps = max_steps
        self.max_duration_minutes = max_duration_minutes
        self.current_step = 0
        self.episode_start_time = None
        self.session_id = None

        # Define action space (hierarchical actions flattened)
        self.action_space = self._create_action_space()

        # Define observation space
        self.observation_space = self._create_observation_space()

        # Tracking
        self.last_observation = None
        self.last_mining_xp = 0
        self.last_mining_level = 1
        self.total_reward = 0.0

    def _create_action_space(self) -> spaces.Discrete:
        """
        Create discrete action space mapping from the hierarchical action structure.

        Actions (22 total):
        Movement (4): no_move, move_to_rock, move_to_bank, move_to_position
        Mining (3): no_mine, mine_rock, mine_nearest
        Inventory (5): no_inventory_action, drop_item, drop_all_ore, bank_all, bank_ore_only
        Wait (3): wait_tick, wait_mining, wait_movement
        Special (7): move_to_rock variants by rock type
        """
        # We'll map complex actions to discrete indices
        # For simplicity, we'll use most common combinations
        return spaces.Discrete(22)

    def _create_observation_space(self) -> spaces.Box:
        """
        Create observation space based on the environment contract.

        Total: 34 observations
        - Player state (9): position_x, position_y, hitpoints, mining_level, mining_xp,
                           total_xp_gained, is_moving, is_mining, animation_id
        - Inventory state (10): inventory_count, inventory_free_slots, has_pickaxe,
                              pickaxe_type, ore_count_total, tin_ore_count, copper_ore_count,
                              iron_ore_count, coal_count
        - Environment state (9): nearest_rock_distance, nearest_rock_type, nearest_rock_available,
                               rocks_in_area_count, available_rocks_count, bank_distance,
                               at_bank, current_area, area_efficiency
        - Session state (6): game_tick, session_duration, xp_per_hour, actions_since_xp,
                           successful_mines_count, failed_mines_count
        """
        low = np.array([
            # Player state
            0, 0, 1, 1, 0, 0, 0, 0, 0,
            # position_x, position_y, hitpoints, mining_level, mining_xp, total_xp_gained, is_moving, is_mining, animation_id
            # Inventory state
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            # inventory_count, inventory_free_slots, has_pickaxe, pickaxe_type, ore_count_total, tin_ore_count, copper_ore_count, iron_ore_count, coal_count
            # Environment state
            0.0, 0, 0, 0, 0, 0.0, 0, 0, 0.0,
            # nearest_rock_distance, nearest_rock_type, nearest_rock_available, rocks_in_area_count, available_rocks_count, bank_distance, at_bank, current_area, area_efficiency
            # Session state
            0, 0.0, 0.0, 0, 0, 0
            # game_tick, session_duration, xp_per_hour, actions_since_xp, successful_mines_count, failed_mines_count
        ], dtype=np.float32)

        high = np.array([
            # Player state
            6400, 6400, 99, 99, 200000000, 1000000, 1, 1, 10000,
            # Inventory state
            28, 28, 1, 7, 28, 28, 28, 28, 28, 28,  # pickaxe_type: 0=none, 1=bronze, ..., 7=dragon
            # Environment state
            50.0, 7, 1, 20, 20, 100.0, 1, 6, 1.0,  # current_area: 0=lumbridge_mine, ..., 6=other
            # Session state
            2147483647, 1440.0, 100000.0, 1000, 10000, 10000
        ], dtype=np.float32)

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _action_to_request(self, action: int) -> Dict[str, Any]:
        """Convert discrete action to API request."""
        action_map = {
            0: {"type": "movement", "action": "no_move"},
            1: {"type": "movement", "action": "move_to_rock", "params": {"rock_type": "tin"}},
            2: {"type": "movement", "action": "move_to_rock", "params": {"rock_type": "copper"}},
            3: {"type": "movement", "action": "move_to_rock", "params": {"rock_type": "iron"}},
            4: {"type": "movement", "action": "move_to_rock", "params": {"rock_type": "coal"}},
            5: {"type": "movement", "action": "move_to_bank"},
            6: {"type": "mining", "action": "no_mine"},
            7: {"type": "mining", "action": "mine_nearest"},
            8: {"type": "mining", "action": "mine_rock", "params": {"rock_type": "tin"}},
            9: {"type": "mining", "action": "mine_rock", "params": {"rock_type": "copper"}},
            10: {"type": "mining", "action": "mine_rock", "params": {"rock_type": "iron"}},
            11: {"type": "mining", "action": "mine_rock", "params": {"rock_type": "coal"}},
            12: {"type": "inventory", "action": "no_inventory_action"},
            13: {"type": "inventory", "action": "drop_all_ore"},
            14: {"type": "inventory", "action": "bank_all"},
            15: {"type": "inventory", "action": "bank_ore_only"},
            16: {"type": "wait", "action": "wait_tick"},
            17: {"type": "wait", "action": "wait_mining"},
            18: {"type": "wait", "action": "wait_movement"},
            19: {"type": "movement", "action": "move_to_rock", "params": {"rock_type": "mithril"}},
            20: {"type": "movement", "action": "move_to_rock", "params": {"rock_type": "adamant"}},
            21: {"type": "movement", "action": "move_to_rock", "params": {"rock_type": "runite"}},
        }
        return action_map.get(action, {"type": "wait", "action": "wait_tick"})

    def _make_request(self, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request to Elvarg server."""
        url = f"{self.base_url}/{endpoint}"
        try:
            if data is None:
                response = requests.get(url, timeout=5.0)
            else:
                response = requests.post(url, json=data, timeout=5.0)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return {"success": False, "error": str(e)}

    def _parse_observation(self, server_response: Dict[str, Any]) -> np.ndarray:
        """Parse server response into observation vector."""
        obs = server_response.get("observation", {})

        # Extract observation components
        player = obs.get("player_state", {})
        inventory = obs.get("inventory_state", {})
        environment = obs.get("environment_state", {})
        session = obs.get("session_state", {})

        # Map categorical values to numbers
        pickaxe_type_map = {"none": 0, "bronze": 1, "iron": 2, "steel": 3,
                            "mithril": 4, "adamant": 5, "rune": 6, "dragon": 7}
        rock_type_map = {"none": 0, "tin": 1, "copper": 2, "iron": 3,
                         "coal": 4, "mithril": 5, "adamant": 6, "runite": 7}
        area_map = {"lumbridge_mine": 0, "varrock_east_mine": 1, "varrock_west_mine": 2,
                    "al_kharid_mine": 3, "dwarven_mine": 4, "mining_guild": 5, "other": 6}

        observation = np.array([
            # Player state (9)
            player.get("position_x", 0),
            player.get("position_y", 0),
            player.get("hitpoints", 10),
            player.get("mining_level", 1),
            player.get("mining_xp", 0),
            player.get("total_xp_gained", 0),
            float(player.get("is_moving", False)),
            float(player.get("is_mining", False)),
            player.get("animation_id", 0),

            # Inventory state (10)
            inventory.get("inventory_count", 0),
            inventory.get("inventory_free_slots", 28),
            float(inventory.get("has_pickaxe", False)),
            pickaxe_type_map.get(inventory.get("pickaxe_type", "none"), 0),
            inventory.get("ore_count_total", 0),
            inventory.get("tin_ore_count", 0),
            inventory.get("copper_ore_count", 0),
            inventory.get("iron_ore_count", 0),
            inventory.get("coal_count", 0),
            0,  # Placeholder for additional inventory data

            # Environment state (9)
            environment.get("nearest_rock_distance", 50.0),
            rock_type_map.get(environment.get("nearest_rock_type", "none"), 0),
            float(environment.get("nearest_rock_available", False)),
            environment.get("rocks_in_area_count", 0),
            environment.get("available_rocks_count", 0),
            environment.get("bank_distance", 100.0),
            float(environment.get("at_bank", False)),
            area_map.get(environment.get("current_area", "other"), 6),
            environment.get("area_efficiency", 0.0),

            # Session state (6)
            session.get("game_tick", 0),
            session.get("session_duration", 0.0),
            session.get("xp_per_hour", 0.0),
            session.get("actions_since_xp", 0),
            session.get("successful_mines_count", 0),
            session.get("failed_mines_count", 0),
        ], dtype=np.float32)

        return observation

    def _calculate_reward(self, obs: np.ndarray, server_response: Dict[str, Any]) -> float:
        """Calculate reward based on observation and server response."""
        reward = 0.0

        # Extract current values
        current_mining_xp = obs[4]  # mining_xp
        current_mining_level = obs[3]  # mining_level
        is_mining = obs[7]  # is_mining
        inventory_count = obs[9]  # inventory_count
        actions_since_xp = obs[27]  # actions_since_xp
        xp_per_hour = obs[26]  # xp_per_hour

        # XP gained reward (primary)
        xp_gained = current_mining_xp - self.last_mining_xp
        if xp_gained > 0:
            reward += xp_gained * 1.0  # Base XP reward

        # Level up bonus
        if current_mining_level > self.last_mining_level:
            reward += 100.0  # Level up bonus

        # Efficiency bonus (maintaining high XP/hour)
        if xp_per_hour > 10000:  # Above 10k xp/hour
            reward += 0.1 * (xp_per_hour / 10000)

        # Inventory management reward
        if inventory_count == 28:  # Full inventory
            reward += 0.5

        # Idle penalty
        if actions_since_xp > 50:  # Too many actions without XP
            reward -= 0.1 * (actions_since_xp / 50)

        # Action-specific penalties from server
        penalties = server_response.get("penalties", {})
        reward -= penalties.get("failed_action_penalty", 0)
        reward -= penalties.get("idle_penalty", 0)
        reward -= penalties.get("death_penalty", 0)

        # Update tracking
        self.last_mining_xp = current_mining_xp
        self.last_mining_level = current_mining_level

        return reward

    def _check_done(self, obs: np.ndarray, server_response: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if episode should terminate."""
        info = {}

        # Max steps reached
        if self.current_step >= self.max_steps:
            return True, {"reason": "max_steps", "success": False}

        # Max duration reached
        if self.episode_start_time:
            duration_minutes = (time.time() - self.episode_start_time) / 60
            if duration_minutes >= self.max_duration_minutes:
                return True, {"reason": "max_duration", "success": False}

        # Success conditions
        current_level = obs[3]  # mining_level
        if current_level > self.last_mining_level:
            return True, {"reason": "level_up", "success": True}

        # Failure conditions
        hitpoints = obs[2]  # hitpoints
        if hitpoints <= 0:
            return True, {"reason": "death", "success": False}

        # Check server-defined conditions
        episode_status = server_response.get("episode_status", {})
        if episode_status.get("done", False):
            return True, {"reason": episode_status.get("reason", "server_end"),
                          "success": episode_status.get("success", False)}

        return False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset tracking
        self.current_step = 0
        self.episode_start_time = time.time()
        self.total_reward = 0.0

        # Initialize session with server
        reset_response = self._make_request("reset", {"seed": seed, "options": options})
        self.session_id = reset_response.get("session_id")

        # Get initial observation
        obs_response = self._make_request("observation")
        observation = self._parse_observation(obs_response)

        # Initialize tracking values
        self.last_mining_xp = observation[4]  # mining_xp
        self.last_mining_level = observation[3]  # mining_level
        self.last_observation = observation

        info = {"session_id": self.session_id, "step": self.current_step}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return next state."""
        self.current_step += 1

        # Convert action to server request
        action_request = self._action_to_request(action)
        action_request["session_id"] = self.session_id
        action_request["step"] = self.current_step

        # Send action to server
        step_response = self._make_request("step", action_request)

        # Parse observation
        observation = self._parse_observation(step_response)

        # Calculate reward
        reward = self._calculate_reward(observation, step_response)
        self.total_reward += reward

        # Check if episode is done
        terminated, info = self._check_done(observation, step_response)
        truncated = False  # We handle truncation in terminated

        # Update info
        info.update({
            "step": self.current_step,
            "total_reward": self.total_reward,
            "session_id": self.session_id,
            "action_executed": action_request,
            "server_response": step_response.get("action_result", {})
        })

        self.last_observation = observation
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment state."""
        if mode == "human" and self.last_observation is not None:
            obs = self.last_observation
            print(f"\n=== Mining Environment Step {self.current_step} ===")
            print(f"Position: ({obs[0]:.0f}, {obs[1]:.0f})")
            print(f"Mining Level: {obs[3]:.0f} (XP: {obs[4]:.0f})")
            print(f"HP: {obs[2]:.0f}")
            print(f"Inventory: {obs[9]:.0f}/28 ({obs[14]:.0f} ore)")
            print(f"Is Mining: {bool(obs[7])}")
            print(f"Nearest Rock: {obs[15]:.1f} tiles away")
            print(f"XP/Hour: {obs[26]:.0f}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print("=" * 40)

    def close(self):
        """Clean up environment."""
        if self.session_id:
            self._make_request("close", {"session_id": self.session_id})
        super().close()


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = MiningEnvironment(server_host="localhost", server_port=8080)

    # Test basic functionality
    print("Testing MiningEnvironment...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Example episode (will fail without server running)
    try:
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Reset info: {info}")

        # Take a few random actions
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            print(f"Action: {action}, Reward: {reward:.3f}")

            if terminated or truncated:
                break

    except Exception as e:
        print(f"Server connection failed (expected): {e}")
        print("Environment structure is ready for server integration!")

    env.close()