from debug.state_utils import load_state
from workflow import critic_node


state = load_state(
    "pre_critic_state.json"
)

print("\n=== TESTING CRITIC ONLY ===")

result = critic_node(state)

print("\n=== CRITIC RETURN VALUE ===")
print(result)