from debug.state_utils import load_state

from workflow import (
    critic_node,
    revision_node,
)


state = load_state(
    "pre_critic_state.json"
)

# Artificially create a realistic budget failure
state["weekly_budget"] = 35

print("\n=== CRITIC ===")

critic_result = critic_node(state)

state.update(
    critic_result
)

if not state["approved"]:

    print("\n=== REVISION ===")

    revision_result = revision_node(
        state
    )

    state.update(
        revision_result
    )

    print("\n=== REVISED PLAN ===")

    for meal in state["weekly_plan"]:
        print(
            meal["meal_number"],
            meal["name"]
        )

    print(
        "New total:",
        state["estimated_total"]
    )

    # Run critic again
    print("\n=== RE-CHECKING REVISED PLAN ===")

    second_critic_result = critic_node(
        state
    )

    state.update(
        second_critic_result
    )

    print("\n=== FINAL RESULT ===")

    print(
        "Approved:",
        state["approved"]
    )

    print(
        "Feedback:",
        state["critic_feedback"]
    )