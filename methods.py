"""Toolbox controllers over the same frozen Awomo joint-action policy."""
import numpy as np

from harnessstudio.core.method import Budgets, Fidelity, MethodConfig, ModuleRef


def method(arm):
    if arm == "seated":
        return MethodConfig(
            name="robodojo-toolbox-seated",
            version="0.1.0",
            paper="none",
            fidelity=Fidelity.TF_VARIANT,
            substitutions=("Add RGB recess registration, directed transfer, guarded release, and bounded recovery.",),
            description="Camera-grounded grasp and recess placement over frozen Awomo",
            categories=("observation_grounding", "execution_monitoring", "skill_composition", "contact_feedback"),
            required=frozenset({"act"}),
            controller=ModuleRef("placement:SeatedGrasp"),
            null="shadow",
            budgets=Budgets(policy_calls=900, llm_calls=0),
            trainable_parameters=0,
        )
    if arm == "grounded":
        return MethodConfig(
            name="robodojo-toolbox-grounded",
            version="0.1.0",
            paper="none",
            fidelity=Fidelity.TF_VARIANT,
            substitutions=("Localize the requested tool in RGB and substitute a bounded grasp on wrong-arm motion.",),
            description="Camera-grounded approach and grasp with native Awomo placement",
            categories=("observation_grounding", "execution_monitoring", "skill_composition", "contact_feedback"),
            required=frozenset({"act"}),
            controller=ModuleRef("grounded:GroundedGrasp"),
            null="shadow",
            budgets=Budgets(policy_calls=900, llm_calls=0),
            trainable_parameters=0,
        )
    candidate = arm == "staged"
    return MethodConfig(
        name=f"robodojo-toolbox-{arm}",
        version="0.1.0",
        paper="none" if candidate else "Awomo-0.5I / XPolicyLab",
        fidelity=Fidelity.TF_VARIANT if candidate else Fidelity.COMPONENT,
        substitutions=("Replace the global instruction with one tool at a time.",) if candidate else (),
        description="Toolwise instructions with proprioceptive release detection" if candidate else "Native Awomo",
        categories=("language_alignment", "execution_monitoring") if candidate else (),
        required=frozenset({"act"}),
        controller=ModuleRef(f"methods:{'Staged' if candidate else 'Native'}"),
        null="shadow",
        budgets=Budgets(policy_calls=900, llm_calls=0),
        trainable_parameters=0,
    )


class Native:
    async def run(self, ctx):
        while not ctx.ended:
            await ctx.execute(await ctx.policy.act())


class Staged:
    """Use a close, move, release sequence as a phase signal, not a success label."""
    tools = ("hammer", "pliers", "tape measure", "wrench")
    grippers = (6, 13)
    arms = (slice(0, 6), slice(7, 13))
    max_phase_steps = 185

    async def run(self, ctx):
        for name in self.tools:
            prompt = f"Pick up the {name} and put it into its matching shaped slot in the toolbox."
            start = ctx.steps
            closed_at = [None, None]
            finished = False
            ctx.record("tool_phase_started", tool=name)
            while not ctx.ended and not finished and ctx.steps - start < self.max_phase_steps:
                chunk = await ctx.policy.act(instruction=prompt)
                for action in chunk:
                    await ctx.execute(action[None, :])
                    if ctx.ended:
                        return
                    state = ctx.observe()["state"]
                    for i, gripper in enumerate(self.grippers):
                        if state[gripper] < 0.35 and closed_at[i] is None:
                            closed_at[i] = state[self.arms[i]].copy()
                        elif state[gripper] > 0.8 and closed_at[i] is not None:
                            moved = np.linalg.norm(state[self.arms[i]] - closed_at[i])
                            closed_at[i] = None
                            if moved > 0.25 and ctx.steps - start >= 32:
                                finished = True
                    if finished or ctx.steps - start >= self.max_phase_steps:
                        break
            ctx.record("tool_phase_ended", tool=name, reason="release" if finished else "phase_budget")
        while not ctx.ended:
            prompt = "Return both robot arms to their starting positions with their grippers open."
            await ctx.execute(await ctx.policy.act(instruction=prompt))
