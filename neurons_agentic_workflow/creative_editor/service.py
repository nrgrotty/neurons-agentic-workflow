from pydantic_ai import Agent, RunContext
from neurons_agentic_workflow.creative_editor.models import CreativeFeedback, SubTask

planner_agent = Agent(
    "google-gla:gemini-3-pro-preview",
    deps_type=CreativeFeedback,
    output_type=list[SubTask],
    output_retries=3,
    system_prompt="Given the CreativeFeedback, create actionable SubTasks for editing the creative image.",
)


@planner_agent.system_prompt
async def add_create_feedback(ctx: RunContext[CreativeFeedback]) -> str:
    data = ctx.deps
    return f"Creative image feedback:\n{data.model_dump_json(indent=2)}"


async def plan_editions(creative_feedback: CreativeFeedback) -> list[SubTask]:
    result = await planner_agent.run(
        "Apply the creative feedback.", deps=creative_feedback
    )
    print("##### Planning result:", result)
    return result.output
