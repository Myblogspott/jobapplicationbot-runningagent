import asyncio
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

async def main() -> None:

    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    web_surfer = MultimodalWebSurfer("web_surfer", model_client, headless=False, animate_actions=True)
    

    user_proxy = UserProxyAgent("user_proxy")
    

    termination = TextMentionTermination("exit", sources=["user_proxy"])
    

    team = RoundRobinGroupChat([web_surfer, user_proxy], termination_condition=termination)

    task_instruction = (
        "You are a job scraping assistant. Your task is to navigate to a job portal website, "
        "for example, 'https://www.amazon.jobs/en/'. "
        "First, ask the user which job role they want to apply for. "
        "Then, analyze the webpage and identify the placeholder text for the job search input field. "
        "Finally, provide a short summary of the placeholder and instructions on how to proceed."
    )
    
    try:

        await Console(team.run_stream(task=task_instruction))
    finally:

        await web_surfer.close()
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())