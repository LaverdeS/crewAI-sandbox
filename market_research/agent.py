from crewai import Agent, Task, Crew
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
from dotenv import load_dotenv


load_dotenv()

product_name= "AI Apps and/or social platform for psychological assistance for loneliness."
strategist_backstory = "and marketing strategy."
market_researcher = Agent(
    role="Market Researcher",
    goal=f"Analyse market trends for the product launch.",
    backstory="",
    tools=[SerperDevTool()],
    verbose= True,
)

strategist = Agent(
    role="Product Strategist",
    goal=f"Create effective positioning strategies for the product.",
    backstory=f"Skilled in competitive positioning {strategist_backstory}.",
    verbose= True,
)

gather_market_insights_task = Task(
    description=f"Browse the internet to gather insights on current market trends for the launch of the '{product_name}' product.",
    expected_output=f"List of relevant market trends and consumer preferences, relevant to the {product_name}.",
    agent=market_researcher,
)

develop_positioning_strategy_task = Task(
    description=f"Based on market insights, create a positioning strategy for the {product_name}.",
    expected_output=f"A positioning strategy with target audience and impact notes.",
    agent=strategist,
)

crew = Crew(
    agents=[market_researcher, strategist],
    tasks=[gather_market_insights_task, develop_positioning_strategy_task],
    planning=True,
)

crew.kickoff()