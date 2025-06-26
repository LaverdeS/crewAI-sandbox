from crewai import Agent, Task, Crew
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
from dotenv import load_dotenv

load_dotenv()


products_to_optimize = ["medical supplies", "consumer electronics", "grocery perishables"]


logistics_analyst = Agent(
    role="Logistics Analyst",
    goal="Research the current state of logistics operations for key products.",
    backstory="An experienced logistics professional focused on supply chain efficiency, with a deep understanding of route planning and inventory turnover trends.",
    tools=[SerperDevTool()],  # Useful for researching current logistics data
    verbose=True,
)

optimization_strategist = Agent(
    role="Optimization Strategist",
    goal="Design data-driven logistics optimization strategies.",
    backstory="An expert in supply chain optimization and strategic planning, with a track record of improving delivery systems and inventory turnover.",
    verbose=True,
)


analyze_logistics_task = Task(
    description=(
        f"Using available data and research tools, analyze the current logistics operations for the following product categories: "
        f"{', '.join(products_to_optimize)}. Focus on route efficiency and inventory turnover trends. "
        f"Summarize your insights in a structured format."
    ),
    expected_output="A concise analysis of logistics challenges and performance for each product category, highlighting inefficiencies or opportunities.",
    agent=logistics_analyst,
)

generate_optimization_strategy_task = Task(
    description=(
        f"Based on the logistics insights provided by the analyst, propose an optimization strategy for the product categories: "
        f"{', '.join(products_to_optimize)}. The strategy should include suggestions for improving delivery routes and inventory turnover rates."
    ),
    expected_output="A detailed optimization plan with specific actions and justifications, tailored for each product category.",
    agent=optimization_strategist,
    context=[analyze_logistics_task],  # 👈 Forces sequential dependency
)


crew = Crew(
    agents=[logistics_analyst, optimization_strategist],
    tasks=[analyze_logistics_task, generate_optimization_strategy_task],
    planning=True,  # Enables the system to plan task execution
)


crew.kickoff()
