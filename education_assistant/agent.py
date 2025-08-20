import json

from crewai import Agent, Task, Crew, LLM
from crewai.memory import ShortTermMemory
from crewai.tools import BaseTool
from crewai.process import Process
from typing import Type, Any
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
from mem0 import MemoryClient
from dotenv import load_dotenv
from pydantic import BaseModel
from streamlit.runtime.state.common import user_key_from_element_id


# Load environment variables
load_dotenv()


# Schemas
class QuizOutput(BaseModel):
    """
    A model representing the output of a quiz.
    """
    title: str | None = None
    quiz_questions: list[str | Any]
    answers: list[str | Any]


class LearningMaterialOutput(BaseModel):
    """
    A model representing the output of learning materials.
    """
    user_interests: list[str]
    materials: list[str]


class UserProjectInput(BaseModel):
    """
    A model representing a project suggestion based on user interests and current educational trends.
    """
    user_interests: list[str]
    materials: list[Any]
    quiz: Any


class ProjectIdeaOutput(BaseModel):
    """
    A model representing the output of a project idea.
    """
    topic: str
    project_title: str
    project_description: str


# Tools
class ProjectSuggestionTool(BaseTool):
    """
    A tool that suggests a project based on the user's interests and current educational trends.
    """
    name : str = "Project Suggestion Tool"
    description : str  = "Suggests a project based on the user's interests and current educational trends."
    args_schema : Type[BaseModel] = UserProjectInput

    def _run(self, user_interests: list[str], materials: list[Any], quiz: QuizOutput) -> str:
        response = project_idea_llm.call(
            f"Based on the user's interests in {user_interests} "
            f"and the current educational materials {materials}, "
            f"suggest a project that aligns with these interests and trends."
        )
        with open("educational_planning.json", "w") as file_handle:
            data_dict = {
                "user_interests": user_interests,
                "materials": materials,
                "quiz": quiz,
                "project_suggestion": response
            }
            json.dump(data_dict, file_handle, indent=4)
        return response


# Agents

learning_material_llm = LLM(
    model="gpt-4o",
    temperature=0.1,
    max_tokens=15_000,
)
learning_material_agent = Agent(
    role="Learning Material Specialist",
    goal="Curates learning materials based on the users' topics of interest: {user_interests}.",
    backstory="An expert in educational content development with a focus on curriculum design and learner engagement.",
    tools=[SerperDevTool()],
    verbose=True,
    llm=learning_material_llm
)

quiz_creator_llm = LLM(
    model="gpt-4o-mini",
    temperature=0.5,
    max_tokens=10_000,
)
quiz_creator_agent = Agent(
    role="Quiz Creator",
    goal="Creates quizzes to assess understanding of the learning materials.",
    backstory="A specialist in educational assessment with a knack for creating engaging quizzes that reinforce learning.",
    verbose=True,
    llm=quiz_creator_llm
)

project_idea_llm = LLM(
    model="gpt-4o-mini",
    temperature=0.75,
    max_tokens=8_000,
)
project_idea_agent = Agent(
    role="Project Idea Recommender",
    goal="Suggests a project based on the user's interests and current educational trends.",
    backstory="An educational innovator who connects learners with practical projects that enhance their understanding of key concepts.",
    tools=[ProjectSuggestionTool()],
    verbose=True,
    llm=project_idea_llm
)


# Tasks

generate_learning_materials = Task(
    description="Generate learning materials based on the users' interest: {user_interests}.",
    expected_output="A list of curated learning materials that cover the specified topics.",
    agent=learning_material_agent,
    output_pydantic=LearningMaterialOutput
)

create_quizes = Task(
    description="Create quizzes to assess understanding of the learning materials. "
                "Return a title for the quiz, a list of quiz questions, and the corresponding answers.",
    expected_output="A quiz with a title, questions, and answers that test the user's knowledge of the learning materials.",
    output_pydantic=QuizOutput,
    agent=quiz_creator_agent,
)

suggest_project_ideas = Task(
    description="Suggest a project based on the user interest {user_interests}; and current educational materials.",
    expected_output="A project topic, title and description that aligns with the user's interests and educational materials.",
    agent=project_idea_agent,
    output_pydantic=ProjectIdeaOutput,
    context=[generate_learning_materials, create_quizes]
)


# Crew
mem0_client_config ={
        'provider': 'mem0',
        'config': {'user_id': 'jenny'}
    }

educational_assistance_crew = Crew(
    name="Educational Assistance Crew",
    agents=[learning_material_agent, quiz_creator_agent, project_idea_agent],
    tasks=[generate_learning_materials, create_quizes, suggest_project_ideas],
    process=Process.sequential,
    planning = True,
    verbose = True,
    memory=True,
    memory_config=mem0_client_config
)

short_term_memory_mem0_client = ShortTermMemory(crew=educational_assistance_crew)

def run_crew():
    """
    Run the educational assistance crew.
    """
    try:
        print("\nWelcome to the Educational Assistance Crew!\n")
        user_inputs = input("Enter your topics of interests separated by `,`: ")
        interests = user_inputs.split(",")
        educational_assistance_crew.kickoff({"user_interests": interests})
    except Exception as e:
        raise Exception(f"An error occurred while running the educational assistance crew: {e}")


if __name__ == "__main__":
    run_crew()
