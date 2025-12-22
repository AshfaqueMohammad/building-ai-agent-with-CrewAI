from openai import OpenAI

def main():
    # print("Hello from building-ai-agents-with-crewai!")

    # client = OpenAI(
    #     base_url="http://localhost:11434/v1",  # Ollama OpenAI-compatible endpoint
    #     api_key="ollama",                      # dummy key, not from OpenAI
    # )

    # resp = client.chat.completions.create(
    #     model="gpt-oss:20b",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Explain quantization in one paragraph."},
    #     ],
    # )

    # print(resp.choices[0].message.content)

    from crewai import Agent, Task, Crew
    from crewai import LLM

    llm = LLM(
        model="gpt-oss:20b",
        base_url="http://localhost:11434/v1",  # Ollama OpenAI-compatible endpoint
        api_key="ollama",                      # dummy key, not from OpenAI
    )

    senior_technical_writer = Agent(

        role="Senior Technical Writer",
        
        goal="""Craft clear, engaging, and well-structured
                technical content based on research findings""",
        
        backstory="""You are an experienced technical writer
                    with expertise in simplifying complex
                    concepts, structuring content for readability,
                    and ensuring accuracy in documentation.""",
                    
        llm=llm,
                    
        verbose=True
    )

    writing_task = Task(
        description="""Write a well-structured, engaging,
                    and technically accurate article
                    on {topic}.""",
        
        agent=senior_technical_writer, 
        
        
        expected_output="""A polished, detailed, and easy-to-read
                        article on the given topic.""",
    )

    from crewai import Crew

    crew = Crew(
        agents=[senior_technical_writer],
        tasks=[writing_task],
        verbose=True
    )

    response = crew.kickoff(inputs={"topic":"AI Agents"})


if __name__ == "__main__":
    main()
