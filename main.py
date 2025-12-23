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

    senior_executive_coach = Agent(

        role="Senior Job interview Coach",
        
        goal="""Prepare a candidate for CTO interview for a given {company}
                by providing detailed insights,
                potential questions, and strategies, showcase
                their leadership skills and technical expertise.""",

        # backstory="""You are an experienced technical writer
        #             with expertise in simplifying complex
        #             concepts, structuring content for readability,
        #             and ensuring accuracy in documentation.""",

        backstory="""You are an experienced executive coach with expertise in preparing chief technical officers for interviews.
                    You excel at providing insights into leadership skills, technical expertise, and strategic thinking required for CTO roles. 
                    And also have good acumen for a given {company}. You have the ability to know company's tech stack, their products, their customers and their leadership team""",
        llm=llm,
                    
        verbose=True
    )

    comapny_research_agent = Agent(
        role="Company Research Specialist",
        
        goal="""Conduct in-depth research on {company}
                to gather insights about its products, services,
                market position, competitors, and recent developments in the context of a CTO job interview preparation.""",
        
        backstory="""You are a skilled researcher with expertise in gathering
                    and analyzing information about companies.
                    You excel at using various sources to compile comprehensive
                    profiles that highlight key aspects about {company} to prepare for a CTO job interview.""",
        llm=llm,
                    
        verbose=True
    )

    interview_coaching_task = Task(
        description="""You provide interview preparation insights.""",
        
        agent=senior_executive_coach, 
        
        
        expected_output="""A detailed interview preparation material that is easy-to-read, structured by topics
                        in a bulleted format.""",
    )

    company_research_task = Task(
        description="""You should conduct an in-depth research on {company} for CTO job interview preparation.""",
        agent=comapny_research_agent,
        expected_output="""A comprehensive research report on {company}, 
        including insights about its company's operations, company's financials 
        and the company leadership team, products, services, market position, 
        competitors, and recent developments.""",
    )

    from crewai import Crew

    crew = Crew(
        agents=[comapny_research_agent],
        tasks=[company_research_task],
        verbose=True
    )

    response = crew.kickoff(inputs={"company":"MSCI Inc."})


if __name__ == "__main__":
    main()
