from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information = """

    Leonardo da Vinci

    This portrait attributed to Francesco Melzi, c. 1515–1518, is the only certain contemporary depiction of Leonardo.[1][2]
    Born	Leonardo di ser Piero da Vinci
    15 April 1452
    possibly Anchiano,[a] Vinci, Florence
    Died	2 May 1519 (aged 67)
    Clos Lucé, Amboise, France
    Resting place	Château d'Amboise
    Education	Studio of Andrea del Verrocchio
    Years active	c. 1470–1519
    Known for	
    Paintingdrawingengineeringanatomical studieshydrologybotanyopticsgeology
    Notable work	
    Virgin of the Rocks (c. 1483–1493)
    Lady with an Ermine (c. 1489–1491)
    The Vitruvian Man (c. 1490)
    The Last Supper (c. 1495–1498)
    Mona Lisa (c. 1503–1516)
    Movement	High Renaissance
    Family	Da Vinci
    Signature
    Signature written in ink in a flowing script
    Leonardo di ser Piero da Vinci[b][c] (15 April 1452 – 2 May 1519) was an Italian polymath of the High Renaissance who was active as a painter, draughtsman, engineer, scientist, theorist, sculptor, and architect.[3] While his fame initially rested on his achievements as a painter, he has also become known for his notebooks, in which he made drawings and notes on a variety of subjects, including anatomy, astronomy, botany, cartography, painting, and palaeontology. Leonardo is widely regarded as a genius who epitomised the Renaissance humanist ideal,[4] and his collective works contributed to the development of European art to an extent rivalled only by that of his younger contemporary Michelangelo.[3][4]

    Born out of wedlock to a successful notary and a lower-class woman in, or near, Vinci, he was educated in Florence by the Italian painter and sculptor Andrea del Verrocchio. He began his career in the city, but then spent much time in the service of Ludovico Sforza in Milan. Later, he worked in Florence and Milan again, as well as briefly in Rome, all while attracting a large following of imitators and students. Upon the invitation of Francis I, he spent his last three years in France, where he died in 1519. Since his death, there has not been a time when his achievements, diverse interests, personal life, and empirical thinking have failed to incite interest and admiration,[3][4] making him a frequent namesake and subject in culture.
    """

    summary_template = """
    given the information {information} about a person , I want you to create:
    1. A short summary.
    2. two interesting facts about the person."""

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"],
        template = summary_template
    )

    llm = ChatOpenAI(model="gpt-5", temperature=0)
    # llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm
    
    response = chain.invoke(input={"information": information})
    print(response)
    print()
    print(response.content)



if __name__ == "__main__":
    main()
