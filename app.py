import os

import replicate
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")
replicate_key = os.getenv("REPLICATE_API_TOKEN")

llm = OpenAI(temperature=0.5)


def generate_reciepe(food, calories, time):

    prompt = PromptTemplate(
        input_variables=["food", "calories", "time"],
        template="""
        You are an experienced chef, please write a recipe for {food}
        that has a maximum of {calories} calories
        and takes {time} minutes to prepare and cook.
        """
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    reciepe = llm_chain.run({
        "food": food,
        "calories": calories,
        "time": time
    })

    return reciepe


def generate_audio(text, voice):
    audio = generate(text=text, voice=voice, api_key=eleven_api_key)
    return audio


def generate_images(food):
    output = replicate.run(
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"prompt": food}
    )
    return output


def app():
    st.title("Recipe Generator!")

    with st.form(key='recipe_form'):
        food = st.text_input("What food do you want to cook?")
        calories = st.number_input("How many calories do you want to eat?")
        time = st.number_input(
            "How much time do you have to cook? (in minutes)")

        options = ["Bella", "Antoni", "Arnold", "Adam",
                   "Domi", "Elli", "Josh", "Rachel", "Sam"]
        voice = st.selectbox("Select a voice", options)

        submit_button = st.form_submit_button(label="Generate Recipe")

    if submit_button:
        recipe = generate_reciepe(food, calories, time)

        st.markdown(recipe)
        st.audio(generate_audio(recipe, voice))

        images = generate_images(food)
        st.image(images[0])


if __name__ == "__main__":
    app()
