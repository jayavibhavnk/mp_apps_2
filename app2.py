## quiz
import streamlit as st
import json
from functools import lru_cache

def run():
    st.set_page_config(
        page_title="GRAEL QUIZ APP",
        page_icon="❓",
    )

if __name__ == "__main__":
    run()

# Custom CSS for the buttons
st.markdown("""
<style>
div.stButton > button:first-child {
    display: block;
    margin: 0 auto;
</style>
""", unsafe_allow_html=True)

# Initialize session variables if they do not exist
default_values = {'current_index': 0, 'current_question': 0, 'score': 0, 'selected_option': None, 'answer_submitted': False, 'quiz_data': None}
for key, value in default_values.items():
    st.session_state.setdefault(key, value)

def restart_quiz():
    st.session_state.current_index = 0
    st.session_state.score = 0
    st.session_state.selected_option = None
    st.session_state.answer_submitted = False
    st.session_state.quiz_data = None  # Reset quiz_data to None

def submit_answer():
    # Check if an option has been selected
    if st.session_state.selected_option is not None:
        # Mark the answer as submitted
        st.session_state.answer_submitted = True
        # Check if the selected option is correct
        if st.session_state.selected_option == st.session_state.quiz_data[st.session_state.current_index]['answer']:
            st.session_state.score += 10
    else:
        # If no option selected, show a message and do not mark as submitted
        st.warning("Please select an option before submitting.")

def next_question():
    st.session_state.current_index += 1
    st.session_state.selected_option = None
    st.session_state.answer_submitted = False

import os
from openai import OpenAI
client = OpenAI()

@lru_cache(maxsize=1)  # Cache the results of query_ai function
def query_ai(query):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": query}
        ],
        n=1
    )

    return completion.choices[0].message.content

prompt_template = """ you are an expert at generating quizzes

you will be given large text on which youll have to generate questions

you will strictly follow this format
(list of jsons)
you have to always return a list of jsons
{
        "question": "question",
        "information": "explanation of the answer",
        "options": [4 options],
        "answer": "actual answer"
    },

example:
[
    {
        "question": "Which clubs founded the Premier League by breaking away from the English Football League?",
        "information": "The competition was founded as the FA Premier League by clubs including Manchester United, Liverpool, Tottenham Hotspur, Everton, and Arsenal.",
        "options": ["Liverpool, Tottenham Hotspur, Chelsea, Manchester City", "Arsenal, Southampton, Newcastle United, Leicester City", "Manchester United, Chelsea, Aston Villa, West Ham United", "Manchester City, Manchester United, Crystal Palace, Newcastle United"],
        "answer": "Liverpool, Tottenham Hotspur, Everton, Arsenal, Manchester United"
    },
    {
        "question": "Which television companies secured the domestic rights for broadcasting Premier League games in a £5 billion deal?",
        "information": "Sky and BT Group secured the domestic rights to broadcast 128 and 32 games, respectively, in a £5 billion deal.",
        "options": ["BBC and ITV", "Sky and FX", "ESPN and Fox Sports", "Sky and BT Group"],
        "answer": "Sky and BT Group"
    }
]

the output has to be in json format only

generate 5 questions on the topic

topic:
"""

# Title and description
st.title("Quizzy")

st.session_state.text = st.text_area("Enter Content:")
st.session_state.b = st.button("Submit!")

# Load quiz data
if st.session_state.b:
    k = query_ai(prompt_template + st.session_state.text)
    import json
    new_k = k[k.find("["):k.rfind("]") + 1]
    # if new_k:
    #     st.write(new_k)
    st.session_state.quiz_data = json.loads(new_k)

# Check if quiz_data is available
if st.session_state.quiz_data:
    # Progress bar
    progress_bar_value = (st.session_state.current_index + 1) / len(st.session_state.quiz_data)
    st.metric(label="Score", value=f"{st.session_state.score} / {len(st.session_state.quiz_data) * 10}")
    st.progress(progress_bar_value)

    # Display the question and answer options
    question_item = st.session_state.quiz_data[st.session_state.current_index]
    st.subheader(f"Question {st.session_state.current_index + 1}")
    st.title(f"{question_item['question']}")
    # st.write(question_item['information'])

    st.markdown("""___""")

    # Answer selection
    options = question_item['options']
    correct_answer = question_item['answer']

    if st.session_state.answer_submitted:
        for i, option in enumerate(options):
            label = option
            if option == correct_answer:
                st.success(f"{label} (Correct answer)")
            elif option == st.session_state.selected_option:
                st.error(f"{label} (Incorrect answer)")
            else:
                st.write(label)
            # st.write(question_item['information'])
    else:
        for option in options:
            if st.button(option, key=f"option_{option}", use_container_width=True):
                st.session_state.selected_option = option

    st.write(question_item['information'])
    st.markdown("""___""")

    # Submission button and response logic
    if st.session_state.answer_submitted:
        if st.session_state.current_index < len(st.session_state.quiz_data) - 1:
            st.button('Next', on_click=next_question)
        else:
            st.write(f"Quiz completed! Your score is: {st.session_state.score} / {len(st.session_state.quiz_data) * 10}")
            if st.button('Restart', on_click=restart_quiz):
                pass
    else:
        if st.session_state.current_index < len(st.session_state.quiz_data):
            st.button('Submit', on_click=submit_answer)
