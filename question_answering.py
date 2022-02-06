from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField
from transformers import pipeline

sequence = """ The traffic began to slow down on Pioneer Boulevard in Los Angeles, making it difficult
to get out of the city. However, WBGO was playing some cool jazz, and the weather was cool, making it
rather pleasant to be making it out of the city on this Friday afternoon. Nat King Cole was singing as
Jo and Maria slowly made their way out of LA and drove toward Barstow. They planned to get to Las Vegas
early enough in the evening to have a nice dinner and go see a show. """

# sequence = "The traffic began to slow down on Pioneer Boulevard in Los Angeles, making it difficult to get out of the city. However, WBGO was playing some cool jazz, and the weather was cool, making it rather pleasant to be making it out of the city on this Friday afternoon. Nat King Cole was singing as Jo and Maria slowly made their way out of LA and drove toward Barstow. They planned to get to Las Vegas early enough in the evening to have a nice dinner and go see a show."
# question = 'Where is Pioneer Boulevard ?'
question = 'What is playing?'

class Form(FlaskForm):
    text = TextAreaField(render_kw={"rows": 8})
    question = StringField(label='What\'s your question?:', description='sdf')

def question_answering_route():
    form = Form()
    answer = {"answer": "N/A", "score": "N/A"}
    if form.validate_on_submit():
        nlp_qa = pipeline('question-answering')
        answer = nlp_qa(context=form.text.data or sequence, question=form.question.data or question)
        return redirect(url_for('question_answering', form=form, answer=answer["answer"]))

    return render_template('question-answering.html', form=form, answer=answer)

def answer_question():
    args = request.args
    nlp_qa = pipeline('question-answering')
    answer = nlp_qa(context=args["text"] or sequence, question=args["question"] or question)
    
    return answer