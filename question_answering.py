import tensorflow
from transformers import pipeline
print(tensorflow.__version__)
nlp_qa = pipeline('question-answering')

# sequence = """ The traffic began to slow down on Pioneer Boulevard in Los Angeles, making it difficult
# to get out of the city. However, WBGO was playing some cool jazz, and the weather was cool, making it
# rather pleasant to be making it out of the city on this Friday afternoon. Nat King Cole was singing as
# Jo and Maria slowly made their way out of LA and drove toward Barstow. They planned to get to Las Vegas
# early enough in the evening to have a nice dinner and go see a show. """

sequence = "The traffic began to slow down on Pioneer Boulevard in Los Angeles, making it difficult to get out of the city. However, WBGO was playing some cool jazz, and the weather was cool, making it rather pleasant to be making it out of the city on this Friday afternoon. Nat King Cole was singing as Jo and Maria slowly made their way out of LA and drove toward Barstow. They planned to get to Las Vegas early enough in the evening to have a nice dinner and go see a show."
# question = 'Where is Pioneer Boulevard ?'
question = 'What is playing?'
print(nlp_qa(context=sequence, question=question))
