from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import Form, TextAreaField, validators
from werkzeug.utils import secure_filename
import pickle
import sqlite3
import os
import numpy as np

import os
# import sys
SECRET_KEY = os.urandom(32)

# print("Python version")
# print (sys.version)
# print("Version info.")
# print (sys.version_info)

import face_tracking
from face_tracking import process_video

# import HashingVectorizer from local dir
from vectorizer import vect

# import update function from local dir
from update import update_model

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = SECRET_KEY

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
	label = {0: 'negative', 1: 'positive'}
	X = vect.transform([document])
	y = clf.predict(X)[0]
	proba = np.max(clf.predict_proba(X))
	return label[y], proba

def train(document, y):
	X = vect.transform([document])
	clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
	conn = sqlite3.connect(path)
	c = conn.cursor()
	c.execute("INSERT INTO review_db (review, sentiment, date)"\
	" VALUES (?, ?, DATETIME('now'))", (document, y))
	conn.commit()
	conn.close()

######## Flask
class ReviewForm(Form):
	moviereview = TextAreaField('',
		[validators.DataRequired(),
		validators.length(min=15)])

@app.route('/')
def index():
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
	form = ReviewForm(request.form)
	if request.method == 'POST' and form.validate():
		review = request.form['moviereview']
		y, proba = classify(review)
		return render_template('results.html',
				content=review,
				prediction=y,
				probability=round(proba*100, 2))
	return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
	feedback = request.form['feedback_button']
	review = request.form['review']
	prediction = request.form['prediction']

	inv_label = {'negative': 0, 'positive': 1}
	y = inv_label[prediction]
	if feedback == 'Incorrect':
		y = int(not(y))
	train(review, y)
	sqlite_entry(db, review, y)
	return render_template('thanks.html')

# class UploadForm(Form):
# 	video = FileField('Video File', [validators.regexp('^\[^/\\]\.mp4$')])

class UploadForm(FlaskForm):
    file = FileField()

@app.route('/face-tracking', methods=['GET', 'POST'])
def face_tracking():
	form = UploadForm()
 
	if form.validate_on_submit():
		print(os.getcwd())
		filename = secure_filename(form.file.data.filename)
		# movieclassifier_new is the name of the root project directory on the hosting
		try:
			form.file.data.save(f'{os.getcwd()}/movieclassifier_new/uploads/' + filename)
		except:
			form.file.data.save(f'{os.getcwd()}/uploads/' + filename)
		process_video(filename=filename)
		# return redirect(url_for('face_tracking'))
		return render_template('face-tracking.html', processed_video_link=f'movieclassifier_new/static/video_tracked.mp4', form=form)
	
	return render_template('face-tracking.html', form=form)

# @app.route('/face-tracking-results', methods=['GET', 'POST'])
# def face_tracking_results():
	# form = UploadForm(request.form)
	# form = UploadForm()
	# if request.method == 'POST' and form.validate():
		# print(request)
		# print(form)
		# review = request.form['moviereview']
		# y, proba = classify(review)
		# return render_template('results.html',
		# 		content=review,
		# 		prediction=y,
		# 		probability=round(proba*100, 2))
	# return render_template('reviewform.html', form=form)
	
	# if form.validate_on_submit():
	# 	filename = secure_filename(form.file.data.filename)
	# 	form.file.data.save('uploads/' + filename)
	# 	return redirect(url_for('upload'))
    
	# return render_template('face-tracking-results.html', request=request)

if __name__ == '__main__':
	clf = update_model(db_path=db, model=clf, batch_size=10000)
	app.run(debug=True)
