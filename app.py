from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response, abort
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import Form, TextAreaField, BooleanField, validators
from werkzeug.utils import secure_filename
import pickle
import sqlite3
import os
import numpy as np

import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)


SECRET_KEY = os.urandom(32)

import question_answering
from question_answering import question_answering_route, answer_question

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
    areEmotionsTracked = BooleanField()

root_dir = os.getcwd()
if (os.path.isdir(f'{os.getcwd()}/movieclassifier_new')):
	root_dir = f'{os.getcwd()}/movieclassifier_new'


@app.route('/face-tracking', methods=['GET', 'POST'])
def face_tracking():
	print(root_dir)
	form = UploadForm()
	
	tracked_dir_files = os.listdir(f'{root_dir}/static/tracked')
	tracked_dir_paths = tracked_dir_files
	untracked_dir_files = os.listdir(f'{root_dir}/static/untracked')
	untracked_dir_paths = untracked_dir_files

	if form.validate_on_submit():
		filename = secure_filename(form.file.data.filename)
		# movieclassifier_new is the name of the root project directory on the hosting
		form.file.data.save(f'{root_dir}/static/untracked/' + filename)
		are_emotions_tracked = form.areEmotionsTracked.data
		
		process_video(filename=filename, are_emotions_tracked=are_emotions_tracked)
		return redirect(url_for('face_tracking', form=form, tracked_dir_paths=tracked_dir_paths, untracked_dir_paths=untracked_dir_paths))
	
	return render_template('face-tracking.html', form=form, tracked_dir_paths=tracked_dir_paths, untracked_dir_paths=untracked_dir_paths)

@app.route('/delete-video/<filename>', methods=['DELETE'])
def delete_video(filename):
	folder, filename = filename.split(':::')
	full_folder_path = f'{root_dir}/static/tracked' if folder == 'tracked' else f'{root_dir}/static/untracked'

	try:
		os.remove(f'{full_folder_path}/{filename}')
		print('trying')
		response = make_response(
			jsonify(
				{"message": 'File deleted'}
			),
			200
		)
		response.headers["Content-Type"] = "application/json"

		return response
	except Exception as e:
		print('errored')
		print(e)
		response = make_response(
			jsonify(
				{"message": 'Error when deleting file'}
			),
			400,
		)
		response.headers["Content-Type"] = "application/json"
		abort(400)
		# return response
		# return Response(status_code=400)

	# return 'ok'

@app.route('/question-answering', methods=['GET', 'POST'])
def question_answering():
	return question_answering_route()

@app.route('/answer', methods=['GET', 'POST'])
def answer():
	return answer_question()

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
