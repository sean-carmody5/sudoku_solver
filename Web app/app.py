import os
from flask import Flask, render_template, request, jsonify
from combined_solver import solve_sudoku_upload, solve_sudoku_manual_input
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
	os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
	if 'file' not in request.files:
		return jsonify(success=False, error='No file part')
	file = request.files['file']
	if file.filename == '':
		return jsonify(success=False, error='No selected file')
	if not allowed_file(file.filename):
		return jsonify(success=False, error='File type not allowed')
	filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
	file.save(filepath)
	return jsonify(success=True, message='File uploaded successfully')

def allowed_file(filename):
	ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/solve', methods=['POST'])
def solve():
	# Assume the board is sent as a JSON array and manual_input as a boolean
	data = request.get_json()
	board = np.array(data['board'])
	manual_input = data['manual_input']

	# Call the solve_sudoku function
	if not manual_input:
		solved_board = solve_sudoku_upload()
	else:
		solved_board = solve_sudoku_manual_input(board)

	# Convert the numpy array to a JSON array and return it
	solved_board_list = solved_board.tolist()
	print(solved_board_list)
	return jsonify(solved_board=solved_board_list)


if __name__ == '__main__':
	app.run(debug=True)
