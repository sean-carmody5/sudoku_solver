import os
from flask import Flask, render_template, request, jsonify
from combined_solver

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
def solve(problem, manual_bool):
	# Here, you would call your Sudoku solving functions
	# You'll get the data from the request object
	data = request.get_json()
	# puzzle = data['puzzle']
	solve_sudoku(problem=problem, manual_input=manual_bool)

	# Call your Sudoku solving functions here and get the solution

	solution = []  # Replace with your Sudoku solution
	return jsonify(solution=solution)


if __name__ == '__main__':
	app.run(debug=True)
