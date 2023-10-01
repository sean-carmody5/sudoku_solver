"""
Script for solving a sudoku puzzle given a sudoku image/ manual input
Solves any square sudoku
possible array sizes: 4, 9, 16

Logic:
1) Run through array until we hit an empty element.
2) Check all possible values that can be placed in this element. Call this poss_values
3) Check rest of row, column and block to see what values can be placed elsewhere. Call these poss_values_elsewhere_row, _col and _block
4) If there is a value in poss_values that is not contained in any single one of these, populate the empty element with this value.

"""
import os
import io
import sys

import easyocr
import keras_ocr
import numpy as np
import cv2
import math
import time
import pytesseract
from tqdm import tqdm
from collections import Counter
import tkinter as tk
import contextlib
import concurrent.futures
import pygame
from moviepy.editor import *

POSSIBLE_CHARACTERS = list(map(str, range(0, 10))) + [chr(x) for x in (range(ord('A'), ord('G')))]
CUSTOM_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEF'
THRESHOLD = 50   # threshold for white pixels - indicating a character present in a box
reader = easyocr.Reader(['en'], gpu=False)
reader2 = keras_ocr.recognition.Recognizer()


def play_video_on_loop(video_file, frame_delay_ms=20, scale_factor=0.5):
	cap = cv2.VideoCapture(video_file)

	if not cap.isOpened():
		print("Error opening the video file.")
		exit()

	video = VideoFileClip(video_file)
	audio = video.audio
	audio.write_audiofile("audio.mp3")
	# Load the audio track using pygame
	pygame.mixer.init()
	pygame.mixer.music.load("audio.mp3")

	video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	current_frame = 0

	pygame.mixer.music.play()

	while True:
		ret, frame = cap.read()

		# Restart the video and audio when it reaches the end
		if not ret:
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			pygame.mixer.music.stop()
			pygame.mixer.music.play()
			current_frame = 0
			continue

		# Resize the frame
		frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

		cv2.imshow('Video', frame_resized)

		# Break the loop if a key is pressed, and add a delay based on the frame rate
		if cv2.waitKey(frame_delay_ms) & 0xFF != 0xFF:
			break

	cap.release()
	cv2.destroyAllWindows()

	# Stop the audio playback
	pygame.mixer.music.stop()


def max_time(grid_size):
	if grid_size == 16:
		return 6
	elif grid_size == 9:
		return 2
	else:
		return 1


def check_row(row, characters):
	# return all possible characters not in row
	return [x for x in characters if x not in list(np.unique(row))]


def check_column(column, characters):
	return [x for x in characters if x not in list(np.unique(column))]


def check_block(block, characters):
	return [x for x in characters if x not in list(np.unique(block))]


def check_possible_values(poss_in_row, poss_in_column, poss_in_block):
	return [x for x in poss_in_row if x in poss_in_column and x in poss_in_block]


def decide_block(x, y, block_size):
	# return end of slice parameters
	# make sure upper bound is returned
	if x % block_size == 0:
		x += 1
	if y % block_size == 0:
		y += 1
	block_row_edge = math.ceil(x/block_size)*block_size
	block_col_edge = math.ceil(y/block_size)*block_size

	return block_row_edge, block_col_edge


def return_poss_for_element(problem, x, y, characters, block_size):
	row = problem[x]
	possible_in_row = check_row(row, characters)
	possible_in_col = check_column(problem[:, y], characters)
	block_row_edge, block_col_edge = decide_block(x, y, block_size)
	block = problem[block_row_edge - block_size: block_row_edge, block_col_edge - block_size: block_col_edge]
	possible_in_block = check_block(block, characters)
	possible_values = check_possible_values(possible_in_row, possible_in_col, possible_in_block)

	return possible_values


def cross_check_row(problem, x, y, characters, block_size):
	# return all the possible numbers that can fit in the rest of the row
	unique_values = []
	for j, col in enumerate(problem[x]):
		if j != y and problem[x, j] == ' ':
			possible_values = return_poss_for_element(problem, x, j,  characters, block_size)
			unique_values = list(np.unique(unique_values + possible_values))

	return unique_values


def cross_check_column(problem, x, y, characters, block_size):
	# return all the possible numbers that can fit in the rest of the column
	unique_values = []
	for i, element in enumerate(problem[:, y]):
		if i != x and element == ' ':
			possible_values = return_poss_for_element(problem, i, y, characters, block_size)
			unique_values = list(np.unique(unique_values + possible_values))

	return unique_values


def cross_check_block(problem, x, y, characters, block_size):
	block_row_edge, block_col_edge = decide_block(x, y, block_size)
	unique_values = []
	for i in range(block_row_edge - block_size, block_row_edge):
		for j in range(block_col_edge - block_size, block_col_edge):
			# skip current element in the block
			if not(i == x and j == y) and problem[i, j] == ' ':
				possible_values = return_poss_for_element(problem, i, j, characters, block_size)
				unique_values = list(np.unique(unique_values + possible_values))

	return unique_values


def check_naked_single(problem, x, y, characters, block_size):
	# check for case that all characters except one are in the same row, column and block
	# note: see screenshot in folder for logic not included...
	row = list(problem[x])
	column = list(problem[:, y])
	block_row_edge, block_col_edge = decide_block(x, y, block_size)
	block = list(problem[block_row_edge - block_size: block_row_edge, block_col_edge - block_size: block_col_edge].flatten())

	# print(f'row: {row}')
	# print(f'column: {column}')
	# print(f'block: {block}')
	combined = row + column + block

	# print(combined)

	combined_unique = np.unique(combined)

	# print(combined_unique)

	return [_ for _ in characters if _ not in combined_unique]


def solve(problem, characters, block_size):
	counter = 0
	for x, row in enumerate(problem):
		for y, element in enumerate(row):
			if element == ' ':
				possible_values = return_poss_for_element(problem, x, y, characters, block_size)
				# print(possible_values)
				# now run through all possible values
				# if for any one of them, it cannot fit in another place
				# in any of the row, column or block, we place it in the current location
				possible_elsewhere_in_row = cross_check_row(problem, x, y, characters, block_size)
				possible_elsewhere_in_col = cross_check_column(problem, x, y, characters, block_size)
				possible_elsewhere_in_block = cross_check_block(problem, x, y, characters, block_size)

				# check which element of possible_values are not in the other lists
				other_lists = [possible_elsewhere_in_row, possible_elsewhere_in_col, possible_elsewhere_in_block]
				# print(other_lists)

				# Check if an element in possible_values is not in one or more of the other lists
				not_in_other_lists = np.array([~np.isin(possible_values, other_list) for other_list in other_lists])

				index = np.argwhere(np.any(not_in_other_lists, axis=0))
				list_number = np.argwhere(np.any(not_in_other_lists, axis=1))
				list_names = ["row", "column", "block"]

				value = check_naked_single(problem, x, y, characters, block_size)

				# if there is a value that works, use it
				# print(index)
				if len(index):
					print(f"Adding value {possible_values[index[0, 0]]} to space (row: {x + 1}, column: {y + 1})...")
					print(f"Value cannot fit elsewhere in the current {list_names[list_number[0, 0]]}")
					problem[x, y] = possible_values[index[0, 0]]
					# pprint(problem)
					counter += 1
					continue

				if len(value) == 1:
					print(f"Adding value {value[0]} to space (row: {x + 1}, column: {y + 1})...")
					print(f"Value is the only unique value remaining in the row, column and block")
					problem[x, y] = value[0]
					counter += 1

	print(f"Filled {counter} values...\n")
	return problem

	# ---------------------------------------------------------------------------------------------------------------


def line_intersection(h_line, v_line):
	y = h_line[0][1]
	x = v_line[0][0]
	return x, y


def filter_spatial_outliers(image, lines, line_orientation, tolerance=15):
	sorted_lines = sorted(lines, key=lambda x: x[0][1] if line_orientation == 'h' else x[0][0])
	sorted_values = [line[0][1] if line_orientation == 'h' else line[0][0] for line in sorted_lines]
	distances = [abs(line[0][1] - prev_line[0][1]) if line_orientation == 'h' else abs(line[0][0] - prev_line[0][0]) for prev_line, line in zip(sorted_lines, sorted_lines[1:])]

	median = np.median(distances)

	lines_to_filter = []

	for i in range(len(distances)):
		if distances[i] < median - tolerance:
			line1 = sorted_values[i]
			line2 = sorted_values[i + 1]
			next_line = sorted_values[i + 2] if i + 2 < len(sorted_lines) else sorted_values[i - 1]

			# append line1 or line2 to filtered lines depending on which one is closer to the median away from dist_next
			append_line = sorted_lines[i + 1] if abs(abs(line1 - next_line) - median) < abs(abs(line2 - next_line) - median) else sorted_lines[i]
			lines_to_filter.append(tuple(append_line[0]))
			# print("Got one!!", append_line, line1, line2, next_line)

	filtered_lines = []
	# extend length of lines
	for line in sorted_lines:
		line_tuple = tuple(line[0])  # Convert the numpy array to a tuple
		if line_tuple not in lines_to_filter:
			# print(line)
			if line_orientation == 'h':
				line[0][0] = 0
				line[0][2] = image.shape[1]
			else:
				line[0][1] = 0
				line[0][3] = image.shape[0]
			filtered_lines.append(line)

	return filtered_lines


def filter_duplicate_lines(lines, orientation, delta_rho=10, delta_theta=np.pi/90, tolerance=6):
	filtered_lines = []

	for line in lines:
		x1, y1, x2, y2 = line[0]
		duplicate_found = False

		for fl in filtered_lines:
			x1_fl, y1_fl, x2_fl, y2_fl = fl[0]

			# Calculate rho and theta for the current line
			rho = np.linalg.norm(np.cross(np.array([x2, y2]) - np.array([x1, y1]), np.array([x1, y1]) - np.array([0, 0]))) / np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
			theta = np.arctan2(y2 - y1, x2 - x1)

			# Calculate rho and theta for the filtered line
			rho_fl = np.linalg.norm(np.cross(np.array([x2_fl, y2_fl]) - np.array([x1_fl, y1_fl]), np.array([x1_fl, y1_fl]) - np.array([0, 0]))) / np.linalg.norm(np.array([x2_fl, y2_fl]) - np.array([x1_fl, y1_fl]))
			theta_fl = np.arctan2(y2_fl - y1_fl, x2_fl - x1_fl)

			if (abs(rho - rho_fl) < delta_rho and abs(theta - theta_fl) < delta_theta) or (orientation == 'v' and abs(x1 - x1_fl) <= tolerance) or (orientation == 'h' and abs(y1 - y1_fl) <= tolerance):
				duplicate_found = True
				break

		if not duplicate_found:
			filtered_lines.append(line)

	return filtered_lines


def crop_center(image, ratio=0.8):
	h, w = image.shape
	ch, cw = int(h * ratio), int(w * ratio)
	y_offset, x_offset = (h - ch) // 2, (w - cw) // 2
	return image[y_offset:y_offset + ch, x_offset:x_offset + cw]


def create_sudoku_image(problem, img, grid_size):
	block_size = int(grid_size ** (1 / 2))
	# Create a blank image to draw the numbers on
	image = np.zeros((500, 500, 3), dtype=np.uint8)
	image.fill(255)  # Fill the image with white

	# Calculate the size of each cell
	cell_size = image.shape[0] // grid_size

	line_thickness = 2

	# Draw the grid lines
	for i in range(grid_size + 1):
		thickness = line_thickness if i % block_size == 0 else 1
		x1 = i * cell_size
		y1 = 0
		x2 = i * cell_size
		y2 = image.shape[0]
		cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness)
		x1 = 0
		y1 = i * cell_size
		x2 = image.shape[0]
		y2 = i * cell_size
		cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), thickness)

	# Loop through each cell of the grid and draw the number
	for i in range(grid_size):
		for j in range(grid_size):
			if problem[i][j] != ' ':
				# Calculate the position of the number
				x = j * cell_size + cell_size // 2
				y = i * cell_size + cell_size // 2

				# Determine the color of the number
				colour = (0, 0, 0) if problem[i][j] == img[i][j] else (0, 180, 0)
				# if problem[i][j] != img[i][j]:
					# print(f"Not equal: {problem[i][j]}, {img[i][j]}")
					# print(f"Types: {type(problem[i][j])}, {type(img[i][j])}")

				# Write the number on the image
				font = cv2.FONT_HERSHEY_SIMPLEX
				font_scale = 1
				font_thickness = 2
				text_size = cv2.getTextSize(str(problem[i][j]), font, font_scale, font_thickness)[0]
				text_x = x - text_size[0] // 2
				text_y = y + text_size[1] // 2
				cv2.putText(image, str(problem[i][j]), (text_x, text_y), font, font_scale, colour, font_thickness)

	return image


def centre_character(box):
	y, x = np.where(box == 255)
	center_of_mass = np.mean(y), np.mean(x)

	# Calculate the translation required to move the center of mass to the center of the image
	image_center = (box.shape[0] // 2, box.shape[1] // 2)
	translation_y = image_center[0] - center_of_mass[0]
	translation_x = image_center[1] - center_of_mass[1]

	# Apply the translation
	translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
	box_centred_character = cv2.warpAffine(box, translation_matrix, (box.shape[1], box.shape[0]))

	return box_centred_character


# define a function to count the most common element in an array
# define a function to count the most common element in an array
def most_common(characters, arr):
	filtered_arr = [[x for x in list(elem) if x in characters] for elem in arr]  # take away extra characters
	if len(filtered_arr) > 0:
		filtered_arr = [x[0] for x in filtered_arr if len(x) > 0 ]  # in case a list is left (two possible characters were predicted in the same spot)
		# print(filtered_arr)
		if len(filtered_arr) == 0:
			return ' '
		counts = Counter(filtered_arr)
		return counts.most_common(1)[0][0]
	else:
		return ' '


def display_image(im, title='Image', wait=0):
	im = cv2.resize(im, (600, 600))
	cv2.imshow(title, im)
	cv2.waitKey(wait)
	cv2.destroyAllWindows()


# used to supress output
@contextlib.contextmanager
def suppress_stdout():
	original_stdout = sys.stdout
	sys.stdout = io.StringIO()
	try:
		yield
	finally:
		sys.stdout = original_stdout


def identify_character(box, reader, reader2):
	# crop box
	box = crop_center(box)

	# Resize the box
	box = cv2.resize(box, (50, 50))

	# Thresholding
	_, box = cv2.threshold(box, 100, 255, cv2.THRESH_BINARY_INV)

	# Denoising
	box = cv2.fastNlMeansDenoising(box, None, 10, 7, 21)

	# Dilate the image
	kernel = np.ones((2, 2), np.uint8)
	box = cv2.dilate(box, kernel, iterations=1)
	box = cv2.erode(box, kernel / 2)

	# Count number of white pixels
	centred_box = centre_character(box)
	num_white_pixels = cv2.countNonZero(centred_box)

	if num_white_pixels > THRESHOLD:    # means a character is contained in this box
		text_rec = ' '
		count = 0
		# if none of the methods can predict anything or they predict something impossible
		while text_rec in [' ', '', []] and count < 2:
			text_rec = reader.readtext(box)

			if len(text_rec) > 0:
				text_rec = text_rec[0][-2]
			else:
				text_rec = ' '

			if text_rec == ' ':    # try another text rec
				text_rec = pytesseract.image_to_string(box, config=CUSTOM_CONFIG)

			if text_rec == ' ' or text_rec == '':
				box2 = cv2.cvtColor(box, cv2.COLOR_GRAY2RGB)
				with suppress_stdout():
					text_rec = reader2.recognize(box2)


			box = centre_character(box)
			box = cv2.morphologyEx(box, cv2.MORPH_OPEN, kernel/2)

			if count > 0:
				top = 5
				bottom = 5
				left = 5
				right = 5
				box = cv2.copyMakeBorder(box, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
				# cv2.putText(box, text_rec, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
				# cv2.imshow('Cell With Text', box)
				# cv2.waitKey(400)
				# cv2.destroyAllWindows()
			count += 1
	else:
		text_rec = ' '

	# cv2.putText(box, text_rec, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
	# cv2.imshow('Cell With Text', box)
	# cv2.waitKey(400)
	# cv2.destroyAllWindows()

	# fix 0's being read as o's
	if text_rec == 'o' or text_rec == 'O':
		text_rec = '0'
	return text_rec


def process_box(box):
	return identify_character(box, reader, reader2)


def edit_prediction_gui(original, grid_size):
	size = 700
	# Create a new window
	window = tk.Tk()
	window.title('Sudoku Problem')

	box_size = int(grid_size ** (1/2))

	# Define the size of the window
	window.geometry('{}x{}'.format(size, size+50))

	# Create a grid_size x grid_size grid of input fields
	input_fields = []
	for i in range(box_size):
		for j in range(box_size):
			box = []
			box_frame = tk.Frame(window, borderwidth=2, relief="solid")
			box_frame.grid(row=i, column=j, padx=3, pady=3)

			for x in range(box_size):
				for y in range(box_size):
					# Get the value from the original array, or leave blank if None
					value = original[i * box_size + x][j * box_size + y]
					if value is None:
						value = ''
					else:
						value = str(value)

					input_field = tk.Text(box_frame, width=2, height=1, font=('Helvetica', int(70 - box_size*12)), borderwidth=1)
					input_field.insert('1.0', value)
					input_field.tag_configure('center', justify='center')
					input_field.tag_add('center', '1.0', 'end')
					input_field.grid(row=x, column=y, padx=1, pady=1)
					box.append(input_field)

			box = np.reshape(box, (box_size, box_size))
			input_fields.append(box)


	# Create a Solve button
	solve_button = tk.Button(window, text='Solve', font=('Helvetica', 20), width=12, height=1, bg='#28a745', fg='white', activebackground='#218838', activeforeground='white')
	if box_size % 3 == 0:
		solve_button.grid(row=grid_size, column=1)  # Adjust the column value
	else:
		solve_button.grid(row=grid_size, column=int(math.log(box_size, 4)), columnspan=2)  # Adjust the column value

	print(int(math.log(box_size, 4)))
	to_solve = np.zeros_like(original)

	# Reshape the input_fields list into an array with the shape (grid_size, grid_size)
	# Convert the list of boxes into a box_size^4 array
	input_fields = np.array(input_fields)
	input_fields = np.reshape(input_fields, (box_size, box_size, box_size, box_size))
	input_fields = np.concatenate(np.concatenate(input_fields, axis=1), axis=1)

	# read from the grid back to the original array
	def save_values():
		# Loop through the input fields and update the original array
		for i in range(grid_size):
			for j in range(grid_size):
				value = input_fields[i][j].get('1.0', 'end-1c')  # Access input_fields using i and j directly
				if value == '':
					to_solve[i][j] = ''
				else:
					to_solve[i][j] = value

		# Close the window
		window.destroy()

	# Bind the Save button to the save_values function
	solve_button.config(command=save_values)

	# Start the main loop
	window.mainloop()

	return to_solve


# solves the problem given either a numpy array of the problem or an image of the problem and a boolean input
# indicating whether the input was manual or not
def solve_sudoku_upload():
	t1 = time.time()

	# Specify the directory
	directory = 'uploads'

	# Get a list of all the files in the directory
	files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

	# Check if there are any files in the directory
	if files:
		# Get the most recently modified file
		latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))

		# Get the path to the most recently modified file
		problem = os.path.join(directory, latest_file)

	else:
		print(f'No files found in {directory}.')
		sys.exit()

	image = cv2.imread(problem)
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Apply Gaussian blur
	blur = cv2.GaussianBlur(gray, (5, 5), 0)

	# Apply Canny edge detection
	edges = cv2.Canny(blur, 50, 150, apertureSize=3)
	# edges = cv2.Canny(blur, 100, 200, apertureSize=3)

	lines = cv2.HoughLinesP(edges, rho=1, theta=1*np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

	# Separate horizontal and vertical lines
	horizontal_lines = []
	vertical_lines = []

	for line in lines:
		x1, y1, x2, y2 = line[0]
		if abs(y2 - y1) < abs(x2 - x1):  # horizontal lines
			horizontal_lines.append(line)
		else:  # vertical lines
			vertical_lines.append(line)

	# Remove duplicate horizontal and vertical lines
	horizontal_lines = filter_duplicate_lines(horizontal_lines, 'h', delta_rho=10, delta_theta=np.pi / 90)
	vertical_lines = filter_duplicate_lines(vertical_lines, 'v', delta_rho=10, delta_theta=np.pi / 90)

	# filter outliers
	horizontal_lines = filter_spatial_outliers(image, horizontal_lines, 'h')
	vertical_lines = filter_spatial_outliers(image, vertical_lines, 'v')

	for i, line in enumerate(horizontal_lines):
		x1, y1, x2, y2 = line[0]
		cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

	for i, line in enumerate(vertical_lines):
		x1, y1, x2, y2 = line[0]
		cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

	# Find line intersections (grid corners)
	intersections = []
	for h_line in horizontal_lines:
		for v_line in vertical_lines:
			intersection = line_intersection(h_line, v_line)
			intersections.append(intersection)

	# Sort intersections by row and column
	intersections = sorted(intersections, key=lambda x: (x[1], x[0]))

	num_intersections = len(intersections)

	num_rows = int(num_intersections ** (1/2))
	grid_size = num_rows - 1
	rows = [intersections[i:i + num_rows] for i in range(0, len(intersections), num_rows)]

	# Extract boxes from the grid
	box_images = []
	for i in range(num_rows - 1):
		for j in range(num_rows - 1):
			x1, y1 = rows[i][j]
			x2, y2 = rows[i + 1][j + 1]
			box_image = gray[y1:y2, x1:x2]
			box_images.append(box_image)

	original_stdout = sys.stdout
	sys.stdout = io.StringIO()

	with concurrent.futures.ThreadPoolExecutor() as executor:
		# Parallelize the processing of box_images using threads
		text = list(tqdm(executor.map(process_box, box_images), total=len(box_images)))

	sys.stdout = original_stdout

	characters = POSSIBLE_CHARACTERS[0:grid_size]
	if grid_size < 16:
		characters = [str(int(ch) + 1) for ch in characters]

	text = [[x for x in list(elem) if x in characters] for elem in text]  # take away extra characters
	text = [x[0] if len(x) > 0 else ' ' for x in text]  # in case a list is left (two possible characters were predicted in the same spot)

	problem = np.reshape(text, (grid_size, grid_size))

	solve_start = time.time()

	block_size = int(grid_size ** (1 / 2))
	while ' ' in np.unique(problem):
		problem = solve(problem, characters, block_size)
		solve_end = time.time()

		if (solve_end - solve_start) > max_time(grid_size):
			print("Couldn't solve...")
			sys.exit()

	problem_solved = problem
	# t2 = time.time()
	# solve_end = time.time()
	return problem_solved


def solve_sudoku_manual_input(problem):
	grid_size = problem.shape[0]
	characters = POSSIBLE_CHARACTERS[0:grid_size]
	if grid_size < 16:
		characters = [str(int(ch) + 1) for ch in characters]

	solve_start = time.time()

	block_size = int(grid_size ** (1 / 2))
	while ' ' in np.unique(problem):
		problem = solve(problem, characters, block_size)
		solve_end = time.time()

		if (solve_end - solve_start) > max_time(grid_size):
			print("Couldn't solve...")
			sys.exit()

	problem_solved = problem
	# t2 = time.time()
	# solve_end = time.time()
	return problem_solved


