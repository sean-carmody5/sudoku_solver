const manualInputBtn = document.getElementById('manual-input-btn');
const photoUploadBtn = document.getElementById('photo-upload-btn');
const solveBtn = document.getElementById('solve-btn');
const inputSection = document.getElementById('input-section');
const solutionDiv = document.getElementById('solution');
const highlight = document.querySelector('.selector-highlight');



let inputType = 'photo'; // Default to 'photo'

window.addEventListener('DOMContentLoaded', (event) => {
    highlight.style.left = '50%';
    showPhotoUpload();
});


manualInputBtn.addEventListener('click', () => {
	clearSolutionAndMessage();
	console.log('Selected manual input');  // Debugging statement
    inputType = 'manual';
    highlight.style.left = '0';
    showManualInput();
});

photoUploadBtn.addEventListener('click', () => {
	console.log('Selected photo upload');  // Debugging statement
    inputType = 'photo';
    highlight.style.left = '50%';
    showPhotoUpload();
});

solveBtn.addEventListener('click', async () => {
	clearSolutionAndMessage();
    console.log('Solve button clicked');  // Debugging statement
    const board = getBoardData();  
    const manual_input = (inputType === 'manual');
    
    const response = await fetch('/solve', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            board: board,
            manual_input: manual_input,
        }),
    });
    
    const data = await response.json();

    // Check if the response was successful
    if (data.success) {
        // Only generate the grid if in photo upload mode
        if (!manual_input) {
			console.log("Attempting to generate grid for photo upload");
            let gridSize = data.grid_size;
            generateGrid(gridSize);  // Generate the Sudoku grid based on the received size
        }
        displaySolvedBoard(data.solution);  // Populate the grid with the solution
    } else {
        alert('Failed to solve the Sudoku: ' + data.error);
    }
});



function showManualInput() {
    let inputSection = document.getElementById('input-section');
    inputSection.innerHTML = `
        <div class="grid-size-selector">
            <button id="grid-4x4-btn">4x4</button>
            <button id="grid-9x9-btn">9x9</button>
            <button id="grid-16x16-btn">16x16</button>
        </div>
        <div id="sudoku-grid-container"></div>
    `;

	document.getElementById('grid-4x4-btn').addEventListener('click', () => {
	updateActiveButton('grid-4x4-btn');
	generateGrid(4);
    });
    document.getElementById('grid-9x9-btn').addEventListener('click', () => {
        updateActiveButton('grid-9x9-btn');
        generateGrid(9);
    });
    document.getElementById('grid-16x16-btn').addEventListener('click', () => {
        updateActiveButton('grid-16x16-btn');
        generateGrid(16);
    });
}

function updateActiveButton(buttonId) {
    document.querySelectorAll('.grid-size-selector button').forEach(button => {
        button.classList.remove('active');
    });
    document.getElementById(buttonId).classList.add('active');
}

function generateGrid(gridSize) {
	console.log("Generating grid with size:", gridSize);
    let gridContainer = document.getElementById('sudoku-grid-container');
	
	    // Check if gridContainer is not null
    if (!gridContainer) {
        console.error("sudoku-grid-container not found in the DOM.");
        return;
    }
	
	// Clear any existing grid
    gridContainer.innerHTML = '';
	
    let boxSize = Math.sqrt(gridSize);

    // Determine cell size and gap based on grid size
    let cellSize, gridGap;
    if (gridSize === 4) {
        cellSize = '7vmin';
        gridGap = '2px';
    } else if (gridSize === 16) {
        cellSize = '3.5vmin';
        gridGap = '1px';
    } else {
        cellSize = '5vmin'; // Default size for 9x9
        gridGap = '1.5px';
    }

    document.documentElement.style.setProperty('--grid-size', gridSize);
    document.documentElement.style.setProperty('--box-size', boxSize);
    document.documentElement.style.setProperty('--cell-size', cellSize);
    document.documentElement.style.setProperty('--grid-gap', gridGap);
    document.documentElement.style.setProperty('--grid-size-factor', Math.sqrt(gridSize));
    
    let gridHTML = `<div class="sudoku-grid size-${gridSize}x${gridSize}">`;
    for(let i = 0; i < gridSize; i++) {
        gridHTML += '<div class="sudoku-row">';
        for(let j = 0; j < gridSize; j++) {
            let boxClass = ((Math.floor(i / boxSize) + Math.floor(j / boxSize)) % 2 === 0) ? 'box1' : 'box2';
            gridHTML += `<input type="text" maxlength="1" class="sudoku-cell ${boxClass}" style="${getBorderStyle(i, j, boxSize)}" />`;
        }
        gridHTML += '</div>';
    }
    gridHTML += '</div>';
    gridContainer.innerHTML = gridHTML;
}



function getBorderStyle(row, col, boxSize) {
    let styles = [];
    if (row % boxSize === 0 && row !== 0) {
        styles.push('border-top: 4px solid #000');
    }
    if (col % boxSize === 0 && col !== 0) {
        styles.push('border-left: 4px solid #000');
    }
    return styles.join('; ');
}


function showPhotoUpload() {
    inputSection.innerHTML = `
        <form id="upload-form" action="/upload" method="POST" enctype="multipart/form-data">
            <input type="file" name="file">
            <button type="submit">Upload</button>
        </form>
        <div id="upload-success" style="display: none;">
            <span style="color: green;">✔ File uploaded successfully</span>
        </div>
    `;

    $('#upload-form').on('submit', function(e) {
        e.preventDefault();

        var formData = new FormData(this);

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            success: function(response) {
                if(response.success) {
                    $('#upload-success').show();
                } else {
                    alert('File upload failed: ' + response.error);
                }
            },
            cache: false,
            contentType: false,
            processData: false
        });
    });
	
	// Add an empty sudoku-grid-container
    let gridContainer = document.createElement('div');
    gridContainer.id = 'sudoku-grid-container';
    let container = document.querySelector('.container');
	container.insertBefore(gridContainer, solveBtn);

}


function getBoardData() {
    const gridSize = Math.sqrt(document.querySelectorAll('.sudoku-cell').length);
    let board = Array(gridSize).fill().map(() => Array(gridSize).fill(' '));
    document.querySelectorAll('.sudoku-row').forEach((row, i) => {
        row.querySelectorAll('.sudoku-cell').forEach((cell, j) => {
            board[i][j] = cell.value ? parseInt(cell.value) : ' ';
        });
    });
    return board;
}


function displaySolvedBoard(solvedBoard) {
	console.log("Populating grid with solution:", solvedBoard);
    document.querySelectorAll('.sudoku-row').forEach((row, i) => {
        row.querySelectorAll('.sudoku-cell').forEach((cell, j) => {
            cell.value = solvedBoard[i][j];
        });
    });
}

function clearSolutionAndMessage() {
    // Clear the solution grid
    let gridContainer = document.getElementById('sudoku-grid-container');
    if (gridContainer) {
        gridContainer.innerHTML = '';
    }

    // Hide the success message
    let successMessage = document.getElementById('upload-success');
    if (successMessage) {
        successMessage.style.display = 'none';
    }
}
