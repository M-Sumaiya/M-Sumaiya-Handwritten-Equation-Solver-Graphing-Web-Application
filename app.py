from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from solver_core import recognize_and_solve_two_images  # Move your solution code here
import uuid
from solver_core import recognize_and_solve_two_images


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve_equations():
    eq1 = request.files['eq1']
    eq2 = request.files['eq2']

    id1 = str(uuid.uuid4()) + "_eq1.png"
    id2 = str(uuid.uuid4()) + "_eq2.png"
    path1 = os.path.join(app.config['UPLOAD_FOLDER'], id1)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], id2)

    eq1.save(path1)
    eq2.save(path2)

    equations, solution = recognize_and_solve_two_images(path1, path2)
    
    return jsonify({
        'equation1': equations[0],
        'equation2': equations[1],
        'solution': str(solution) if solution else "No solution found",
        'graph_url': '/graph'
    })

@app.route('/graph')
def graph():
    return send_file('static/graph.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
