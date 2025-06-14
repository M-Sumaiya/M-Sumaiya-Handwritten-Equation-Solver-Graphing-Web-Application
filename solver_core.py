import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ Set non-GUI backend FIRST

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sympy import symbols, Eq, solve
import re



model = load_model("model/digit_recognition_model_new_2.h5")
def recognize_and_solve_two_images(img_path1, img_path2):
    expr1, img1 = recognize_equation_from_image(img_path1)
    expr2, img2 = recognize_equation_from_image(img_path2)

    print(f"\n📘 Equation 1: {expr1}")
    print(f"📘 Equation 2: {expr2}")

    # Fix sizes and types
    img1, img2 = resize_to_same_height(img1, img2)
    img1, img2 = ensure_same_type(img1, img2)
    combined = cv2.hconcat([img1, img2])

    # Save graph image instead of showing it
    v, y = symbols('v y')
    try:
        if '=' not in expr1:
            expr1 += '=0'
        if '=' not in expr2:
            expr2 += '=0'

        lhs1, rhs1 = expr1.split('=')
        lhs2, rhs2 = expr2.split('=')

        lhs1 = insert_multiplication(lhs1)
        rhs1 = insert_multiplication(rhs1)
        lhs2 = insert_multiplication(lhs2)
        rhs2 = insert_multiplication(rhs2)

        eq1 = Eq(eval(lhs1), eval(rhs1))
        eq2 = Eq(eval(lhs2), eval(rhs2))

        print(f"🧠 Parsed Equations:\n{eq1}\n{eq2}")
        solutions = solve((eq1, eq2), (v, y), dict=True)

        if solutions:
            sol = solutions[0]
            y_expr1 = solve(eq1, y)
            y_expr2 = solve(eq2, y)

            if y_expr1 and y_expr2:
                v_vals = np.linspace(sol[v] - 10, sol[v] + 10, 400)
                y_vals1 = [float(y_expr1[0].subs(v, val)) for val in v_vals]
                y_vals2 = [float(y_expr2[0].subs(v, val)) for val in v_vals]

                plt.figure(figsize=(8, 6))
                plt.plot(v_vals, y_vals1, label=str(eq1), color='blue')
                plt.plot(v_vals, y_vals2, label=str(eq2), color='green')
                plt.scatter(sol[v], sol[y], color='red', s=80, label=f"Solution: (v={sol[v]}, y={sol[y]})")
                plt.title("Graphical Solution of the System")
                plt.xlabel("v")
                plt.ylabel("y")
                plt.axhline(0, color='black', lw=1)
                plt.axvline(0, color='black', lw=1)
                plt.legend()
                plt.grid(True)
                plt.savefig("static/graph.png")  # Save to static folder
                plt.close()
        else:
            print("❌ No solution or invalid format.")
            solutions = None

    except Exception as e:
        print(f"❌ Error parsing or solving equations: {e}")
        solutions = None

    return (expr1, expr2), solutions


# Label mapping (adjust if your model differs)
class_labels_dict = {
    0: '()',  1: ')',  2: '+',  3: '-',  4: '0',  5: '1',
    6: '2',  7: '3',  8: '4',  9: '5',  10: '6', 11: '7',
    12: '8', 13: '9', 14: '=', 15: 'div', 16: 'times',
    17: 'v', 18: 'w', 19: 'y'
}

MINUS_CONFIDENCE_THRESHOLD = 0.85

def preprocess_image(img, target_size=(64, 64)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def segment_characters(image, padding=10, target_size=(64, 64)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Improve segmentation with dilation and noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    characters = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 100:
            continue
        char_img = thresh[y:y+h, x:x+w]
        char_img = cv2.copyMakeBorder(char_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        char_img = preprocess_image(char_img, target_size)
        characters.append((x, char_img))
        bounding_boxes.append((x, y, w, h))

    characters.sort(key=lambda item: item[0])
    bounding_boxes.sort(key=lambda item: item[0])
    return characters, bounding_boxes

def recognize_equation_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image.")

    characters, boxes = segment_characters(image)

    if not characters:
        print("❌ No characters detected.")
        return "", image

    chars, confidences = [], []

    for x, char_img in characters:
        prediction_probs = model.predict(char_img)[0]
        predicted_index = np.argmax(prediction_probs)
        predicted_label = class_labels_dict[predicted_index]
        confidence_score = prediction_probs[predicted_index]

        # Handle minus override
        minus_prob = prediction_probs[3]
        if minus_prob > MINUS_CONFIDENCE_THRESHOLD and predicted_label != '-':
            predicted_label = '-'
            confidence_score = minus_prob

        chars.append(predicted_label)
        confidences.append(confidence_score)

    # Convert 'times' and 'div'
    expression = ''.join(['*' if c == 'times' else '/' if c == 'div' else c for c in chars])
    return expression, image

def insert_multiplication(expr):
    expr = expr.replace('()', '')
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expr)
    return expr

def resize_to_same_height(img1, img2):
    h1, h2 = img1.shape[0], img2.shape[0]
    if h1 != h2:
        target_height = max(h1, h2)
        img1 = cv2.resize(img1, (int(img1.shape[1] * (target_height / h1)), target_height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * (target_height / h2)), target_height))
    return img1, img2

def ensure_same_type(img1, img2):
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    return img1, img2
