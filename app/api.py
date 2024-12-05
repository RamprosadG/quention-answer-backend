from flask import Blueprint, request, jsonify
from app.model import get_answer

api_bp = Blueprint('api', __name__)

@api_bp.route('/ask', methods=['POST'])
def answer_question():
    data = request.get_json()

    question = data.get('question')
    context = data.get('context', "")  # Optional context

    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = get_answer(question, context)
    return jsonify({"answer": answer})
