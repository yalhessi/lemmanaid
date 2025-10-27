from flask import Flask, request, jsonify

app = Flask(__name__)

# Dummy function for generating templates
def generate_templates(function_names):
    return [f"Template for {name}" for name in function_names]

@app.route("/generate_templates", methods=["POST"])
def handle_generate_templates():
    data = request.json  # Parse JSON body
    
    function_names = data.get("function_names", [])
    
    if not isinstance(function_names, list):
        return jsonify({"error": "function_names must be a list"}), 400
    
    # TODO ACTUALLY GENERATE TEMPLATES WITH THE LLM HERE
    templates = generate_templates(function_names)
    
    return jsonify({"templates": templates})

if __name__ == "__main__":
    app.run()
